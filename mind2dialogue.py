from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mind2dialogue")

ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILES = ROOT / "data" / "filterd_refined_profiles" / "summary_refined_profiles_us.jsonl"
DEFAULT_PROMPTS = ROOT / "data" / "rewritten_prompts" / "original_rewritten_selected_prompts_us.jsonl"
SCENARIOS_DIR = ROOT / "data" / "deep_scenarios"

QA_STYLES = ["personamem_mcq", "prefeval_gen", "bigtom_tom", "lamp_cls"]

_SCENARIO_CONSTRUCTORS = {
    "lifelong": "simulator_lifelong_scenario_constructor",
    "highfreq": "simulator_highfreq_scenario_constructor",
    "affective": "simulator_affective_scenario_constructor",
    "concerning": "simulator_concerning_scenario_constructor",
}
_CONSTRUCTOR_PREFIX = {v: k for k, v in _SCENARIO_CONSTRUCTORS.items()}


def personas_by_id(path: Path | str | None) -> dict[str, Any]:
    from user_simulator.data import load_personas

    return {p.id: p for p in load_personas(Path(path) if path else DEFAULT_PROFILES)}


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    out: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if limit is not None and len(out) >= limit:
            break
    return out


async def gather_bounded(
    factories: Sequence[Callable[[], Awaitable[Any]]], concurrency: int
) -> list[Any]:
    sem = asyncio.Semaphore(concurrency)

    async def _run(make: Callable[[], Awaitable[Any]]) -> Any:
        async with sem:
            return await make()

    return await asyncio.gather(*[_run(f) for f in factories])


class JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")
        self._lock = asyncio.Lock()
        self.path = path
        self.n = 0

    async def write(self, obj: dict) -> None:
        async with self._lock:
            self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._fh.flush()
            self.n += 1

    def close(self) -> None:
        self._fh.close()


def _out_dirs(output_dir: str | None) -> tuple[Path, Path]:
    from user_simulator.data import CONV_DIR, SFT_DIR

    if output_dir:
        base = Path(output_dir)
        return base / "conversations", base / "sft"
    return CONV_DIR, SFT_DIR


async def cmd_rollout(args: argparse.Namespace) -> None:
    from user_simulator.ablation import AblationConfig
    from user_simulator.data import LLM, SIM_MODEL, save_json
    from user_simulator.sft import build_sft_instance
    from user_simulator.simulator import rollout_conversation

    config = AblationConfig.from_name(args.ablation)
    conv_root, sft_root = _out_dirs(args.output_dir)

    persona_map = personas_by_id(args.profiles)
    logger.info("Loaded %d personas", len(persona_map))

    id_set = set(args.persona_ids) if args.persona_ids else None
    prompts = read_jsonl(Path(args.prompts_jsonl) if args.prompts_jsonl else DEFAULT_PROMPTS)
    if id_set:
        prompts = [d for d in prompts if d.get("persona_id") in id_set]
    if args.max_prompts:
        per = defaultdict(list)
        for d in prompts:
            per[d["persona_id"]].append(d)
        prompts = [d for pid in sorted(per) for d in per[pid][: args.max_prompts]]

    tasks = [(persona_map[d["persona_id"]], d) for d in prompts if d.get("persona_id") in persona_map]
    logger.info("[%s] %d rollouts, concurrency=%d", config.name, len(tasks), args.concurrency)

    llm = LLM(model=SIM_MODEL, max_concurrent=args.concurrency)
    conv_dir = conv_root / config.name
    sft = JsonlWriter(sft_root / f"train_{config.name}.jsonl")
    counter = {"done": 0, "skipped": 0, "failed": 0, "total": len(tasks)}

    async def rollout_one(persona, prompt_data) -> None:
        prompt_id = prompt_data.get("prompt_id", "unknown")
        initial = prompt_data.get("rewritten") or prompt_data.get("prompt_text", "")
        safe_id = prompt_id.replace("/", "_").replace("\\", "_")
        conv_path = conv_dir / persona.id / f"{safe_id}.json"
        if not initial or conv_path.exists():
            counter["skipped"] += 1
            return
        try:
            session = await rollout_conversation(
                persona, initial, prompt_id, llm,
                max_turns=args.max_turns, min_turns=args.min_turns, config=config,
            )
            session["profile_summary"] = persona.refined_summary
            session["behavioral_metadata"] = persona.behavioral_metadata
            save_json(session, conv_path)
            instance = build_sft_instance(session, config)
            if instance:
                await sft.write(instance)
            counter["done"] += 1
            if counter["done"] % 10 == 0:
                logger.info("[%s] %d/%d done", config.name, counter["done"], counter["total"])
        except Exception as e:  # noqa: BLE001
            counter["failed"] += 1
            logger.error("[%s/%s] %s: %s", config.name, persona.id, type(e).__name__, e)

    try:
        await gather_bounded([lambda p=p, d=d: rollout_one(p, d) for p, d in tasks], args.concurrency)
    finally:
        sft.close()
    logger.info("[%s] done=%d skipped=%d failed=%d | conv→%s | sft→%s | %s",
                config.name, counter["done"], counter["skipped"], counter["failed"],
                conv_dir, sft.path, llm.stats)


async def _build_scenarios(persona, llm, tmpl, config, prefix, cache_dir: Path, force: bool):
    from user_simulator.prompts import render

    cache_path = cache_dir / f"{persona.id}__{prefix}.json"
    legacy_path = cache_dir / f"{persona.id}.json"
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    if legacy_path.exists() and not force and prefix == "lifelong":
        return json.loads(legacy_path.read_text(encoding="utf-8"))

    summary = persona.refined_summary or persona.summary
    bm = (json.dumps(persona.behavioral_metadata, indent=2, ensure_ascii=False)
          if persona.behavioral_metadata else "N/A")
    prompt = render(tmpl, profile_summary=summary, profile_block=summary,
                    behavior_metadata=bm, persona_id=persona.id)
    data = await llm.chat_json(
        [{"role": "system", "content": prompt},
         {"role": "user", "content": "Generate the scenarios JSON now."}],
        temperature=config.scenario_constructor_temperature,
        max_tokens=config.scenario_constructor_max_tokens,
    )
    scenarios = data.get("scenarios", []) if isinstance(data, dict) else []
    for i, s in enumerate(scenarios):
        sid = (s.get("scenario_id") or f"{persona.id}_scenario_{i}").replace("{persona_id}", persona.id)
        s["scenario_id"] = sid if sid.startswith(f"{prefix}_") else f"{prefix}_{sid}"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(scenarios, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Constructed %d scenarios for %s (%s)", len(scenarios), persona.id, prefix)
    return scenarios


async def cmd_scenario(args: argparse.Namespace) -> None:
    from user_simulator.ablation import AblationConfig
    from user_simulator.data import LLM, SIM_MODEL, save_json
    from user_simulator.prompts import load_prompt
    from user_simulator.sft import build_sft_instance
    from user_simulator.simulator import rollout_conversation

    config = AblationConfig.from_name(args.ablation)
    conv_root, sft_root = _out_dirs(args.output_dir)
    constructor = _SCENARIO_CONSTRUCTORS.get(args.constructor, args.constructor)
    prefix = _CONSTRUCTOR_PREFIX.get(constructor,
                                     constructor.replace("simulator_", "").replace("_scenario_constructor", ""))
    tmpl = load_prompt(constructor)

    persona_map = personas_by_id(args.profiles)
    personas = list(persona_map.values())
    if args.persona_ids:
        keep = set(args.persona_ids)
        personas = [p for p in personas if p.id in keep]
    logger.info("Processing %d personas (constructor=%s)", len(personas), prefix)

    llm = LLM(model=SIM_MODEL, max_concurrent=args.concurrency)
    run_tag = f"deep_{config.name}"
    conv_dir = conv_root / run_tag

    logger.info("Phase 1: constructing scenarios (cache: %s)", SCENARIOS_DIR)
    per_persona = await asyncio.gather(*[
        _build_scenarios(p, llm, tmpl, config, prefix, SCENARIOS_DIR, args.force_reconstruct)
        for p in personas
    ])

    tasks = []
    for persona, scenarios in zip(personas, per_persona):
        for s in (scenarios[: args.max_scenarios] if args.max_scenarios else scenarios):
            tasks.append((persona, s))
    logger.info("Phase 2 [%s]: %d rollouts, concurrency=%d", config.name, len(tasks), args.concurrency)

    sft = JsonlWriter(sft_root / f"train_{run_tag}.jsonl")
    counter = {"done": 0, "skipped": 0, "failed": 0, "total": len(tasks)}

    async def rollout_one(persona, scenario) -> None:
        scenario_id = scenario.get("scenario_id", "unknown")
        initial = scenario.get("initial_prompt", "")
        safe_id = scenario_id.replace("/", "_").replace("\\", "_")
        conv_path = conv_dir / persona.id / f"{safe_id}.json"
        if not initial or conv_path.exists():
            counter["skipped"] += 1
            return
        try:
            session = await rollout_conversation(
                persona, initial, scenario_id, llm,
                max_turns=args.max_turns, min_turns=args.min_turns, config=config,
            )
            session.update(
                profile_summary=persona.refined_summary,
                behavioral_metadata=persona.behavioral_metadata,
                scenario_category=scenario.get("category", ""),
                scenario_context_note=scenario.get("context_note", ""),
                initial_prompt=initial,
                source="deep_scenario",
            )
            save_json(session, conv_path)
            instance = build_sft_instance(session, config)
            if instance:
                await sft.write(instance)
            counter["done"] += 1
            if counter["done"] % 10 == 0:
                logger.info("[%s] %d/%d done", run_tag, counter["done"], counter["total"])
        except Exception as e:  # noqa: BLE001
            counter["failed"] += 1
            logger.error("[%s/%s] %s: %s", run_tag, persona.id, type(e).__name__, e)

    try:
        await gather_bounded([lambda p=p, s=s: rollout_one(p, s) for p, s in tasks], args.concurrency)
    finally:
        sft.close()
    logger.info("[%s] done=%d skipped=%d failed=%d | conv→%s | sft→%s | %s",
                run_tag, counter["done"], counter["skipped"], counter["failed"],
                conv_dir, sft.path, llm.stats)


def _summarize_qc(results_path: Path, summary_path: Path) -> dict:
    tier_counts: Counter[str] = Counter()
    fail_counts: Counter[str] = Counter()
    d5_dist: Counter[Any] = Counter()
    d6_dist: Counter[str] = Counter()
    n = 0
    for r in read_jsonl(results_path):
        n += 1
        tier_counts[r.get("tier", "C")] += 1
        for d in r.get("failed_dims") or []:
            fail_counts[d] += 1
        d5_dist[r.get("d5_persona_consistency", "null")] += 1
        d6_dist[r.get("d6_conflict") or "null"] += 1
    summary = {
        "total": n,
        "tier_counts": dict(tier_counts),
        "tier_pass_rate": tier_counts["A"] / n if n else 0.0,
        "failed_dim_counts": dict(fail_counts),
        "d5_distribution": {str(k): v for k, v in d5_dist.items()},
        "d6_distribution": dict(d6_dist),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


async def cmd_qc(args: argparse.Namespace) -> None:
    import os

    from user_simulator.data import LLM, load_json
    from user_simulator.qc import score_conversation

    conv_dir = Path(args.conversations_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "qc_results.jsonl"

    profiles = personas_by_id(args.profiles_jsonl) if args.profiles_jsonl else {}
    if not profiles:
        logger.warning("No profiles provided; the profile-binding check (D4) will fail for every conv.")

    conv_files = sorted(conv_dir.rglob("*.json"))
    if args.sample:
        conv_files = conv_files[: args.sample]
    logger.info("Found %d conversation JSONs in %s", len(conv_files), conv_dir)

    existing = {(r.get("persona_id", ""), r.get("scenario_id", "")) for r in read_jsonl(results_path)} \
        if results_path.exists() else set()
    if existing:
        logger.info("Resuming: %d existing results", len(existing))

    llm = None
    if not args.skip_judges:
        judge_model = os.getenv("JUDGE_MODEL") or args.judge_model
        llm = LLM(model=judge_model, max_concurrent=args.concurrency, log_calls=args.log_calls)
        logger.info("Judge model: %s", judge_model)

    writer = JsonlWriter(results_path)
    counter = {"done": 0, "skipped": 0, "failed": 0, "total": len(conv_files)}

    async def score_one(path: Path) -> None:
        try:
            conv = load_json(path)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to load %s: %s", path, e)
            counter["failed"] += 1
            return
        if (conv.get("persona_id", ""), conv.get("prompt_id", "")) in existing:
            counter["skipped"] += 1
            return
        persona = profiles.get(conv.get("persona_id", ""))
        try:
            result = await score_conversation(conv, persona, llm, skip_judges=args.skip_judges)
        except Exception as e:  # noqa: BLE001
            logger.exception("Score failed for %s: %s", path, e)
            counter["failed"] += 1
            return
        await writer.write(result.to_dict())
        counter["done"] += 1
        if counter["done"] % 50 == 0:
            logger.info("Progress: %d/%d", counter["done"], counter["total"])

    t0 = time.time()
    try:
        await gather_bounded([lambda p=p: score_one(p) for p in conv_files], args.concurrency)
    finally:
        writer.close()
    summary = _summarize_qc(results_path, out_dir / "qc_summary.json")
    logger.info("Done in %.1fs: scored=%d skipped=%d failed=%d | tiers=%s pass-rate=%.3f",
                time.time() - t0, counter["done"], counter["skipped"], counter["failed"],
                summary["tier_counts"], summary["tier_pass_rate"])


def _load_tier_a_keys(qc_jsonl: Path) -> set[tuple[str, str]]:
    keys = {(r.get("persona_id", ""), r.get("scenario_id", ""))
            for r in read_jsonl(qc_jsonl) if r.get("tier") == "A"}
    logger.info("Loaded %d Tier-A keys from %s", len(keys), qc_jsonl)
    return keys


async def cmd_qa_build(args: argparse.Namespace) -> None:
    from user_simulator.data import LLM, load_json
    from user_simulator.qa import (
        QAStyle, generate_for_conv, qa_item_to_sft_line, self_consistency_check_mcq,
    )

    conv_dir = Path(args.conversations_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = personas_by_id(args.profiles_jsonl) if args.profiles_jsonl else {}
    tier_a = _load_tier_a_keys(Path(args.qc_results)) if args.qc_results else None

    conv_files = sorted(conv_dir.rglob("*.json"))
    if args.sample:
        conv_files = conv_files[: args.sample]
    styles = [QAStyle(s) for s in args.styles]
    logger.info("Found %d conversations; generating styles %s", len(conv_files), [s.value for s in styles])

    llm = LLM(model=args.generator_model, max_concurrent=args.concurrency, log_calls=args.log_calls)
    writers = {s: JsonlWriter(out_dir / f"{s.value}.jsonl") for s in styles}
    counter = {"convs": 0, "items": 0, "skipped": 0, "qc_dropped": 0, "failed": 0}

    async def process_one(path: Path) -> None:
        try:
            session = load_json(path)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to load %s: %s", path, e)
            counter["failed"] += 1
            return
        pid, sid = session.get("persona_id", ""), session.get("prompt_id", "")
        if tier_a is not None and (pid, sid) not in tier_a:
            counter["skipped"] += 1
            return
        persona = profiles.get(pid)
        counter["convs"] += 1
        for style in styles:
            try:
                item = await generate_for_conv(persona, session, style, llm)
            except Exception as e:  # noqa: BLE001
                logger.warning("Generation failed %s/%s/%s: %s", pid, sid, style.value, e)
                counter["failed"] += 1
                continue
            if item is None:
                continue
            if args.self_consistency_qc and style is QAStyle.PERSONAMEM_MCQ:
                if not await self_consistency_check_mcq(item, llm):
                    counter["qc_dropped"] += 1
                    continue
            await writers[style].write(qa_item_to_sft_line(item, session))
            counter["items"] += 1
            if counter["items"] % 25 == 0:
                logger.info("Progress: %d items (%d convs)", counter["items"], counter["convs"])

    t0 = time.time()
    try:
        await gather_bounded([lambda p=p: process_one(p) for p in conv_files], args.concurrency)
    finally:
        for w in writers.values():
            w.close()
    logger.info("Done in %.1fs: items=%d convs=%d skipped=%d qc_dropped=%d failed=%d",
                time.time() - t0, counter["items"], counter["convs"],
                counter["skipped"], counter["qc_dropped"], counter["failed"])
    for s, w in writers.items():
        logger.info("  %s → %s (%d lines)", s.value, w.path, w.n)
    logger.info("LLM stats: %s", llm.stats)


_PERSONAMEM_RECALL_SUFFIX = (
    " Please recall my related preferences from our conversation history "
    "to give personalized responses."
)
_PERSONAMEM_OPTIONS_INTRO = "Please choose the best answer from the following options:"
_PERSONAMEM_OPTIONS_OUTRO = (
    "Think step by step about which answer best fits the user's query and "
    "conversation context. Provide your reasoning first, then give your final "
    "answer as 'Final Answer: [Letter]'"
)
_OPTION_RE = re.compile(r"^([A-D])\.\s+(.*)$", re.MULTILINE)
_LAMP_ITEM_RE = re.compile(r"^- input:\s*(.*?)\s*\|\s*output:\s*(.*)$", re.MULTILINE)
_REWRITE_PROMPTS = {
    "personamem_mcq": "qa_rewrite_personamem",
    "prefeval_gen": "qa_rewrite_prefeval",
    "bigtom_tom": "qa_rewrite_bigtom",
    "lamp_cls": "qa_rewrite_lamp",
}


def _personamem_options_block(options: list[str]) -> str:
    parts = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
    return f"{_PERSONAMEM_OPTIONS_INTRO}\n\n" + "\n".join(parts) + f"\n\n{_PERSONAMEM_OPTIONS_OUTRO}"


def _parse_personamem(item: dict, _ctx: dict | None = None) -> dict | None:
    msgs = item["messages"]
    correct_letter = item["metadata"].get("correct_letter")
    if not correct_letter:
        return None
    m = re.search(r"<persona>\s*\n(.*?)\n</persona>", msgs[0]["content"], re.DOTALL)
    profile_block = m.group(1).strip() if m else ""
    history = msgs[1:-2]
    conv_excerpt = "\n".join(
        f"{('User' if x['role'] == 'user' else 'Assistant')}: {x['content']}" for x in history
    )
    last_user = msgs[-2]["content"]
    if _PERSONAMEM_RECALL_SUFFIX in last_user:
        head, _, tail = last_user.partition(_PERSONAMEM_RECALL_SUFFIX)
        user_query, opts_part = head.strip(), tail
    else:
        m = re.search(re.escape(_PERSONAMEM_OPTIONS_INTRO), last_user)
        if not m:
            return None
        user_query, opts_part = last_user[: m.start()].strip(), last_user[m.start():]
    options = ["", "", "", ""]
    for letter, text in _OPTION_RE.findall(opts_part):
        idx = ord(letter) - ord("A")
        if 0 <= idx < 4:
            options[idx] = text.strip().split("\n\n")[0].strip()
    if not all(options):
        return None
    return {
        "profile_block": profile_block, "conversation_excerpt": conv_excerpt,
        "user_query": user_query, "options": options, "correct_letter": correct_letter,
        "correct_text": options[ord(correct_letter) - ord("A")],
    }


def _parse_prefeval(item: dict, ctx: dict | None = None) -> dict | None:
    msgs = item["messages"]
    if len(msgs) < 5:
        return None
    ctx = ctx or {}
    return {
        "profile_block": ctx.get("profile_block", ""),
        "conversation_excerpt": ctx.get("conversation_excerpt", ""),
        "preference": msgs[1]["content"], "question": msgs[3]["content"], "response": msgs[4]["content"],
    }


def _parse_bigtom(item: dict, ctx: dict | None = None) -> dict | None:
    msgs = item["messages"]
    correct_letter = item["metadata"].get("correct_letter", "").lower()
    if correct_letter not in {"a", "b"}:
        return None
    user_msg = msgs[1]["content"]
    ctx = ctx or {}
    m = re.search(r"\n\nChoose one of the following:\n", user_msg)
    if not m:
        return None
    pre, post = user_msg[: m.start()], user_msg[m.end():]
    parts = pre.split("\n\n", 1)
    if len(parts) != 2:
        return None
    narrative, question = parts[0].strip(), parts[1].strip()
    opt_a = opt_b = ""
    for line in post.splitlines():
        line = line.strip()
        if line.lower().startswith("a)"):
            opt_a = line[2:].strip()
        elif line.lower().startswith("b)"):
            opt_b = line[2:].strip()
    if not opt_a or not opt_b:
        return None
    return {
        "profile_block": ctx.get("profile_block", ""), "conversation_excerpt": ctx.get("conversation_excerpt", ""),
        "narrative": narrative, "question": question, "option_a": opt_a, "option_b": opt_b,
        "correct_letter": correct_letter,
    }


def _parse_lamp(item: dict, ctx: dict | None = None) -> dict | None:
    ctx = ctx or {}
    user_msg = item["messages"][1]["content"]
    items = [{"input": i.strip(), "output": o.strip()} for i, o in _LAMP_ITEM_RE.findall(user_msg)]
    if len(items) < 2:
        return None
    m = re.search(r"New input:\s*(.*?)\nProduce the output\.", user_msg, re.DOTALL)
    if not m:
        return None
    return {
        "profile_block": ctx.get("profile_block", ""), "profile_items": items,
        "profile_items_str": "\n".join(f"- input: {it['input']} | output: {it['output']}" for it in items),
        "query": m.group(1).strip(), "gold_target": item["messages"][-1]["content"].strip(),
        "task_family": item["metadata"].get("task_family", "tag_classify"),
    }


def _render_personamem_v2(orig: dict, parsed: dict, rw: dict) -> dict | None:
    options_dict = rw.get("options") or {}
    if not isinstance(options_dict, dict):
        return None
    options = [options_dict.get(L, "").strip() for L in "ABCD"]
    new_query = str(rw.get("user_query", "")).strip()
    answer = str(rw.get("reasoning_trace_and_answer", "")).strip()
    correct_letter = parsed["correct_letter"]
    if not all(options) or not new_query or not answer:
        return None
    if f"Final Answer: {correct_letter}" not in answer:
        answer = f"{answer}\n\nFinal Answer: {correct_letter}"
    msgs = [{"role": "system", "content": orig["messages"][0]["content"]}]
    msgs.extend(orig["messages"][1:-2])
    msgs.append({"role": "user",
                 "content": new_query.rstrip() + _PERSONAMEM_RECALL_SUFFIX + "\n\n" + _personamem_options_block(options)})
    msgs.append({"role": "assistant", "content": answer})
    md = dict(orig["metadata"], v1_user_query=parsed["user_query"], v1_options=parsed["options"], rewrite_model="gpt-4o")
    return {"messages": msgs, "metadata": md}


def _render_prefeval_v2(orig: dict, parsed: dict, rw: dict) -> dict | None:
    new_pref = str(rw.get("preference", "")).strip()
    new_q = str(rw.get("question", "")).strip()
    new_resp = str(rw.get("assistant_response", "")).strip()
    ack_quote = str(rw.get("acknowledge_quote", "")).strip()
    if not new_pref or not new_q or not new_resp:
        return None
    resp_lower = new_resp.lower()
    if not (ack_quote and ack_quote.lower() in resp_lower):
        pref_tokens = {t.lower().strip(".,!?;:\"'") for t in new_pref.split() if len(t) >= 4}
        if sum(1 for t in pref_tokens if t and t in resp_lower) < 2:
            return None
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": new_pref},
        {"role": "assistant", "content": "Got it — I'll keep that in mind for our conversation."},
        {"role": "user", "content": new_q},
        {"role": "assistant", "content": new_resp},
    ]
    md = dict(orig["metadata"], preference=new_pref, v1_preference=parsed["preference"],
              v1_question=parsed["question"], rewrite_model="gpt-4o")
    return {"messages": msgs, "metadata": md}


def _render_bigtom_v2(orig: dict, parsed: dict, rw: dict) -> dict | None:
    narrative = str(rw.get("narrative", "")).strip()
    question = str(rw.get("question", "")).strip()
    opt_a = str(rw.get("option_a", "")).strip()
    opt_b = str(rw.get("option_b", "")).strip()
    answer = str(rw.get("reasoning_trace_and_answer", "")).strip()
    correct_letter = parsed["correct_letter"]
    correct_text = opt_a if correct_letter == "a" else opt_b
    if not all([narrative, question, opt_a, opt_b, answer]):
        return None
    tail = f"Answer:{correct_letter}){correct_text}"
    if tail not in answer:
        answer = f"{answer}\n\n{tail}"
    user_text = f"{narrative}\n\n{question}\n\nChoose one of the following:\na) {opt_a}\nb) {opt_b}"
    msgs = [
        {"role": "system", "content": orig["messages"][0]["content"]},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer},
    ]
    md = dict(orig["metadata"], v1_narrative=parsed["narrative"], v1_question=parsed["question"], rewrite_model="gpt-4o")
    return {"messages": msgs, "metadata": md}


def _render_lamp_v2(orig: dict, parsed: dict, rw: dict) -> dict | None:
    items = rw.get("profile_items") or []
    if not isinstance(items, list) or len(items) < 2:
        return None
    lines = [f"- input: {str(it.get('input','')).strip()} | output: {str(it.get('output','')).strip()}"
             for it in items if isinstance(it, dict) and it.get("input") and it.get("output")]
    query = str(rw.get("query", "")).strip()
    answer = str(rw.get("reasoning_trace_and_answer", "")).strip()
    gold = parsed["gold_target"]
    if not lines or not query or not answer:
        return None
    answer = answer.rstrip()
    if not (answer.splitlines()[-1].strip().lower().endswith(gold.lower()) if answer else False):
        answer = answer + f"\n{gold}"
    user_text = ("Here are some past items belonging to this user:\n" + "\n".join(lines)
                 + f"\n\nNew input: {query}\nProduce the output.")
    msgs = [
        {"role": "system", "content": orig["messages"][0]["content"]},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer},
    ]
    md = dict(orig["metadata"], v1_query=parsed["query"], gold_target=gold, rewrite_model="gpt-4o")
    return {"messages": msgs, "metadata": md}


_REWRITE_PARSERS = {"personamem_mcq": _parse_personamem, "prefeval_gen": _parse_prefeval,
                    "bigtom_tom": _parse_bigtom, "lamp_cls": _parse_lamp}
_REWRITE_RENDERERS = {"personamem_mcq": _render_personamem_v2, "prefeval_gen": _render_prefeval_v2,
                      "bigtom_tom": _render_bigtom_v2, "lamp_cls": _render_lamp_v2}


def _render_rewrite_prompt(style: str, parsed: dict, tmpl: str) -> str:
    from user_simulator.prompts import render

    if style == "personamem_mcq":
        opts = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(parsed["options"]))
        return render(tmpl, profile_block=parsed["profile_block"], conversation_excerpt=parsed["conversation_excerpt"],
                      original_user_query=parsed["user_query"], original_options_block=opts,
                      correct_letter=parsed["correct_letter"], original_correct_text=parsed["correct_text"])
    if style == "prefeval_gen":
        return render(tmpl, profile_block=parsed["profile_block"], conversation_excerpt=parsed["conversation_excerpt"],
                      original_preference=parsed["preference"], original_question=parsed["question"],
                      original_response=parsed["response"])
    if style == "bigtom_tom":
        return render(tmpl, profile_block=parsed["profile_block"], conversation_excerpt=parsed["conversation_excerpt"],
                      original_narrative=parsed["narrative"], original_question=parsed["question"],
                      original_option_a=parsed["option_a"], original_option_b=parsed["option_b"],
                      correct_letter=parsed["correct_letter"])
    if style == "lamp_cls":
        return render(tmpl, profile_block=parsed["profile_block"], original_profile_items=parsed["profile_items_str"],
                      original_query=parsed["query"], gold_target=parsed["gold_target"], task_family=parsed["task_family"])
    raise ValueError(style)


def _build_ctx_lookup(conv_root: Path) -> dict:
    lookup: dict = {}
    for p in conv_root.rglob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(d, dict) or not d.get("persona_id") or not d.get("prompt_id"):
            continue
        conv = d.get("conversation", []) or []
        excerpt = "\n".join(
            f"{('User' if m['role'] == 'user' else 'Assistant')}: {m['content']}" for m in conv[:6]
        )
        lookup[(d["persona_id"], d["prompt_id"])] = {
            "profile_block": d.get("profile_summary", "") or "", "conversation_excerpt": excerpt}
    logger.info("Built context lookup for %d (persona, scenario) pairs", len(lookup))
    return lookup


async def _rewrite_one(style: str, orig: dict, tmpl: str, llm, ctx_lookup: dict) -> dict | None:
    md = orig.get("metadata", {})
    parsed = _REWRITE_PARSERS[style](orig, ctx_lookup.get((md.get("persona_id", ""), md.get("scenario_id", ""))))
    if parsed is None:
        return None
    try:
        rewritten = await llm.chat_json(
            [{"role": "system", "content": _render_rewrite_prompt(style, parsed, tmpl)},
             {"role": "user", "content": "Generate the rewritten JSON now."}],
            temperature=0.7, max_tokens=2200, call_type=f"rewrite_{style}",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("rewrite failed %s/%s: %s", md.get("persona_id"), style, e)
        return None
    return _REWRITE_RENDERERS[style](orig, parsed, rewritten)


async def cmd_qa_rewrite(args: argparse.Namespace) -> None:
    from user_simulator.data import LLM
    from user_simulator.prompts import load_prompt

    qa_dir = Path(args.qa_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx_lookup = _build_ctx_lookup(Path(args.conversations_dir).resolve()) if args.conversations_dir else {}

    llm = LLM(model=args.rewriter_model, max_concurrent=args.concurrency, retries=2)
    logger.info("Rewriter model: %s", args.rewriter_model)
    counter = {"in": 0, "out": 0}

    for style in args.styles:
        in_path, out_path = qa_dir / f"{style}.jsonl", out_dir / f"{style}.jsonl"
        if not in_path.exists():
            logger.warning("Skipping %s: no input at %s", style, in_path)
            continue
        items = read_jsonl(in_path, limit=args.sample)
        counter["in"] += len(items)
        tmpl = load_prompt(_REWRITE_PROMPTS[style])
        logger.info("Style %s: %d items", style, len(items))

        t0 = time.time()
        results = await gather_bounded(
            [lambda it=it: _rewrite_one(style, it, tmpl, llm, ctx_lookup) for it in items], args.concurrency)
        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                if r is not None:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
        counter["out"] += n
        logger.info("  %s: %d/%d written → %s (%.1fs)", style, n, len(items), out_path, time.time() - t0)

    logger.info("Done: %d in, %d out (%.1f%% kept) | %s",
                counter["in"], counter["out"], 100 * counter["out"] / max(counter["in"], 1), llm.stats)


_MCQ_LETTER_PATTERNS = [
    r"\$\$\s*\\boxed\{([A-Z])\}\s*\$\$", r"\$\\boxed\{([A-Z])\}\$", r"\\boxed\{([A-Z])\}",
    r"\*{0,2}\s*Final Answer\s*:?\s*\*{0,2}[\s*]*\[?([A-Z])\]?\*{0,2}",
    r"\*{0,2}\s*final answer\s*:?\s*\*{0,2}[\s*]*\[?([A-Z])\]?\*{0,2}",
    r"Answer:\s*\*{0,2}\[?([A-Z])\]?\*{0,2}", r"answer:\s*\*{0,2}\[?([A-Z])\]?\*{0,2}",
    r"correct answer is\s*\*{0,2}\[?([A-Z])\]?\*{0,2}",
    r"the answer is\s*\$?\\boxed\{([A-Z])\}\$?", r"the answer is\s*\*{0,2}\(?\[?([A-Z])\]?\)?\*{0,2}",
    r"final answer is\s*\$?\\boxed\{([A-Z])\}\$?", r"final answer is\s*\*{0,2}\(?\[?([A-Z])\]?\)?\*{0,2}",
    r"option\s+\*{0,2}\(?([A-Z])\)?\*{0,2}", r"\b([A-Z])\.\s*$",
]
_PREFEVAL_JUDGE_PROMPT = """You are evaluating whether an AI assistant response respects a stated user preference.

<user_preference>
{preference}
</user_preference>

<user_question>
{question}
</user_question>

<assistant_response>
{response}
</assistant_response>

Score the response on three binary dimensions:
1. acknowledges: does the response explicitly reference the preference (e.g., "since you prefer..." or quoting it)? Implicit alignment without a verbal nod counts as FALSE here.
2. violates_preference: does the response make a recommendation that actively contradicts the preference? (Adding helpful tangential information that the user did not ask for is NOT a violation by itself.)
3. helpful: does the response answer the question with useful, concrete content (not platitudes, not refusal)?

Return JSON only:
{{"acknowledges": true|false, "violates_preference": true|false, "helpful": true|false, "reason": "<one sentence>"}}
"""


def extract_mcq_letter(response: str) -> str | None:
    if not response:
        return None
    text = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)
    text = re.sub(r"</think>\s*", "", text)
    for pat in _MCQ_LETTER_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


def score_bigtom(response: str, correct_letter: str, correct_text: str) -> bool:
    if not response or not correct_letter:
        return False
    p = response.lower()
    target = f"{correct_letter.lower()})"
    other = "b)" if correct_letter.lower() == "a" else "a)"
    if target in p:
        return True
    if other in p:
        return False
    return correct_text.lower()[:20] in p


def score_lamp(response: str, target: str) -> bool:
    if not response or not target:
        return False
    r, t = response.strip().lower(), target.strip().lower()
    if t == r or t in r:
        return True
    t_tokens = [w.strip(".,!?;:\"'") for w in re.split(r"[\s_]+", t) if len(w) >= 3]
    r_tokens = set(re.split(r"[\s_]+", r))
    if not t_tokens:
        return False
    return sum(1 for w in t_tokens if w in r_tokens or w in r) / len(t_tokens) >= 0.5


async def judge_prefeval(response: str, preference: str, question: str, llm) -> tuple[bool, dict]:
    prompt = _PREFEVAL_JUDGE_PROMPT.format(preference=preference, question=question, response=response)
    try:
        data = await llm.chat_json(
            [{"role": "system", "content": prompt}, {"role": "user", "content": "Score now."}],
            temperature=0.0, max_tokens=300, call_type="eval_prefeval_judge")
    except Exception as e:  # noqa: BLE001
        return False, {"error": str(e)}
    if not isinstance(data, dict):
        return False, {"raw": str(data)[:200]}
    return (bool(data.get("helpful")) and not bool(data.get("violates_preference"))), data


async def _evaluate_one(item: dict, model_llm, judge_llm, max_tokens: int) -> dict:
    msgs, meta = item["messages"], item["metadata"]
    style = meta["qa_style"]
    try:
        response = await model_llm.chat(msgs[:-1], temperature=0.0, max_tokens=max_tokens, call_type=f"eval_{style}")
    except Exception as e:  # noqa: BLE001
        return {"persona_id": meta.get("persona_id"), "scenario_id": meta.get("scenario_id"),
                "qa_style": style, "error": str(e), "correct": False}
    out = {"persona_id": meta.get("persona_id"), "scenario_id": meta.get("scenario_id"),
           "qa_style": style, "response": response}
    if style == "personamem_mcq":
        gold, pred = meta.get("correct_letter"), extract_mcq_letter(response)
        out.update(predicted_letter=pred, gold_letter=gold,
                   correct=pred is not None and pred.upper() == (gold or "").upper())
    elif style == "bigtom_tom":
        gold_letter = meta.get("correct_letter", "").lower()
        m = re.search(r"Answer:[ab]\)(.*)$", msgs[-1]["content"], re.IGNORECASE | re.DOTALL)
        out.update(gold_letter=gold_letter,
                   correct=score_bigtom(response, gold_letter, m.group(1).strip() if m else ""))
    elif style == "prefeval_gen":
        question = msgs[-2]["content"] if msgs[-2]["role"] == "user" else ""
        passed, judge = await judge_prefeval(response, meta.get("preference", ""), question, judge_llm)
        out.update(judge=judge, correct=passed)
    elif style == "lamp_cls":
        gold = meta.get("gold_target") or msgs[-1]["content"]
        out.update(gold=gold, correct=score_lamp(response, gold))
    else:
        out.update(correct=False, error=f"unknown qa_style: {style}")
    return out


def _resolve_max_tokens(model_name: str, default: int, overrides: dict[str, int]) -> int:
    if model_name in overrides:
        return overrides[model_name]
    if "gpt-5" in model_name.lower():
        return max(default, 16384)
    return default


async def _eval_style_model(model_name, items, output_path: Path, judge_llm, concurrency, max_tokens) -> dict:
    from user_simulator.data import LLM

    model_llm = LLM(model=model_name, max_concurrent=concurrency, retries=2)
    t0 = time.time()
    results = await gather_bounded(
        [lambda i=i: _evaluate_one(items[i], model_llm, judge_llm, max_tokens) for i in range(len(items))],
        concurrency)
    elapsed = time.time() - t0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(results)
    return {
        "model": model_name, "qa_style": items[0]["metadata"]["qa_style"] if items else "?",
        "n": n, "n_correct": sum(1 for r in results if r and r.get("correct")),
        "accuracy": (sum(1 for r in results if r and r.get("correct")) / n) if n else 0.0,
        "n_errored": sum(1 for r in results if r and "error" in r),
        "n_unparseable_mcq": sum(1 for r in results if r and r.get("qa_style") == "personamem_mcq"
                                 and r.get("predicted_letter") is None),
        "elapsed_s": round(elapsed, 1),
        "max_tokens": max_tokens, "llm_calls": model_llm.calls, "tokens": model_llm.tokens,
    }


async def cmd_qa_eval(args: argparse.Namespace) -> None:
    from user_simulator.data import LLM

    qa_dir = Path(args.qa_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    style_files = {p.stem: p for p in sorted(qa_dir.glob("*.jsonl")) if p.stem in args.styles}
    for s in (s for s in args.styles if s not in style_files):
        logger.warning("No JSONL found for style: %s", s)

    judge_llm = LLM(model=args.judge_model, max_concurrent=args.concurrency, retries=2)
    overrides: dict[str, int] = {}
    for spec in args.model_max_tokens or []:
        if "=" not in spec:
            raise SystemExit(f"--model-max-tokens entry {spec!r} must be model=N")
        m, v = spec.split("=", 1)
        overrides[m.strip()] = int(v.strip())

    sample = None if (args.sample is not None and args.sample <= 0) else args.sample
    summaries: list[dict] = []
    for style in args.styles:
        if style not in style_files:
            continue
        items = read_jsonl(style_files[style], limit=sample)
        logger.info("Style %s: %d items", style, len(items))
        for model_name in args.models:
            mt = _resolve_max_tokens(model_name, args.max_tokens, overrides)
            preds_path = out_dir / f"{style}__{model_name.replace('/', '_')}.predictions.jsonl"
            logger.info("→ %s on %s (%d items, max_tokens=%d)", model_name, style, len(items), mt)
            s = await _eval_style_model(model_name, items, preds_path, judge_llm, args.concurrency, mt)
            logger.info("   %s/%s: acc=%.3f (%d/%d) errs=%d unparseable_mcq=%d in %.1fs", model_name, style,
                        s["accuracy"], s["n_correct"], s["n"], s["n_errored"],
                        s["n_unparseable_mcq"], s["elapsed_s"])
            summaries.append(s)

    (out_dir / "eval_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    md = ["# QA-eval summary", "",
          "| model | style | n | accuracy | errs | unparseable_mcq | max_tokens | wall |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    md += [f"| {s['model']} | {s['qa_style']} | {s['n']} | {s['accuracy']:.3f} | "
           f"{s['n_errored']} | {s['n_unparseable_mcq']} | {s.get('max_tokens', '?')} | {s['elapsed_s']}s |"
           for s in summaries]
    (out_dir / "eval_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + "\n".join(md))
    logger.info("Summary → %s", out_dir / "eval_summary.json")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mind2dialogue",
        description="Unified CLI for the Mind2Dialogue state-aware simulation data pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--concurrency", type=int, default=40, help="Max concurrent LLM calls")

    sp = sub.add_parser("rollout", help="Roll out conversations from rewritten persona prompts")
    sp.add_argument("--concurrency", type=int, default=80, help="Max concurrent LLM calls")
    sp.add_argument("--ablation", default="full",
                    choices=["full", "no_privilege", "no_behavior", "no_state", "oracle_profile_only"])
    sp.add_argument("--max-turns", type=int, default=12)
    sp.add_argument("--min-turns", type=int, default=3)
    sp.add_argument("--persona-ids", nargs="*", help="Filter to specific persona IDs")
    sp.add_argument("--max-prompts", type=int, default=None, help="Max prompts per persona")
    sp.add_argument("--output-dir", default=None, help="Custom output directory (default: output/)")
    sp.add_argument("--prompts-jsonl", default=None, help="Prompt JSONL (persona_id + rewritten/prompt_text)")
    sp.add_argument("--profiles", default=None, help="Persona JSONL file or YAML directory")
    sp.set_defaults(func=cmd_rollout)

    sp = sub.add_parser("scenario", help="Construct per-persona scenarios, then roll out deep dialogues")
    add_common(sp)
    sp.add_argument("--ablation", default="full",
                    choices=["full", "no_privilege", "no_behavior", "no_state", "oracle_profile_only"])
    sp.add_argument("--constructor", default="lifelong",
                    help="Scenario family (lifelong|highfreq|affective|concerning) or a full prompt name")
    sp.add_argument("--max-turns", type=int, default=12)
    sp.add_argument("--min-turns", type=int, default=3)
    sp.add_argument("--persona-ids", nargs="*", help="Filter to specific persona IDs")
    sp.add_argument("--max-scenarios", type=int, default=None, help="Cap scenarios per persona")
    sp.add_argument("--output-dir", default=None, help="Custom output directory (default: output/)")
    sp.add_argument("--force-reconstruct", action="store_true", help="Regenerate scenarios even if cached")
    sp.add_argument("--profiles", default=None, help="Persona JSONL file or YAML directory")
    sp.set_defaults(func=cmd_scenario)

    sp = sub.add_parser("qc", help="Quality-control scoring + A/B/C tiering of conversations")
    add_common(sp)
    sp.add_argument("--conversations-dir", required=True, help="Directory of conversation JSONs (recursive)")
    sp.add_argument("--output-dir", default="output/qc/v1_demo", help="Where qc_results.jsonl + qc_summary.json go")
    sp.add_argument("--profiles-jsonl", default=str(DEFAULT_PROFILES), help="JSONL of personas")
    sp.add_argument("--skip-judges", action="store_true", help="Skip D5/D6 LLM judges (programmatic only)")
    sp.add_argument("--judge-model", default="gpt-4.1-mini", help="Override JUDGE_MODEL env var")
    sp.add_argument("--sample", type=int, default=None, help="Score only the first N convs (smoke)")
    sp.add_argument("--log-calls", action="store_true", help="JSONL-log every judge call")
    sp.set_defaults(func=cmd_qc)

    sp = sub.add_parser("qa-build", help="Build QA-format SFT data from (Tier-A) conversations")
    add_common(sp)
    sp.add_argument("--conversations-dir", required=True, help="Directory of conversation JSONs (recursive)")
    sp.add_argument("--output-dir", default="output/qa/v1_demo", help="Where per-style JSONL files go")
    sp.add_argument("--qc-results", default=None, help="qc_results.jsonl; if given, only Tier-A convs are used")
    sp.add_argument("--profiles-jsonl", default=str(DEFAULT_PROFILES), help="JSONL of personas")
    sp.add_argument("--styles", nargs="+", default=QA_STYLES, choices=QA_STYLES)
    sp.add_argument("--generator-model", default="gpt-4.1-mini", help="QA generator model")
    sp.add_argument("--sample", type=int, default=None, help="Process only the first N convs (smoke)")
    sp.add_argument("--self-consistency-qc", action="store_true",
                    help="Run profile-stripped self-consistency check on MCQ items")
    sp.add_argument("--log-calls", action="store_true")
    sp.set_defaults(func=cmd_qa_build)

    sp = sub.add_parser("qa-rewrite", help="Rewrite v1 QA items into harder, persona-grounded v2 items")
    sp.add_argument("--concurrency", type=int, default=30, help="Max concurrent LLM calls")
    sp.add_argument("--qa-dir", required=True, help="Directory with v1 {style}.jsonl files")
    sp.add_argument("--output-dir", required=True, help="Where v2 {style}.jsonl files go")
    sp.add_argument("--conversations-dir", default=None,
                    help="Source conversation JSONs (injects profile + excerpt for prefeval/bigtom/lamp)")
    sp.add_argument("--rewriter-model", default="gpt-4o")
    sp.add_argument("--styles", nargs="+", default=QA_STYLES, choices=QA_STYLES)
    sp.add_argument("--sample", type=int, default=None, help="Process only the first N items per style")
    sp.set_defaults(func=cmd_qa_rewrite)

    sp = sub.add_parser("qa-eval", help="Benchmark models on the QA-format JSONL data")
    sp.add_argument("--concurrency", type=int, default=10, help="Max concurrent LLM calls")
    sp.add_argument("--qa-dir", default="output/qa/v1_demo_full", help="Directory with per-style JSONL files")
    sp.add_argument("--output-dir", default="output/eval/v1_demo_subset", help="Where predictions + summary go")
    sp.add_argument("--styles", nargs="+", default=QA_STYLES, choices=QA_STYLES)
    sp.add_argument("--models", nargs="+", default=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-5-mini"],
                    help="Models to evaluate (reachable via OPENAI_BASE_URL)")
    sp.add_argument("--judge-model", default="gpt-4.1-mini", help="Judge model for PrefEval scoring")
    sp.add_argument("--sample", type=int, default=10, help="Items per style; 0 (or less) = full set")
    sp.add_argument("--max-tokens", type=int, default=512, help="Default max_tokens (GPT-5 auto-bumped to 16384)")
    sp.add_argument("--model-max-tokens", nargs="+", default=None, metavar="MODEL=N",
                    help="Per-model max_tokens overrides, e.g. 'gpt-5-mini=16384'")
    sp.set_defaults(func=cmd_qa_eval)

    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
