from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mind2dialogue")

ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILES = ROOT / "data" / "refined_persona_profiles" / "summary_refined_profiles_us.jsonl"
DEFAULT_PROMPTS = ROOT / "data" / "rewritten_prompts" / "original_rewritten_selected_prompts_us.jsonl"
SCENARIOS_DIR = ROOT / "data" / "deep_scenarios"

QUESTION_ANSWERING_STYLES = ["personamem_mcq", "prefeval_gen", "lamp_cls"]

_SCENARIO_CONSTRUCTORS = {
    "lifelong": "simulator_lifelong_scenario_constructor",
    "high_frequency": "simulator_high_frequency_scenario_constructor",
    "affective": "simulator_affective_scenario_constructor",
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
    from user_simulator.data import CONV_DIR, TRAINING_DATA_DIR

    if output_dir:
        base = Path(output_dir)
        return base / "conversations", base / "training_data"
    return CONV_DIR, TRAINING_DATA_DIR


async def command_generate_conversations(args: argparse.Namespace) -> None:
    from user_simulator.ablation import AblationConfig
    from user_simulator.data import LLM, SIM_MODEL, save_json
    from user_simulator.supervised_finetuning import build_training_instance
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
            instance = build_training_instance(session, config)
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


async def command_generate_scenarios(args: argparse.Namespace) -> None:
    from user_simulator.ablation import AblationConfig
    from user_simulator.data import LLM, SIM_MODEL, save_json
    from user_simulator.prompts import load_prompt
    from user_simulator.supervised_finetuning import build_training_instance
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
            instance = build_training_instance(session, config)
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


async def command_check_quality(args: argparse.Namespace) -> None:
    import os

    from user_simulator.data import LLM, load_json
    from user_simulator.quality_control import score_conversation

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


async def command_build_question_answering_data(args: argparse.Namespace) -> None:
    from user_simulator.data import LLM, load_json
    from user_simulator.question_answering import (
        QuestionAnsweringStyle, generate_for_conversation, question_answering_item_to_training_line,
    )

    conv_dir = Path(args.conversations_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = personas_by_id(args.profiles_jsonl) if args.profiles_jsonl else {}
    tier_a = _load_tier_a_keys(Path(args.quality_control_results)) if args.quality_control_results else None

    conv_files = sorted(conv_dir.rglob("*.json"))
    if args.sample:
        conv_files = conv_files[: args.sample]
    styles = [QuestionAnsweringStyle(s) for s in args.styles]
    logger.info("Found %d conversations; generating styles %s", len(conv_files), [s.value for s in styles])

    llm = LLM(model=args.generator_model, max_concurrent=args.concurrency, log_calls=args.log_calls)
    writers = {s: JsonlWriter(out_dir / f"{s.value}.jsonl") for s in styles}
    counter = {"convs": 0, "items": 0, "skipped": 0, "failed": 0}

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
                item = await generate_for_conversation(persona, session, style, llm)
            except Exception as e:  # noqa: BLE001
                logger.warning("Generation failed %s/%s/%s: %s", pid, sid, style.value, e)
                counter["failed"] += 1
                continue
            if item is None:
                continue
            await writers[style].write(question_answering_item_to_training_line(item, session))
            counter["items"] += 1
            if counter["items"] % 25 == 0:
                logger.info("Progress: %d items (%d convs)", counter["items"], counter["convs"])

    t0 = time.time()
    try:
        await gather_bounded([lambda p=p: process_one(p) for p in conv_files], args.concurrency)
    finally:
        for w in writers.values():
            w.close()
    logger.info("Done in %.1fs: items=%d convs=%d skipped=%d failed=%d",
                time.time() - t0, counter["items"], counter["convs"],
                counter["skipped"], counter["failed"])
    for s, w in writers.items():
        logger.info("  %s → %s (%d lines)", s.value, w.path, w.n)
    logger.info("LLM stats: %s", llm.stats)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mind2dialogue",
        description="Unified CLI for the Mind2Dialogue state-aware simulation data pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--concurrency", type=int, default=40, help="Max concurrent LLM calls")

    sp = sub.add_parser("generate-conversations", help="Generate conversations from the rewritten persona prompts")
    sp.add_argument("--concurrency", type=int, default=80, help="Max concurrent LLM calls")
    sp.add_argument("--ablation", default="full",
                    choices=["full", "no_behavior", "no_state", "oracle_profile_only"])
    sp.add_argument("--max-turns", type=int, default=12)
    sp.add_argument("--min-turns", type=int, default=3)
    sp.add_argument("--persona-ids", nargs="*", help="Filter to specific persona IDs")
    sp.add_argument("--max-prompts", type=int, default=None, help="Max prompts per persona")
    sp.add_argument("--output-dir", default=None, help="Custom output directory (default: output/)")
    sp.add_argument("--prompts-jsonl", default=None, help="Prompt JSONL (persona_id + rewritten/prompt_text)")
    sp.add_argument("--profiles", default=None, help="Persona JSONL file or YAML directory")
    sp.set_defaults(func=command_generate_conversations)

    sp = sub.add_parser("generate-scenarios", help="Construct one scenario set per persona, then generate deep dialogues")
    add_common(sp)
    sp.add_argument("--ablation", default="full",
                    choices=["full", "no_behavior", "no_state", "oracle_profile_only"])
    sp.add_argument("--constructor", default="lifelong",
                    choices=["lifelong", "high_frequency", "affective"],
                    help="Paper scenario family")
    sp.add_argument("--max-turns", type=int, default=12)
    sp.add_argument("--min-turns", type=int, default=3)
    sp.add_argument("--persona-ids", nargs="*", help="Filter to specific persona IDs")
    sp.add_argument("--max-scenarios", type=int, default=None, help="Cap scenarios per persona")
    sp.add_argument("--output-dir", default=None, help="Custom output directory (default: output/)")
    sp.add_argument("--force-reconstruct", action="store_true", help="Regenerate scenarios even if cached")
    sp.add_argument("--profiles", default=None, help="Persona JSONL file or YAML directory")
    sp.set_defaults(func=command_generate_scenarios)

    sp = sub.add_parser("check-quality", help="Score conversations and sort them into quality tiers A, B and C")
    add_common(sp)
    sp.add_argument("--conversations-dir", required=True, help="Directory of conversation JSONs (recursive)")
    sp.add_argument("--output-dir", default="output/quality_control/v1_demo", help="Where qc_results.jsonl + qc_summary.json go")
    sp.add_argument("--profiles-jsonl", default=str(DEFAULT_PROFILES), help="JSONL of personas")
    sp.add_argument("--skip-judges", action="store_true", help="Skip D5/D6 LLM judges (programmatic only)")
    sp.add_argument("--judge-model", default="gpt-4o-mini", help="Override JUDGE_MODEL env var")
    sp.add_argument("--sample", type=int, default=None, help="Score only the first N convs (smoke)")
    sp.add_argument("--log-calls", action="store_true", help="JSONL-log every judge call")
    sp.set_defaults(func=command_check_quality)

    sp = sub.add_parser("build-question-answering-data", help="Build question answering training data from tier A conversations")
    add_common(sp)
    sp.add_argument("--conversations-dir", required=True, help="Directory of conversation JSONs (recursive)")
    sp.add_argument("--output-dir", default="output/question_answering/v1_demo", help="Where per-style JSONL files go")
    sp.add_argument("--quality-control-results", default=None, help="qc_results.jsonl; if given, only Tier-A convs are used")
    sp.add_argument("--profiles-jsonl", default=str(DEFAULT_PROFILES), help="JSONL of personas")
    sp.add_argument("--styles", nargs="+", default=QUESTION_ANSWERING_STYLES, choices=QUESTION_ANSWERING_STYLES)
    sp.add_argument("--generator-model", default="gpt-4o-mini", help="QA generator model")
    sp.add_argument("--sample", type=int, default=None, help="Process only the first N convs (smoke)")
    sp.add_argument("--log-calls", action="store_true")
    sp.set_defaults(func=command_build_question_answering_data)

    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
