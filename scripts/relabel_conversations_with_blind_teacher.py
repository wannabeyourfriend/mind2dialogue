"""Blind-teacher relabel of the released M2D conversations (dialogue view).

For the privileged-supervision ablation: keep every stored conversation
exactly as released (system prompt, user turns, history) and regenerate each
assistant turn with a teacher that sees profile + behavior_metadata +
history but NOT the latent user state (template assistant_vanilla_with_profile,
the repo's oracle_profile_only arm), mirroring the rollout call path
(temperature=0.7, max_tokens=1024, system + "Generate your response.").

The prefix fed to the teacher for turn k is the ORIGINAL stored prefix
(privileged assistant turns 1..k-1 included) — history is held fixed by design.

Output: one JSONL line per conversation with the blind replacement turns.
Idempotent: resumes by content hash.

Usage:
  python scripts/relabel_conversations_with_blind_teacher.py [--limit 20] [--concurrency 60] [--model gpt-4o-mini]
"""

import argparse
import asyncio
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from user_simulator.data import LLM, format_conversation  # noqa: E402
from user_simulator.prompts import load_prompt, render  # noqa: E402

TMPL = load_prompt("assistant_vanilla_with_profile")

PROFILE_RE = re.compile(r"<user_profile>\n(.*?)\n</user_profile>", re.DOTALL)
BM_RE = re.compile(r"<behavior_metadata>\n(.*?)\n</behavior_metadata>", re.DOTALL)

CONV_GLOB = str(ROOT / "output/hf_corpus/data/conversations/*/*.parquet")
# "highfreq" is the spelling in the published parquet files; "high_frequency" is
# what a fresh generate-scenarios run writes. Accept both.
FAMILIES = {"affective", "highfreq", "high_frequency", "lifelong", "random_multisource"}  # incl. conv-v1realquery (mix_v6)
OUT_PATH = ROOT / "output/relabel_blind/dialogue_blind.jsonl"


def content_hash(msgs) -> str:
    return hashlib.sha256(
        json.dumps([[m["role"], m["content"]] for m in msgs]).encode()
    ).hexdigest()


def load_conversations():
    rows = []
    seen = set()
    for path in sorted(glob.glob(CONV_GLOB)):
        family = Path(path).parent.name
        if family not in FAMILIES:
            continue
        df = pd.read_parquet(path)
        for _, r in df.iterrows():
            msgs = [dict(m) for m in r["messages"]]
            h = content_hash(msgs)
            if h in seen:
                continue
            seen.add(h)
            md = dict(r["metadata"])
            rows.append(
                {
                    "hash": h,
                    "family": family,
                    "persona_id": md.get("persona_id"),
                    "scenario_id": md.get("scenario_id"),
                    "messages": msgs,
                }
            )
    rows.sort(key=lambda r: (r["family"], str(r["persona_id"]), str(r["scenario_id"]), r["hash"]))
    return rows


async def relabel_conversation(row, llm, model):
    sysmsg = row["messages"][0]
    assert sysmsg["role"] == "system"
    m_p = PROFILE_RE.search(sysmsg["content"])
    m_b = BM_RE.search(sysmsg["content"])
    profile_summary = m_p.group(1) if m_p else "N/A"
    behavior_metadata = m_b.group(1) if m_b else "N/A"
    conv = row["messages"][1:]

    async def one_turn(i):
        prefix = conv[:i]
        if not prefix or prefix[-1]["role"] != "user":
            return i, None  # malformed position; skip
        prompt = render(
            TMPL,
            profile_summary=profile_summary,
            behavior_metadata=behavior_metadata,
            conversation_prefix=format_conversation(prefix),
        )
        out = await llm.chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Generate your response."},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return i, out or None

    tasks = [one_turn(i) for i, m in enumerate(conv) if m["role"] == "assistant"]
    results = await asyncio.gather(*tasks)
    blind_turns = {str(i): text for i, text in results if text}
    empty = [i for i, text in results if not text]
    return {
        "content_hash": row["hash"],
        "family": row["family"],
        "persona_id": row["persona_id"],
        "scenario_id": row["scenario_id"],
        "teacher_model": model,
        "teacher_template": "assistant_vanilla_with_profile",
        "blind_turns": blind_turns,
        "empty_turns": empty,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=60)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--only-hashes", help="JSON list of content hashes to relabel")
    args = ap.parse_args()

    rows = load_conversations()
    if args.only_hashes:
        want = set(json.loads(Path(args.only_hashes).read_text()))
        rows = [r for r in rows if r["hash"] in want]
        print(f"restricted to {len(rows)} conversations from {args.only_hashes}", flush=True)
    done = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text().splitlines():
            try:
                done.add(json.loads(line)["content_hash"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [r for r in rows if r["hash"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"conversations: {len(rows)} total, {len(done)} done, {len(todo)} to relabel", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=args.model, max_concurrent=args.concurrency)
    lock = asyncio.Lock()
    n_done = 0

    async def run_one(row):
        nonlocal n_done
        rec = await relabel_conversation(row, llm, args.model)
        async with lock:
            with open(OUT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 50 == 0:
                print(f"[{n_done}/{len(todo)}] llm_calls={llm.calls} tokens={llm.tokens}", flush=True)

    await asyncio.gather(*(run_one(r) for r in todo))
    print(f"DONE: {n_done} conversations, {llm.calls} calls, {llm.tokens} total tokens", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
