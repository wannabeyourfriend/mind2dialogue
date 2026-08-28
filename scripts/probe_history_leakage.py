"""Probe: is the MCQ answer recoverable without the privileged assistant turns?

The personamem_mcq student prompt replays the last 8 conversation messages,
which include ORIGINAL (state-privileged) assistant turns. A blind teacher
answering from that prompt may be reading state that leaked through those
turns rather than inferring it from the user's own words.

This probe re-asks a sample of items with all assistant turns removed from
the replayed history (user turns + question + options only) and reports
agreement with the stored state-derived label.

Usage: python scripts/probe_history_leakage.py [--sample 400] [--concurrency 15]
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from relabel_questions_with_blind_teacher import FINAL_RE, MCQ_SUFFIX, load_items  # noqa: E402
from user_simulator.data import LLM  # noqa: E402

OUT_PATH = ROOT / "output/relabel_blind/probe_history_leak.jsonl"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=15)
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    items = [
        i
        for i in load_items()
        if i["qa_style"] == "personamem_mcq" and i["rewrite_status"] == "v2_rewritten"
    ]
    random.Random(42).shuffle(items)
    items = items[: args.sample]
    print(f"probing {len(items)} personamem_mcq items with assistant turns stripped", flush=True)

    llm = LLM(model=args.model, max_concurrent=args.concurrency)
    lock = asyncio.Lock()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    open(OUT_PATH, "w").close()

    async def run_one(item):
        msgs = item["messages"][:-1]
        stripped = [m for m in msgs if m["role"] != "assistant"]
        stripped[-1] = dict(stripped[-1])
        stripped[-1]["content"] = stripped[-1]["content"] + MCQ_SUFFIX
        out = await llm.chat(stripped, temperature=0.7, max_tokens=1024)
        m = FINAL_RE.search(out or "")
        letter = m.group(1).upper() if m else None
        stored = (item["stored_letter"] or "").upper()
        rec = {
            "sample_id": item["sample_id"],
            "letter_no_assistant": letter,
            "stored_letter": stored or None,
            "agree": (letter == stored) if (letter and stored) else None,
            "n_msgs_before": len(msgs),
            "n_msgs_after": len(stripped),
        }
        async with lock:
            with open(OUT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    await asyncio.gather(*(run_one(i) for i in items))

    recs = [json.loads(line) for line in OUT_PATH.read_text().splitlines()]
    parsed = [r for r in recs if r["agree"] is not None]
    agree = sum(1 for r in parsed if r["agree"])
    print(f"parsed {len(parsed)}/{len(recs)} | agreement without assistant turns: {agree}/{len(parsed)} = {agree/max(len(parsed),1):.1%}")


if __name__ == "__main__":
    asyncio.run(main())
