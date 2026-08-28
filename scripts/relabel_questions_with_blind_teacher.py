"""Blind-teacher relabel of the released M2D synthetic QA (questions held fixed).

For each released QA line, keep the student-view input (messages[:-1] —
persona block, replayed history, question, options) byte-identical and
regenerate only the final assistant target with a state-blind teacher:

  personamem_mcq  — teacher answers the MCQ from the visible context
                    (blind letter + CoT; stored correct_letter came from the
                    state-conditioned generator). Letter agreement is the
                    label-agreement metric.
  prefeval_gen    — teacher writes the response from the visible
                    preference + question.
  bigtom_tom      — analysis only (distractor is by construction the
                    surface-only reading); never used for training.

Output: one JSONL line per item. Idempotent: resumes by sample_id.

Usage:
  python scripts/relabel_questions_with_blind_teacher.py [--limit 30] [--concurrency 60] [--model gpt-4o-mini]
"""

import argparse
import asyncio
import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from user_simulator.data import LLM  # noqa: E402

QA_GLOB = str(ROOT / "output/hf_corpus/data/synthetic_qa/*/*.parquet")
OUT_PATH = ROOT / "output/relabel_blind/qa_blind.jsonl"

MCQ_SUFFIX = (
    "\n\nThink step by step about which option best matches this user, "
    "then end your reply with 'Final Answer: X' where X is the letter of your chosen option."
)
FINAL_RE = re.compile(r"Final Answer:\s*\(?([A-Da-d])\)?")
BIGTOM_RE = re.compile(r"<answer>\s*\(?([ab])\b|\b([ab])\)", re.IGNORECASE)


def load_items():
    # sample_ids collide between v1 and v2_rewritten rows; prefer v2 (the trained set)
    by_id = {}
    for path in sorted(glob.glob(QA_GLOB)):
        df = pd.read_parquet(path)
        for _, r in df.iterrows():
            md = dict(r["metadata"])
            sid = md.get("sample_id")
            if not sid:
                continue
            is_v2 = md.get("rewrite_status") == "v2_rewritten"
            if sid in by_id and not is_v2:
                continue
            by_id[sid] = {
                "sample_id": sid,
                "qa_style": md.get("qa_style"),
                "stored_letter": (md.get("correct_letter") or "").strip(),
                "rewrite_status": md.get("rewrite_status"),
                "messages": [dict(m) for m in r["messages"]],
            }
    return sorted(by_id.values(), key=lambda x: x["sample_id"])


async def relabel_item(item, llm, model):
    msgs = item["messages"]
    if len(msgs) < 2 or msgs[-1]["role"] != "assistant" or msgs[-2]["role"] != "user":
        return None
    teacher_msgs = [dict(m) for m in msgs[:-1]]
    style = item["qa_style"]
    if style == "personamem_mcq":
        teacher_msgs[-1]["content"] = teacher_msgs[-1]["content"] + MCQ_SUFFIX
    out = await llm.chat(teacher_msgs, temperature=0.7, max_tokens=1024)
    if not out:
        return {"sample_id": item["sample_id"], "qa_style": style, "blind_target": None}

    blind_letter = None
    if style == "personamem_mcq":
        m = FINAL_RE.search(out)
        blind_letter = m.group(1).upper() if m else None
    elif style == "bigtom_tom":
        m = BIGTOM_RE.search(out)
        blind_letter = (m.group(1) or m.group(2)).lower() if m else None

    stored = item["stored_letter"]
    agree = None
    if blind_letter and stored:
        agree = blind_letter.lower() == stored.lower()
    return {
        "sample_id": item["sample_id"],
        "qa_style": style,
        "rewrite_status": item["rewrite_status"],
        "teacher_model": model,
        "blind_target": out,
        "blind_letter": blind_letter,
        "stored_letter": stored or None,
        "agree": agree,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=60)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--styles", nargs="+", default=["personamem_mcq", "prefeval_gen", "bigtom_tom"])
    args = ap.parse_args()

    items = [i for i in load_items() if i["qa_style"] in args.styles]
    done = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text().splitlines():
            try:
                done.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [i for i in items if i["sample_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"qa items: {len(items)} total, {len(done)} done, {len(todo)} to relabel", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=args.model, max_concurrent=args.concurrency)
    lock = asyncio.Lock()
    n_done = 0

    async def run_one(item):
        nonlocal n_done
        rec = await relabel_item(item, llm, args.model)
        if rec is None:
            return
        async with lock:
            with open(OUT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 200 == 0:
                print(f"[{n_done}/{len(todo)}] llm_calls={llm.calls} tokens={llm.tokens}", flush=True)

    await asyncio.gather(*(run_one(i) for i in todo))
    print(f"DONE: {n_done} items, {llm.calls} calls, {llm.tokens} total tokens", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
