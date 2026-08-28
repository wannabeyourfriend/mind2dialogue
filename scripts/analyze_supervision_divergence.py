"""Analyze how far blind-teacher supervision diverges from privileged supervision.

Three views, written to output/relabel_blind/divergence_report.json:

1. QA label agreement — blind teacher's answer vs the stored state-derived
   gold letter (personamem_mcq, bigtom_tom), plus the history-stripped probe
   if present.
2. Dialogue lexical divergence — length ratio and unigram Jaccard between the
   privileged and blind response for the same turn, stratified by scenario
   family and turn depth.
3. Dialogue judged divergence (--judge) — an LLM judge sees the history and
   both responses (blinded, order randomized per item) and reports whether
   they differ in the user-state assumptions they make, and which is better
   grounded in the user's actual state.

Usage:
  python scripts/analyze_supervision_divergence.py [--judge 200] [--judge-model gpt-4.1-mini]
"""

import argparse
import asyncio
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from relabel_conversations_with_blind_teacher import load_conversations  # noqa: E402
from user_simulator.data import LLM, format_conversation  # noqa: E402

BLIND_DIALOGUE = ROOT / "output/relabel_blind/dialogue_blind.jsonl"
BLIND_QA = ROOT / "output/relabel_blind/qa_blind.jsonl"
PROBE = ROOT / "output/relabel_blind/probe_history_leak.jsonl"
OUT = ROOT / "output/relabel_blind/divergence_report.json"
JUDGE_OUT = ROOT / "output/relabel_blind/judge_pairs.jsonl"

WORD_RE = re.compile(r"\w+")

JUDGE_PROMPT = """You are comparing two candidate assistant replies at the same point in a conversation.

<conversation_so_far>
{history}
</conversation_so_far>

<reply_1>
{reply1}
</reply_1>

<reply_2>
{reply2}
</reply_2>

Both replies were written by the same model. One author could read a hidden, ground-truth record of the user's current inner state (emotions, beliefs, intentions); the other could only read the conversation above. You are NOT told which is which.

Answer strictly as JSON with these keys:
- "differ_in_state_assumptions": true/false — do the two replies make materially different assumptions about what the user is feeling, believing, or intending?
- "better_grounded": 1, 2, or 0 — which reply is better grounded in what this specific user actually needs right now (0 = no meaningful difference)?
- "guess_privileged": 1, 2, or 0 — which reply do you think had access to the hidden state (0 = cannot tell)?
- "reason": one sentence."""


def toks(s):
    return set(WORD_RE.findall((s or "").lower()))


def jaccard(a, b):
    ta, tb = toks(a), toks(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(len(ta | tb), 1)


def qa_agreement():
    out = {}
    if not BLIND_QA.exists():
        return out
    by = defaultdict(list)
    for line in BLIND_QA.read_text().splitlines():
        r = json.loads(line)
        by[r["qa_style"]].append(r)
    for style, items in by.items():
        parsed = [r for r in items if r.get("agree") is not None]
        if parsed:
            agree = sum(1 for r in parsed if r["agree"])
            out[style] = {
                "n": len(items),
                "n_parsed": len(parsed),
                "n_agree": agree,
                "agreement": round(agree / len(parsed), 4),
            }
        else:
            out[style] = {"n": len(items), "note": "free-form, no label to compare"}
    if PROBE.exists():
        recs = [json.loads(line) for line in PROBE.read_text().splitlines()]
        p = [r for r in recs if r["agree"] is not None]
        if p:
            out["personamem_mcq_history_stripped_probe"] = {
                "n_parsed": len(p),
                "agreement": round(sum(1 for r in p if r["agree"]) / len(p), 4),
                "note": "assistant turns removed from replayed history",
            }
    return out


def dialogue_pairs():
    blind = {}
    for line in BLIND_DIALOGUE.read_text().splitlines():
        r = json.loads(line)
        blind[r["content_hash"]] = r
    pairs = []
    for row in load_conversations():
        rec = blind.get(row["hash"])
        if rec is None:
            continue
        msgs = row["messages"]
        conv = msgs[1:]
        asst_idx = [i for i, m in enumerate(conv) if m["role"] == "assistant"]
        for depth, i in enumerate(asst_idx, start=1):
            b = rec["blind_turns"].get(str(i))
            if not b:
                continue
            pairs.append(
                {
                    "hash": row["hash"],
                    "family": row["family"],
                    "turn_index": i,
                    "depth": depth,
                    "n_asst": len(asst_idx),
                    "history": conv[:i],
                    "privileged": conv[i]["content"],
                    "blind": b,
                }
            )
    return pairs


def lexical_stats(pairs):
    def agg(items):
        if not items:
            return None
        js = [jaccard(p["privileged"], p["blind"]) for p in items]
        lp = [len(p["privileged"]) for p in items]
        lb = [len(p["blind"]) for p in items]
        return {
            "n": len(items),
            "mean_jaccard": round(sum(js) / len(js), 4),
            "mean_len_privileged": round(sum(lp) / len(lp)),
            "mean_len_blind": round(sum(lb) / len(lb)),
            "len_ratio_blind_over_priv": round(sum(lb) / max(sum(lp), 1), 4),
        }

    out = {"overall": agg(pairs), "by_family": {}, "by_depth_bin": {}}
    by_fam = defaultdict(list)
    for p in pairs:
        by_fam[p["family"]].append(p)
    for fam, items in sorted(by_fam.items()):
        out["by_family"][fam] = agg(items)
    bins = {"1": lambda d: d == 1, "2-4": lambda d: 2 <= d <= 4, "5-9": lambda d: 5 <= d <= 9, "10+": lambda d: d >= 10}
    for name, fn in bins.items():
        out["by_depth_bin"][name] = agg([p for p in pairs if fn(p["depth"])])
    return out


async def judge_pairs(pairs, n, model, concurrency):
    rng = random.Random(42)
    sample = rng.sample(pairs, min(n, len(pairs)))
    llm = LLM(model=model, max_concurrent=concurrency)
    lock = asyncio.Lock()
    JUDGE_OUT.parent.mkdir(parents=True, exist_ok=True)
    open(JUDGE_OUT, "w").close()
    results = []

    async def one(p, idx):
        swap = (idx % 2) == 1  # counterbalance position
        r1, r2 = (p["blind"], p["privileged"]) if swap else (p["privileged"], p["blind"])
        prompt = JUDGE_PROMPT.format(
            history=format_conversation(p["history"])[-6000:], reply1=r1[:4000], reply2=r2[:4000]
        )
        out = await llm.chat(
            [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=400, json_mode=True
        )
        try:
            v = json.loads(out)
        except (json.JSONDecodeError, TypeError):
            return
        # map positional verdicts back to arm labels
        def to_arm(x):
            if x not in (1, 2):
                return "tie"
            if swap:
                return "blind" if x == 1 else "privileged"
            return "privileged" if x == 1 else "blind"

        rec = {
            "hash": p["hash"],
            "family": p["family"],
            "depth": p["depth"],
            "differ": bool(v.get("differ_in_state_assumptions")),
            "better": to_arm(v.get("better_grounded")),
            "guess_privileged": to_arm(v.get("guess_privileged")),
            "reason": v.get("reason"),
        }
        async with lock:
            with open(JUDGE_OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            results.append(rec)

    await asyncio.gather(*(one(p, i) for i, p in enumerate(sample)))
    n_ok = len(results)
    if not n_ok:
        return {"n": 0}
    better = defaultdict(int)
    guess = defaultdict(int)
    for r in results:
        better[r["better"]] += 1
        guess[r["guess_privileged"]] += 1
    return {
        "n": n_ok,
        "differ_rate": round(sum(1 for r in results if r["differ"]) / n_ok, 4),
        "better_grounded": {k: round(v / n_ok, 4) for k, v in better.items()},
        "judge_identifies_privileged": {k: round(v / n_ok, 4) for k, v in guess.items()},
        "note": "position counterbalanced; judge blind to arm identity; chance = 0.5 for a 2-way guess",
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", type=int, default=0, help="sample size for LLM-judge comparison")
    ap.add_argument("--judge-model", default="gpt-4.1-mini")
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()

    report = {"qa_label_agreement": qa_agreement()}
    pairs = dialogue_pairs()
    report["dialogue_lexical"] = lexical_stats(pairs)
    if args.judge:
        report["dialogue_judged"] = await judge_pairs(
            pairs, args.judge, args.judge_model, args.concurrency
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
