"""PrefEval under one protocol, explicit vs implicit — the control that separates
"learned to infer" from "got generally better".

Both conditions run through identical code, identical prompting and the identical
judge; the ONLY difference is whether the user's preference was ever stated:

  explicit  the preference is handed to the model verbatim before the question.
            Nothing needs inferring, so an arm that merely writes better answers
            gains here too.
  implicit  the preference never appears. It has to be read off a persona and a
            few turns of ordinary conversation — the capability the shared-mind
            supervision is supposed to teach.

The prediction that would support the paper: shared beats blind on implicit and
not much on explicit. Equal gains on both would mean cleaner labels helped, which
is a duller and different claim.

METRIC. The headline is the benchmark's own Preference Adherence Accuracy, not a
bare violation rate. From run.py's stage_accuracy, the three error classes all
carry the factor (violate AND NOT unhelpful) and between them exhaust every
(acknowledged, hallucinated) combination, so the aggregate collapses to

    PAA  <=>  helpful=Yes AND violates=No

Reporting violation alone scores a dodge as a pass: on implicit the preference is
never stated, so "I don't know your preferences, could you tell me?" recommends
nothing, violates nothing, and would count as success. The official helpful judge
fails exactly that response. Since this corpus trains guarded users, a student
that learns to hedge rather than to infer would look good on violation alone —
which is precisely the confound PAA closes.

The judge prompts are the benchmark's own files, used verbatim (three calls per
generation: acknow / violate / helpful), so judge strictness is not ours to be
accused of. `hallucinate` is skipped: it contributes nothing to PAA and only
subdivides the acknowledged-violation bucket.

The stock harness cannot do this — multibench's prefeval stage_gen raises
NotImplementedError for --pref-form implicit — and running explicit through it
while implicit runs here would confound the contrast with harness differences.

Usage:
  python scripts/evaluate_prefeval_implicit_preferences.py --models base blind inferred shared \
      --port 8021 --variant persona-driven
  python scripts/evaluate_prefeval_implicit_preferences.py --rejudge output/prefeval_arms.jsonl
"""

import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = Path(os.environ.get("PREFEVAL_DATA", str(Path.home() / "mind2dialogue/evaluations/data/prefeval")))
OUT = ROOT / "output/prefeval_arms.jsonl"
JUDGE_DIR = ROOT / "evaluations/multibench/benchmarks/prefeval/error_type"

# Verbatim from the benchmark's own judge stage (run.py:355).
JUDGE_SYSTEM = (
    "You are a helpful assistant in evaluating an AI assistant's response. "
    "You should be fair and strict and follow the user's instruction"
)
# hallucinate is deliberately absent: PAA does not depend on it.
JUDGE_METRICS = {
    "acknow": "check_acknowledge.txt",
    "violate": "check_violation.txt",
    "helpful": "check_helpful.txt",
}

ANSWER_RE = re.compile(r"<answer>\s*(yes|no)\s*</answer>", re.IGNORECASE)


def load_judge_prompts(only=None):
    out = {}
    for metric, fname in JUDGE_METRICS.items():
        if only and metric not in only:
            continue
        p = JUDGE_DIR / fname
        if not p.exists():
            raise FileNotFoundError(f"judge prompt missing: {p}")
        out[metric] = p.read_text()
    return out


def parse_answer(text):
    """Official format is <answer>Yes/No</answer>; fall back to a bare token."""
    m = ANSWER_RE.search(text or "")
    if m:
        return m.group(1).lower() == "yes"
    m = re.search(r"\b(yes|no)\b", (text or "").strip(), re.IGNORECASE)
    return m.group(1).lower() == "yes" if m else None


def load_items(variant, topics=None):
    """explicit items have no conversation; implicit ones carry persona + turns."""
    out = []
    if variant == "explicit":
        d = DATA / "explicit_preference"
    else:
        d = DATA / "implicit_preference" / variant
    for f in sorted(d.glob("*.json")):
        if topics and f.stem not in topics:
            continue
        try:
            rows = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        for r in rows:
            item = {
                "topic": f.stem,
                "variant": variant,
                "preference": r.get("preference", ""),
                "question": r.get("question") or r.get("implicit_query", ""),
                "explanation": r.get("explanation", ""),
                "persona": r.get("persona", ""),
                "conversation": r.get("conversation"),
            }
            if item["preference"] and item["question"]:
                out.append(item)
    return out


CHOICE_PROMPT_SUFFIX = """

{options_block}

Which option would you recommend for this user? End your reply with the line
"Final Answer: X" where X is the letter."""

# Ordered most- to least-explicit. A bare leading letter is accepted, but a letter
# floating anywhere in prose is NOT: fine-tuned arms open with sentences like
# "A great option here..." and a naive \b([A-F])\b match reads the article as the
# answer, which silently turns their score into noise.
CHOICE_PATTERNS = [
    re.compile(r"FINAL ANSWER\s*:?\s*\(?([A-F])\b", re.I),
    re.compile(r"\bANSWER\s*(?:IS)?\s*:?\s*\(?([A-F])\b", re.I),
    re.compile(r"\bOPTION\s+\(?([A-F])\b", re.I),
    # A leading letter must be followed by punctuation — "A) ..." is an answer,
    # "A great option..." is a sentence that begins with an article.
    re.compile(r"^\s*\(?([A-F])[).:]", re.I),
    re.compile(r"^\s*([A-F])\s*$", re.I),
]


def parse_choice(txt, options):
    """Letter the reply commits to, or None if it never commits to one."""
    t = (txt or "").strip()
    for pat in CHOICE_PATTERNS:
        m = pat.search(t)
        if m:
            return "ABCDEF".index(m.group(1).upper())
    # Last resort: the reply may name the option instead of lettering it.
    low = t.lower()
    hits = [k for k, o in enumerate(options)
            if len(str(o)) > 25 and str(o)[:60].lower() in low]
    return hits[0] if len(hits) == 1 else None


def load_choice_items(topics=None):
    """choice-based: objective scoring, the aligned option is the answer key.

    Upstream stores the aligned option FIRST in every one of the 1000 items, so
    scoring the file order would hand 100% to any model with a bias toward the
    first letter and measure nothing. Options are permuted per item under a seed
    derived from (topic, row) — deterministic so every arm sees the identical
    ordering, which the paired McNemar over matched items requires.
    """
    out = []
    d = DATA / "implicit_preference" / "choice-based"
    for f in sorted(d.glob("*.json")):
        if topics and f.stem not in topics:
            continue
        try:
            rows = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        for i, r in enumerate(rows):
            opts, aligned = r.get("options"), r.get("aligned_op")
            q = r.get("implicit_query") or r.get("question")
            if not (opts and aligned and q):
                continue
            if isinstance(opts, dict):
                keys = sorted(opts)
                texts = [opts[k] for k in keys]
                gold_idx = keys.index(aligned) if aligned in keys else None
            else:
                texts = list(opts)
                gold_idx = (
                    aligned if isinstance(aligned, int)
                    else (texts.index(aligned) if aligned in texts else None)
                )
            if gold_idx is None:
                continue
            order = list(range(len(texts)))
            random.Random(f"{f.stem}:{i}").shuffle(order)
            texts = [texts[k] for k in order]
            gold_idx = order.index(gold_idx)
            out.append({
                "topic": f.stem, "variant": "choice-based",
                "preference": r.get("preference", ""), "question": q,
                "explanation": r.get("explanation", ""), "persona": r.get("persona", ""),
                "conversation": r.get("conversation"),
                "options": texts, "gold_idx": gold_idx,
            })
    return out


def build_messages(item):
    """Explicit states the preference; implicit only replays the conversation it
    is latent in. Everything else about the prompt is held constant."""
    msgs = []
    if item["variant"] == "explicit":
        msgs.append({"role": "user", "content": item["preference"]})
        msgs.append(
            {"role": "assistant", "content": "Got it — I'll keep that in mind for our conversation."}
        )
    else:
        conv = item["conversation"]
        turns = []
        if isinstance(conv, dict):
            for k in sorted(conv, key=lambda x: int(x) if str(x).isdigit() else 0):
                turns.append(conv[k])
        elif isinstance(conv, list):
            turns = conv
        for t in turns:
            if isinstance(t, dict):
                if t.get("user"):
                    msgs.append({"role": "user", "content": t["user"]})
                if t.get("assistant"):
                    msgs.append({"role": "assistant", "content": t["assistant"]})
            elif isinstance(t, str):
                msgs.append({"role": "user" if len(msgs) % 2 == 0 else "assistant", "content": t})
    msgs.append({"role": "user", "content": item["question"]})
    return msgs


def mcnemar(pairs, k1, k2):
    b = sum(1 for r in pairs if r[k1] and not r[k2])
    c = sum(1 for r in pairs if not r[k1] and r[k2])
    if b + c == 0:
        return b, c, float("nan")
    chi = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, math.erfc(math.sqrt(chi / 2))


async def judge_one(judge, templates, model_name, question, preference, reply, sem):
    """Three official judges, run concurrently. Returns dict or None on failure."""
    async def call(metric):
        prompt = (
            templates[metric]
            .replace("{question}", question)
            .replace("{preference}", preference)
            .replace("{end_generation}", reply)
        )
        async with sem:
            r = await judge.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=300,
            )
        return parse_answer(r.choices[0].message.content)

    metrics = list(templates)          # may be a subset under --fast-judge
    try:
        vals = await asyncio.gather(*(call(m) for m in metrics))
    except Exception as e:
        print(f"  judge fail: {str(e)[:90]}")
        return None
    out = dict(zip(metrics, vals))
    if any(v is None for v in out.values()):
        return None
    out.setdefault("acknow", None)     # diagnostic only; PAA does not use it
    return out


def summarize(recs, models, labels):
    by = defaultdict(list)
    for r in recs:
        by[(r["model"], r["condition"])].append(r)

    print(f"\n{'model':<10}{'condition':<11}{'n':>5}{'PAA':>9}{'violate':>9}"
          f"{'unhelp':>9}{'acknow':>9}")
    paa = {}
    for model in models:
        for label in labels:
            rr = by[(model, label)]
            if not rr:
                continue
            n = len(rr)
            p = sum(1 for r in rr if r["paa"]) / n
            v = sum(1 for r in rr if r["violate"]) / n
            u = sum(1 for r in rr if not r["helpful"]) / n
            has_ack = any(r.get("acknow") is not None for r in rr)
            ack = (f"{sum(1 for r in rr if r['acknow']) / n:>9.1%}" if has_ack
                   else f"{'—':>9}")   # not measured under --fast-judge
            paa[(model, label)] = p
            print(f"{model:<10}{label:<11}{n:>5}{p:>8.1%}{v:>9.1%}{u:>9.1%}{ack}")

    # The unaware/acknowledged split needs the acknowledge judge; under --fast-judge
    # it was never run, and treating a missing verdict as "did not acknowledge" would
    # invent a decomposition rather than report one.
    if any(r.get("acknow") is not None for r in recs):
        print("\n=== error decomposition (share of all items) ===")
        print(f"{'model':<10}{'condition':<11}{'unhelpful':>11}{'unaware-viol':>14}{'ack-viol':>10}")
        for model in models:
            for label in labels:
                rr = [r for r in by[(model, label)] if r.get("acknow") is not None]
                if not rr:
                    continue
                n = len(rr)
                unhelp = sum(1 for r in rr if not r["helpful"]) / n
                unaware = sum(1 for r in rr if r["violate"] and r["helpful"] and not r["acknow"]) / n
                ackv = sum(1 for r in rr if r["violate"] and r["helpful"] and r["acknow"]) / n
                print(f"{model:<10}{label:<11}{unhelp:>10.1%}{unaware:>14.1%}{ackv:>10.1%}")
    else:
        print("\n(error decomposition skipped — acknowledge judge not run)")

    print("\n=== the contrast that matters (PAA, higher is better) ===")
    for model in models:
        if model == "blind":
            continue
        for label in labels:
            if (model, label) in paa and ("blind", label) in paa:
                d = paa[(model, label)] - paa[("blind", label)]
                print(f"  {model} vs blind on {label:<9}: {d:+.1%}")

    print("\npaired McNemar on PAA vs blind (matched items):")
    for model in models:
        if model == "blind":
            continue
        for label in labels:
            a = {(r["topic"], r["idx"]): r["paa"] for r in by[(model, label)]}
            b = {(r["topic"], r["idx"]): r["paa"] for r in by[("blind", label)]}
            keys = set(a) & set(b)
            if not keys:
                continue
            pairs = [{"m": a[k], "b": b[k]} for k in keys]
            gain, loss, p = mcnemar(pairs, "m", "b")
            print(f"  {model} vs blind, {label:<9}: {model}-only-passes={gain} "
                  f"blind-only-passes={loss} p={p:.4f}")


async def rejudge(path, args):
    """Re-score stored generations without touching the GPU. The reason replies
    are persisted at all: revising the judge should never cost a regeneration."""
    from openai import AsyncOpenAI

    judge = AsyncOpenAI()
    templates = load_judge_prompts(["violate", "helpful"] if args.fast_judge else None)
    sem = asyncio.Semaphore(args.concurrency)

    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    # choice-based rows now carry their reply text so the parse can be audited, but
    # they are scored against the answer key, not by the preference judge.
    todo = [r for r in rows if r.get("reply") is not None and r.get("condition") != "choice"]
    keep = [r for r in rows if r.get("reply") is None or r.get("condition") == "choice"]
    if args.max_items:
        # Subset evenly per (model, condition) so arms stay matched on the same items.
        seen = defaultdict(int)
        sub = []
        for r in todo:
            k = (r["model"], r["condition"])
            if seen[k] < args.max_items:
                seen[k] += 1
                sub.append(r)
        print(f"capping to {args.max_items}/condition: {len(todo)} -> {len(sub)}")
        todo = sub
    print(f"re-judging {len(todo)} stored generations with {len(templates)} judge(s) each "
          f"= {len(todo)*len(templates)} calls ({len(keep)} objective rows kept as-is)")
    if not todo:
        print("no stored replies — the file predates reply persistence, regeneration required")
        return

    async def one(r):
        v = await judge_one(judge, templates, args.judge_model,
                            r["question"], r["preference"], r["reply"], sem)
        if v is None:
            return None
        r.update(v)
        r["paa"] = (not v["violate"]) and v["helpful"]
        return r

    done = [r for r in await asyncio.gather(*(one(r) for r in todo)) if r]
    recs = keep + done
    out = Path(path).with_suffix(".rejudged.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out}")

    models = sorted({r["model"] for r in recs})
    labels = [c for c in ["explicit", "implicit", "choice"]
              if any(r["condition"] == c for r in recs)]
    summarize(recs, models, labels)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["base", "blind", "inferred", "shared"])
    ap.add_argument("--variant", default="persona-driven")
    ap.add_argument("--port", type=int, default=8021)
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--choice-only", action="store_true",
                    help="run only the judge-free choice-based subset")
    ap.add_argument("--gen-only", action="store_true",
                    help="generate and store replies without judging; score later with "
                         "--rejudge. For GPU hosts that cannot reach the judge API.")
    ap.add_argument("--rejudge", metavar="JSONL",
                    help="re-score an existing run's stored replies, no generation")
    ap.add_argument("--out", default=str(OUT),
                    help="where to write generations; give each concurrent run its own "
                         "path or they silently overwrite each other")
    ap.add_argument("--fast-judge", action="store_true",
                    help="run only the two judges PAA needs (violate, helpful), dropping "
                         "the acknowledge diagnostic. Cuts judge calls by a third — worth "
                         "it when the link to the judge API is the bottleneck.")
    args = ap.parse_args()

    if args.rejudge:
        await rejudge(args.rejudge, args)
        return

    from openai import AsyncOpenAI

    local = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="not-needed")
    # Both are unused when generation is split from scoring, and the judge prompts
    # may not even be present on the GPU host — so don't touch them.
    need_judge = not (args.gen_only or args.choice_only)
    judge = AsyncOpenAI() if need_judge else None
    templates = (load_judge_prompts(["violate", "helpful"] if args.fast_judge else None)
                 if need_judge else None)

    conditions = [] if args.choice_only else [("explicit", "explicit"), ("implicit", args.variant)]
    all_items = {}
    for label, variant in conditions:
        items = load_items(variant)
        if args.max_items:
            items = items[: args.max_items]
        all_items[label] = items
        print(f"{label}: {len(items)} items")
    choice_items = load_choice_items()
    if args.max_items:
        choice_items = choice_items[: args.max_items]
    print(f"choice-based: {len(choice_items)} items (objective scoring, no judge)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    open(out_path, "w").close()
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    recs = []

    async def write(rec):
        async with lock:
            recs.append(rec)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    async def one(model, label, item, idx):
        async with sem:
            try:
                r = await local.chat.completions.create(
                    model=model, messages=build_messages(item),
                    temperature=0.0, max_tokens=700
                )
                reply = r.choices[0].message.content or ""
            except Exception as e:
                print(f"  gen fail {model}/{label}: {str(e)[:90]}")
                return
        if not reply.strip():
            return
        rec = {
            "model": model, "condition": label, "topic": item["topic"], "idx": idx,
            "question": item["question"], "preference": item["preference"],
            "reply": reply,
        }
        if args.gen_only:
            await write(rec)
            return
        v = await judge_one(judge, templates, args.judge_model,
                            item["question"], item["preference"], reply[:3000], sem)
        if v is None:
            return
        rec.update(acknow=v["acknow"], violate=v["violate"], helpful=v["helpful"],
                   paa=(not v["violate"]) and v["helpful"])
        await write(rec)

    LETTERS = ["A", "B", "C", "D", "E", "F"]

    async def one_choice(model, item, idx):
        async with sem:
            msgs = build_messages(item)
            block = "\n".join(
                f"{LETTERS[k]}) {o}" for k, o in enumerate(item["options"][: len(LETTERS)])
            )
            msgs[-1] = {
                "role": "user",
                "content": item["question"] + CHOICE_PROMPT_SUFFIX.format(options_block=block),
            }
            try:
                r = await local.chat.completions.create(
                    model=model, messages=msgs, temperature=0.0, max_tokens=160
                )
                txt = r.choices[0].message.content or ""
            except Exception as e:
                print(f"  choice gen fail {model}: {str(e)[:80]}")
                return
        picked = parse_choice(txt, item["options"])
        # Non-compliance is recorded, never dropped. Silently discarding the replies
        # that fail to parse leaves each arm scored on a different, self-selected
        # subset — which is how an arm answering in prose ends up compared against
        # one answering in letters.
        correct = picked is not None and picked == item["gold_idx"]
        # No judge here: picking the aligned option is the whole criterion, so a
        # choice is helpful by construction and PAA reduces to correctness.
        await write({
            "model": model, "condition": "choice", "topic": item["topic"], "idx": idx,
            "reply": txt, "picked": picked, "committed": picked is not None,
            "acknow": correct, "violate": not correct, "helpful": True, "paa": correct,
        })

    jobs = []
    for model in args.models:
        for label, _ in conditions:
            for i, item in enumerate(all_items[label]):
                jobs.append(one(model, label, item, i))
        for i, item in enumerate(choice_items):
            jobs.append(one_choice(model, item, i))
    print(f"running {len(jobs)} generations{'' if args.gen_only else ' + judgements'}")
    await asyncio.gather(*jobs)

    if args.gen_only:
        n_unjudged = sum(1 for r in recs if "paa" not in r)
        print(f"\nwrote {len(recs)} records to {out_path} "
              f"({n_unjudged} awaiting judgement)")
        print(f"score them where the judge API is reachable:\n"
              f"  python {Path(__file__).name} --rejudge {out_path}")
        return

    summarize(recs, args.models, [c[0] for c in conditions] + ["choice"])


if __name__ == "__main__":
    asyncio.run(main())
