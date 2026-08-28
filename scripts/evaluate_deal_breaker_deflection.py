"""Student-level deal-breaker deflection — the metric the s4 corpus actually teaches.

Every earlier student-level evaluation asked a different question than the training
data answers. PrefEval implicit tests inferring an unstated *preference*; the s4
corpus teaches steering around things a user would *refuse* — constraints they
never say out loud because saying them would expose a vulnerability. A null on the
former says nothing about whether the latter transferred.

Here each held-out session carries a private_context sampled BEFORE the dialogue
existed, so its deal-breakers causally shaped how the user behaved but were never
stated. The student sees only the conversation. If privileged supervision taught it
to read hidden state, it should avoid recommending the very things this user would
refuse.

Sessions come from output/latent_heldout — generated after training, with fresh
private contexts, so no deal-breaker in this set was ever a training target.

Split like the PrefEval evaluator: generate where the GPU is, judge where the API
is reachable.

  # on the GPU host
  python scripts/evaluate_deal_breaker_deflection.py --models base blind inferred shared \
      --port 8021 --gen-only
  # wherever the judge API works
  python scripts/evaluate_deal_breaker_deflection.py --rejudge output/dealbreaker.jsonl
"""

import argparse
import asyncio
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "output/latent_heldout"
OUT = ROOT / "output/dealbreaker.jsonl"

JUDGE_PROMPT = """A user asked an assistant for advice. Unknown to the assistant, this
user would refuse the following — these are things they will not do, though they never
said so out loud:

<would_refuse>
{dbs}
</would_refuse>

They asked:
<question>{question}</question>

The assistant replied:
<reply>{reply}</reply>

Did the reply recommend, suggest, or urge the user toward ANY of the things they would
refuse? Advocating it counts even if hedged. Merely mentioning something in order to
rule it out does not count.

Strict JSON:
{{"hits": true/false, "which": "<the refused item it pushed, or empty>"}}"""


def load_sessions(limit=0):
    """Held-out sessions that carry both a conversation and deal-breakers."""
    out = []
    for f in sorted(CORPUS.glob("*/*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        dbs = ((d.get("private_context") or {}).get("deal_breakers")) or []
        conv = d.get("conversation") or []
        if not dbs or len(conv) < 4:
            continue
        # Probe at the OPENING turn, with no history. Deal-breakers are sampled as
        # things a well-meaning assistant would plausibly suggest, so they are only a
        # live temptation before the user has steered anyone away from them. Probing
        # at the last turn instead measured nothing: a pilot with an untrained model
        # hit 0/8 there, versus 25% at the opening.
        idx = next((i for i, m in enumerate(conv) if m.get("role") == "user"), None)
        if idx is None:
            continue
        out.append({
            "id": f"{d.get('persona_id','?')}::{f.stem}",
            "history": [],
            "question": conv[idx]["content"],
            "deal_breakers": [str(x) for x in dbs],
        })
        if limit and len(out) >= limit:
            break
    return out


def mcnemar(pairs, k1, k2):
    b = sum(1 for r in pairs if r[k1] and not r[k2])
    c = sum(1 for r in pairs if not r[k1] and r[k2])
    if b + c == 0:
        return b, c, float("nan")
    chi = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, math.erfc(math.sqrt(chi / 2))


async def judge_one(judge, model, item, reply, sem):
    async with sem:
        try:
            r = await judge.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    dbs="\n".join(f"- {d}" for d in item["deal_breakers"]),
                    question=item["question"][:1500],
                    reply=reply[:3000],
                )}],
                temperature=0.0, max_tokens=200,
                response_format={"type": "json_object"},
            )
            v = json.loads(r.choices[0].message.content)
        except Exception as e:
            print(f"  judge fail: {str(e)[:90]}")
            return None
    return {"hits": bool(v.get("hits")), "which": str(v.get("which", ""))[:200]}


def summarize(recs, models):
    by = defaultdict(list)
    for r in recs:
        by[r["model"]].append(r)
    print(f"\n{'model':<12}{'n':>6}{'deal-breaker hit rate':>24}")
    for m in models:
        rr = by[m]
        if not rr:
            continue
        print(f"{m:<12}{len(rr):>6}{sum(1 for r in rr if r['hits'])/len(rr):>23.1%}")

    ref = "blind" if by.get("blind") else "base"
    print(f"\npaired McNemar vs {ref} (lower hit rate is better):")
    for m in models:
        if m == ref:
            continue
        a = {r["id"]: r["hits"] for r in by[m]}
        b = {r["id"]: r["hits"] for r in by[ref]}
        keys = set(a) & set(b)
        if not keys:
            continue
        pairs = [{"m": a[k], "r": b[k]} for k in keys]
        x, y, p = mcnemar(pairs, "r", "m")
        print(f"  {m:<10} vs {ref:<10}: {ref}-only-hits={x} {m}-only-hits={y} p={p:.4f}")


async def rejudge(path, args):
    from openai import AsyncOpenAI

    judge = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    todo = [r for r in rows if r.get("reply")]
    print(f"re-judging {len(todo)} stored replies at concurrency {args.concurrency}")

    async def one(r):
        v = await judge_one(judge, args.judge_model, r, r["reply"], sem)
        if v is None:
            return None
        r.update(v)
        return r

    done = [r for r in await asyncio.gather(*(one(r) for r in todo)) if r]
    out = Path(path).with_suffix(".judged.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for r in done:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
    summarize(done, sorted({r["model"] for r in done}))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["base", "blind", "inferred", "shared"])
    ap.add_argument("--port", type=int, default=8021)
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT),
                    help="where to write generations; give each concurrent run its own "
                         "path or they silently overwrite each other")
    ap.add_argument("--gen-only", action="store_true")
    ap.add_argument("--rejudge", metavar="JSONL")
    args = ap.parse_args()

    if args.rejudge:
        await rejudge(args.rejudge, args)
        return

    from openai import AsyncOpenAI

    local = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="not-needed")
    items = load_sessions(args.limit)
    print(f"{len(items)} held-out sessions, "
          f"{sum(len(i['deal_breakers']) for i in items)} deal-breakers")
    if not items:
        raise SystemExit(f"no usable sessions under {CORPUS}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    open(out_path, "w").close()
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    recs = []

    async def one(model, item):
        async with sem:
            try:
                r = await local.chat.completions.create(
                    model=model,
                    messages=item["history"] + [{"role": "user", "content": item["question"]}],
                    temperature=0.0, max_tokens=700,
                )
                reply = r.choices[0].message.content or ""
            except Exception as e:
                print(f"  gen fail {model}: {str(e)[:80]}")
                return
        if not reply.strip():
            return
        rec = {"model": model, "id": item["id"], "question": item["question"],
               "deal_breakers": item["deal_breakers"], "reply": reply}
        async with lock:
            recs.append(rec)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(m, it) for m in args.models for it in items))
    print(f"\nwrote {len(recs)} generations to {out_path}")
    if args.gen_only:
        print(f"score them where the judge API works:\n"
              f"  python {Path(__file__).name} --rejudge {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
