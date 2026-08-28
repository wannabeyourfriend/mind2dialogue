"""Does privileged supervision produce BETTER assistant responses, not just better answers?

Metric C measured knowledge recovery. This measures the thing that actually
determines training-data quality: at a turn where the user holds an undisclosed
deal-breaker, does the teacher's response walk into it?

  blind teacher      persona + conversation so far
  privileged teacher the same + private_context

A judge that sees the deal-breakers (but not which teacher wrote which reply,
order counterbalanced) marks each response for deal-breaker violation and for
which better serves the user.

The privileged teacher is NOT asked to state the private facts — doing so would
train the student to assert things it cannot know. It is asked to act
consistently with them, which is learnable from evidence.

Usage: python scripts/evaluate_response_quality.py --limit 80 --concurrency 12
"""

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from user_simulator.data import LLM, format_conversation  # noqa: E402

OUT = ROOT / "output/relabel_blind/response_quality.jsonl"

TEACHER_PROMPT = """You are a personalized AI assistant.

<user_profile>
{profile}
</user_profile>
{private}
<conversation_so_far>
{history}
</conversation_so_far>

Write your next reply to the user. Be genuinely helpful and specific."""

PRIVATE_BLOCK = """
<what_you_privately_understand_about_this_user>
{p}
</what_you_privately_understand_about_this_user>

Act consistently with this understanding, but do NOT state it back to the user or
reveal that you know it — they have not told you. Let it shape what you propose
and what you avoid proposing.
"""

JUDGE_PROMPT = """A user is talking to an assistant. The user privately refuses the following, and has NOT said so out loud:

<deal_breakers>
{dbs}
</deal_breakers>

<conversation_so_far>
{history}
</conversation_so_far>

Two candidate replies:

<reply_1>
{r1}
</reply_1>

<reply_2>
{r2}
</reply_2>

Answer strict JSON:
{{"reply_1_violates": true/false,
  "reply_2_violates": true/false,
  "better_for_this_user": 1 or 2 or 0,
  "reply_1_reveals_private_knowledge": true/false,
  "reply_2_reveals_private_knowledge": true/false,
  "reason": "one sentence"}}

`violates` = the reply proposes something matching a deal-breaker.
`reveals_private_knowledge` = the reply states or strongly implies a fact the user never told it
(this is a FAULT: it means the assistant claimed knowledge it could not have)."""


def mcnemar(recs, k1, k2):
    b = sum(1 for r in recs if r[k1] and not r[k2])
    c = sum(1 for r in recs if not r[k1] and r[k2])
    if b + c == 0:
        return b, c, float("nan")
    chi = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, math.erfc(math.sqrt(chi / 2))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="output/latent")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--turns-per-conv", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--teacher-model", default="gpt-4o-mini")
    ap.add_argument("--judge-model", default="gpt-4o")
    args = ap.parse_args()

    sessions = []
    for p in sorted(Path(args.corpus).rglob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if d.get("private_items") and (d.get("private_context") or {}).get("deal_breakers"):
            sessions.append(d)
    sessions = sessions[: args.limit]
    print(f"sessions: {len(sessions)}")
    if not sessions:
        return

    teacher = LLM(model=args.teacher_model, max_concurrent=args.concurrency)
    judge = LLM(model=args.judge_model, max_concurrent=args.concurrency)
    lock = asyncio.Lock()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(OUT, "w").close()
    recs = []

    async def one(s, idx):
        conv = s["conversation"]
        profile = s.get("profile_summary", "")
        private = "\n".join(f"- {x}" for x in s["private_items"])
        dbs = "\n".join(f"- {x}" for x in s["private_context"]["deal_breakers"])
        asst_idx = [i for i, m in enumerate(conv) if m["role"] == "assistant"]
        picks = asst_idx[1 : 1 + args.turns_per_conv]
        for t_i, i in enumerate(picks):
            hist = format_conversation(conv[:i])

            async def gen(privileged):
                return await teacher.chat(
                    [
                        {
                            "role": "user",
                            "content": TEACHER_PROMPT.format(
                                profile=profile[:2000],
                                private=PRIVATE_BLOCK.format(p=private[:3000])
                                if privileged
                                else "",
                                history=hist[-9000:],
                            ),
                        }
                    ],
                    temperature=0.7,
                    max_tokens=700,
                )

            blind_r, priv_r = await asyncio.gather(gen(False), gen(True))
            if not blind_r or not priv_r:
                continue
            swap = (idx + t_i) % 2 == 1
            r1, r2 = (priv_r, blind_r) if swap else (blind_r, priv_r)
            raw = await judge.chat(
                [
                    {
                        "role": "user",
                        "content": JUDGE_PROMPT.format(
                            dbs=dbs, history=hist[-8000:], r1=r1[:3000], r2=r2[:3000]
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=400,
                json_mode=True,
            )
            try:
                v = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

            def unmap(field1, field2):
                return (v.get(field2), v.get(field1)) if swap else (v.get(field1), v.get(field2))

            blind_viol, priv_viol = unmap("reply_1_violates", "reply_2_violates")
            blind_rev, priv_rev = unmap(
                "reply_1_reveals_private_knowledge", "reply_2_reveals_private_knowledge"
            )
            better = v.get("better_for_this_user")
            better_arm = (
                "tie"
                if better not in (1, 2)
                else ("privileged" if (better == 1) == swap else "blind")
            )
            rec = {
                "persona_id": s["persona_id"],
                "turn": i,
                "blind_violates": bool(blind_viol),
                "priv_violates": bool(priv_viol),
                "blind_reveals": bool(blind_rev),
                "priv_reveals": bool(priv_rev),
                "better": better_arm,
            }
            async with lock:
                recs.append(rec)
                with open(OUT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(s, i) for i, s in enumerate(sessions)))
    if not recs:
        print("no records")
        return

    n = len(recs)
    bv = sum(1 for r in recs if r["blind_violates"])
    pv = sum(1 for r in recs if r["priv_violates"])
    print(f"\n=== response quality (n={n} turn-pairs) ===")
    print(f"deal-breaker violation   blind {bv} ({bv/n:.1%})   privileged {pv} ({pv/n:.1%})")
    b, c, p = mcnemar(recs, "blind_violates", "priv_violates")
    print(f"  McNemar: blind-only={b} priv-only={c} p={p:.4f}")
    br = sum(1 for r in recs if r["blind_reveals"])
    prv = sum(1 for r in recs if r["priv_reveals"])
    print("\nclaims knowledge it cannot have (hallucination risk)")
    print(f"  blind {br} ({br/n:.1%})   privileged {prv} ({prv/n:.1%})")
    from collections import Counter

    cnt = Counter(r["better"] for r in recs)
    print(f"\nbetter for this user: {dict(cnt)}")


if __name__ == "__main__":
    asyncio.run(main())
