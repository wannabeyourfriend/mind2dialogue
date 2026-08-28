"""Build SFT data from the latent corpus with genuine reasoning traces.

Two fixes over the original pipeline:

1. RESPONSES ARE REASONED, NOT JUST CONDITIONED. Both teachers first reason
   explicitly about what the user may be withholding and what to avoid
   proposing, then answer. The scaffold is identical for both arms, so the
   comparison still isolates the private context rather than the prompt shape.

2. QA REASONING DERIVES THE ANSWER INSTEAD OF RATIONALISING IT. The original
   `_write_multiple_choice_reasoning` (qa.py:111) is called AFTER the correct letter is
   chosen and is TOLD that letter, so its "reasoning" is post-hoc justification
   — a student trained on it learns to defend a given answer, not to work one
   out. Here the teacher is never shown the gold: it reasons in <think>...</think>
   and commits to a letter, and we record whether it actually landed on gold.
   Items where a teacher's reasoning fails are kept and flagged, not silently
   fixed, so the supervision reflects real derivations.

Emits, per arm: dialogue-view and QA-view SFT lines in chat schema, plus a
report of deal-breaker violations and derived-answer accuracy.

Usage:
  python scripts/build_latent_supervision_data.py --limit 124 --concurrency 14
"""

import argparse
import asyncio
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from user_simulator.data import LLM, format_conversation  # noqa: E402

OUT_DIR = ROOT / "output/latent_sft"
LETTERS = ["A", "B", "C", "D"]
ANS_RE = re.compile(r"Answer:\s*\(?([A-D])\)?", re.I)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)

RESPONSE_PROMPT = """You are a personalized AI assistant.

<user_profile>
{profile}
</user_profile>
{private}
<conversation_so_far>
{history}
</conversation_so_far>

Before replying, think it through inside <think></think>:
- What has this user NOT said that would change what they need?
- What would a well-meaning assistant propose here that this particular person
  would quietly refuse, and why?
- What should you therefore offer, and what should you steer away from?

Then write the reply itself after </think>. The reply must not mention or imply
anything the user has not told you — act on your understanding without exposing it.

Format exactly:
<think>
your reasoning
</think>
your reply to the user"""

PRIVATE_BLOCK = """
<what_you_privately_understand_about_this_user>
{p}
</what_you_privately_understand_about_this_user>
"""

QA_PROMPT = """You are answering a question about a user you are assisting.

<user_profile>
{profile}
</user_profile>
{private}
<conversation>
{history}
</conversation>

{question}

{options}

Work out the answer inside <think></think> — weigh the options against what you
know about this user and say what rules each one in or out. You are NOT told which
option is correct; derive it. Then commit.

Format exactly:
<think>
your reasoning
</think>
Answer: X"""

JUDGE_PROMPT = """A user privately refuses the following, and has NOT said so out loud:

<deal_breakers>
{dbs}
</deal_breakers>

<conversation_so_far>
{history}
</conversation_so_far>

<reply_1>
{r1}
</reply_1>

<reply_2>
{r2}
</reply_2>

Strict JSON:
{{"reply_1_violates": true/false, "reply_2_violates": true/false,
  "better_for_this_user": 1 or 2 or 0,
  "reply_1_reveals_private_knowledge": true/false,
  "reply_2_reveals_private_knowledge": true/false}}"""


def split_think(text):
    m = THINK_RE.search(text or "")
    if not m:
        return "", (text or "").strip()
    return m.group(1).strip(), (text or "")[m.end():].strip()


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
    ap.add_argument("--limit", type=int, default=124)
    ap.add_argument("--turns-per-conv", type=int, default=2)
    ap.add_argument("--qa-per-conv", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=14)
    ap.add_argument("--teacher-model", default="gpt-4o-mini")
    ap.add_argument("--gen-model", default="gpt-4o")
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
    print(f"sessions: {len(sessions)}", flush=True)
    if not sessions:
        return

    teacher = LLM(model=args.teacher_model, max_concurrent=args.concurrency)
    gen = LLM(model=args.gen_model, max_concurrent=args.concurrency)
    judge = LLM(model=args.judge_model, max_concurrent=args.concurrency)
    lock = asyncio.Lock()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        arm: open(OUT_DIR / f"train_{arm}.jsonl", "w", encoding="utf-8")
        for arm in ("privileged", "blind")
    }
    resp_recs, qa_recs = [], []

    def emit(arm, messages, meta):
        files[arm].write(
            json.dumps({"messages": messages, "metadata": meta}, ensure_ascii=False) + "\n"
        )

    async def do_responses(s, idx):
        conv, profile = s["conversation"], s.get("profile_summary", "")
        private = "\n".join(f"- {x}" for x in s["private_items"])
        dbs = "\n".join(f"- {x}" for x in s["private_context"]["deal_breakers"])
        asst_idx = [i for i, m in enumerate(conv) if m["role"] == "assistant"]
        for t_i, i in enumerate(asst_idx[1 : 1 + args.turns_per_conv]):
            hist = format_conversation(conv[:i])

            async def gen_resp(privileged):
                return await teacher.chat(
                    [
                        {
                            "role": "user",
                            "content": RESPONSE_PROMPT.format(
                                profile=profile[:2000],
                                private=PRIVATE_BLOCK.format(p=private[:3000])
                                if privileged
                                else "",
                                history=hist[-9000:],
                            ),
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1100,
                )

            braw, praw = await asyncio.gather(gen_resp(False), gen_resp(True))
            if not braw or not praw:
                continue
            b_think, b_reply = split_think(braw)
            p_think, p_reply = split_think(praw)
            if not b_reply or not p_reply:
                continue

            swap = (idx + t_i) % 2 == 1
            r1, r2 = (p_reply, b_reply) if swap else (b_reply, p_reply)
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
                max_tokens=300,
                json_mode=True,
            )
            try:
                v = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

            def unmap(f1, f2):
                return (v.get(f2), v.get(f1)) if swap else (v.get(f1), v.get(f2))

            bv, pv = unmap("reply_1_violates", "reply_2_violates")
            brv, prv = unmap(
                "reply_1_reveals_private_knowledge", "reply_2_reveals_private_knowledge"
            )
            better = v.get("better_for_this_user")
            better_arm = (
                "tie"
                if better not in (1, 2)
                else ("privileged" if (better == 1) == swap else "blind")
            )
            meta = {"source": "dialogue", "persona_id": s["persona_id"], "turn": i}
            sysmsg = {
                "role": "system",
                "content": "You are a personalized AI assistant. Reason about what the user "
                "has not said, then respond.",
            }
            for arm, think, reply in (
                ("privileged", p_think, p_reply),
                ("blind", b_think, b_reply),
            ):
                msgs = [sysmsg] + [
                    {"role": m["role"], "content": m["content"]} for m in conv[:i]
                ]
                msgs.append(
                    {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{reply}"}
                )
                async with lock:
                    emit(arm, msgs, {**meta, "arm": arm})
            async with lock:
                resp_recs.append(
                    {
                        "blind_violates": bool(bv),
                        "priv_violates": bool(pv),
                        "blind_reveals": bool(brv),
                        "priv_reveals": bool(prv),
                        "better": better_arm,
                        "blind_think_len": len(b_think),
                        "priv_think_len": len(p_think),
                    }
                )

    async def do_qa(s, idx):
        conv, profile = s["conversation"], s.get("profile_summary", "")
        private = "\n".join(f"- {x}" for x in s["private_items"])
        user_turns = "\n".join(m["content"] for m in conv if m["role"] == "user")
        cut = max(2, len(conv) // 2)
        half = format_conversation(conv[:cut])
        raw = await gen.chat(
            [
                {
                    "role": "user",
                    "content": (
                        "A user is talking to an assistant. Here is what they have NOT said:\n"
                        f"<private_facts>\n{private[:4000]}\n</private_facts>\n\n"
                        f"<what_the_user_said>\n{user_turns[-8000:]}\n</what_the_user_said>\n\n"
                        f"<profile>\n{profile[:2000]}\n</profile>\n\n"
                        f"Write {args.qa_per_conv} multiple-choice questions whose correct answer "
                        "follows from the private facts, is NOT stated in what the user said, and "
                        "does NOT follow from the profile. The three distractors must be exactly as "
                        "plausible to someone lacking the private facts — same kind of thing, same "
                        "length, same specificity.\n\n"
                        'Strict JSON: {"items":[{"question":"...","correct":"...",'
                        '"distractors":["...","...","..."]}]}'
                    ),
                }
            ],
            temperature=0.7,
            max_tokens=1400,
            json_mode=True,
        )
        try:
            items = json.loads(raw)["items"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return
        for j, it in enumerate(items):
            try:
                q, correct = it["question"], it["correct"]
                dis = it["distractors"][:3]
                assert q and correct and len(dis) == 3
            except (KeyError, AssertionError, TypeError):
                continue
            opts = [correct] + dis
            rot = (idx + j) % 4
            order = list(range(4))
            order = order[rot:] + order[:rot]
            shuffled = [opts[o] for o in order]
            gold = LETTERS[order.index(0)]
            opt_txt = "\n".join(f"{LETTERS[k]}) {o}" for k, o in enumerate(shuffled))

            async def answer(privileged):
                return await teacher.chat(
                    [
                        {
                            "role": "user",
                            "content": QA_PROMPT.format(
                                profile=profile[:2000],
                                private=PRIVATE_BLOCK.format(p=private[:3000])
                                if privileged
                                else "",
                                history=half[-6000:],
                                question=q,
                                options=opt_txt,
                            ),
                        }
                    ],
                    temperature=0.3,
                    max_tokens=900,
                )

            braw, praw = await asyncio.gather(answer(False), answer(True))
            b_think, b_tail = split_think(braw or "")
            p_think, p_tail = split_think(praw or "")
            bm, pm = ANS_RE.search(b_tail or ""), ANS_RE.search(p_tail or "")
            if not pm:
                continue
            b_letter = bm.group(1).upper() if bm else None
            p_letter = pm.group(1).upper()
            user_msg = f"{q}\n\n{opt_txt}"
            meta = {
                "source": "qa",
                "persona_id": s["persona_id"],
                "gold": gold,
                "derived_correct": p_letter == gold,
            }
            # supervision carries the DERIVED reasoning and its own answer; items
            # where the teacher derived the wrong letter are flagged, not patched
            for arm, think, letter in (
                ("privileged", p_think, p_letter),
                ("blind", b_think, b_letter),
            ):
                if letter is None:
                    continue
                async with lock:
                    emit(
                        arm,
                        [
                            {
                                "role": "system",
                                "content": "You are a personalized AI assistant. Reason step by "
                                "step, then answer.",
                            },
                            {"role": "user", "content": user_msg},
                            {
                                "role": "assistant",
                                "content": f"<think>\n{think}\n</think>\nAnswer: {letter}",
                            },
                        ],
                        {**meta, "arm": arm, "answer": letter, "correct": letter == gold},
                    )
            async with lock:
                qa_recs.append(
                    {
                        "blind_correct": b_letter == gold if b_letter else False,
                        "priv_correct": p_letter == gold,
                        "blind_think_len": len(b_think),
                        "priv_think_len": len(p_think),
                    }
                )

    await asyncio.gather(
        *([do_responses(s, i) for i, s in enumerate(sessions)]
          + [do_qa(s, i) for i, s in enumerate(sessions)])
    )
    for f in files.values():
        f.close()

    if resp_recs:
        n = len(resp_recs)
        bv = sum(1 for r in resp_recs if r["blind_violates"])
        pv = sum(1 for r in resp_recs if r["priv_violates"])
        b, c, p = mcnemar(resp_recs, "blind_violates", "priv_violates")
        print(f"\n=== responses WITH reasoning (n={n}) ===")
        print(f"deal-breaker violation: blind {bv} ({bv/n:.1%})  privileged {pv} ({pv/n:.1%})")
        print(f"  McNemar blind-only={b} priv-only={c} p={p:.4f}")
        from collections import Counter

        cnt = Counter(r["better"] for r in resp_recs)
        nt = cnt["privileged"] + cnt["blind"]
        if nt:
            from math import comb

            k = cnt["privileged"]
            pb = 2 * sum(comb(nt, i) * 0.5**nt for i in range(k, nt + 1))
            print(f"better: {dict(cnt)}  binomial p={min(pb,1.0):.4f}")
        print(
            f"  reveals private knowledge: blind {sum(1 for r in resp_recs if r['blind_reveals'])}"
            f"  privileged {sum(1 for r in resp_recs if r['priv_reveals'])}"
        )
        print(
            f"  mean <think> chars: blind {sum(r['blind_think_len'] for r in resp_recs)//n}"
            f"  privileged {sum(r['priv_think_len'] for r in resp_recs)//n}"
        )

    if qa_recs:
        n = len(qa_recs)
        bc = sum(1 for r in qa_recs if r["blind_correct"])
        pc = sum(1 for r in qa_recs if r["priv_correct"])
        b, c, p = mcnemar(qa_recs, "blind_correct", "priv_correct")
        print(f"\n=== QA with DERIVED answers (n={n}, teacher never shown gold) ===")
        print(f"derived correctly: blind {bc} ({bc/n:.1%})  privileged {pc} ({pc/n:.1%})")
        print(f"  McNemar blind-only={b} priv-only={c} p={p:.4f}")
        print(
            f"  mean <think> chars: blind {sum(r['blind_think_len'] for r in qa_recs)//n}"
            f"  privileged {sum(r['priv_think_len'] for r in qa_recs)//n}"
        )

    for arm in ("privileged", "blind"):
        pth = OUT_DIR / f"train_{arm}.jsonl"
        print(f"{pth}: {sum(1 for _ in open(pth))} lines")


if __name__ == "__main__":
    asyncio.run(main())
