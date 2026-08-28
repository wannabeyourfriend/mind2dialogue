"""Privileged guidance, observable justification — trainable ToM supervision.

The problem this fixes: a privileged teacher writes reasoning that cites the
private context ("they are dealing with budget constraints"), but the student
never sees that context. Training on it teaches the student to assert facts it
cannot know — confabulation that looks fine in-distribution and breaks outside it.

The fix, in three stages:

  1. GUIDE  — the teacher sees the private context, so it knows which option is
              correct. Privilege determines the TARGET, not the justification.
  2. JUSTIFY— the teacher must then argue for that answer citing ONLY what is
              visible in the conversation: things the user actually said, chose,
              declined, or asked about. It is explicitly forbidden to reference
              anything it was told privately.
  3. VERIFY — a blind checker, holding only the conversation and the reasoning,
              rules on whether every claim is traceable to the transcript and
              whether the reasoning actually supports the answer. Items that fail
              are DROPPED, not repaired: an answer that cannot be justified from
              evidence is not something a student can learn to derive.

What survives is supervision a state-blind student can actually reproduce:
infer the hidden state from observable behaviour, which is the capability the
paper claims to teach.

Usage:
  python scripts/build_grounded_supervision_data.py --corpus output/latent_full --limit 400
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from user_simulator.data import LLM, format_conversation  # noqa: E402

OUT_DIR = ROOT / "output/grounded_sft"
LETTERS = ["A", "B", "C", "D"]

GEN_PROMPT = """A user is talking to an assistant. Here is what the user has NOT said out loud:

<private_facts>
{private}
</private_facts>

<conversation>
{history}
</conversation>

Write {n} multiple-choice questions about this user's hidden state — what they are
constrained by, what they would refuse, what they already tried.

CRITICAL: the correct answer must be REACHABLE from the conversation. There must be
real evidence in what the user said, chose, avoided, or asked about that a careful
reader could follow to it. The private facts tell you what is TRUE; the conversation
must contain the trail. If a fact leaves no trace in the conversation, do not write a
question about it.

Distractors must be equally plausible to a careless reader and must be ruled out by
that same evidence.

Strict JSON:
{{"items": [{{"question": "...", "correct": "...", "distractors": ["...","...","..."],
  "evidence_in_conversation": "what the user said or did that points to the correct answer"}}]}}"""

JUSTIFY_PROMPT = """<conversation>
{history}
</conversation>

{question}

{options}

The correct answer is {gold}.

Write the reasoning that leads to it, inside <think></think>.

HARD CONSTRAINT: cite only what is visible in the conversation above — what the user
said, asked, chose, avoided, or reacted to. You may infer beyond it, but every inference
must start from something quotable. You must NOT state any fact about the user that does
not have a trace in the conversation, and you must NOT refer to knowing anything privately.

Walk through why the evidence favours {gold} and rules out the others.

Format exactly:
<think>
your reasoning
</think>
Answer: {gold}"""

VERIFY_PROMPT = """<conversation>
{history}
</conversation>

A question about the user, and someone's reasoning for their answer:

{question}

{options}

<reasoning>
{reasoning}
</reasoning>

You can see ONLY the conversation above. Judge the reasoning:

Strict JSON:
{{"all_claims_traceable": true/false,
  "unsupported_claims": ["any claim about the user that the conversation does not support"],
  "reasoning_supports_answer": true/false,
  "a_careful_reader_could_reach_this": true/false}}

`all_claims_traceable` = every factual claim about the user can be tied to something in
the conversation. Mark false if the reasoning asserts knowledge that is simply not there."""


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="output/latent_full")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--per-conv", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--gen-model", default="gpt-4o")
    ap.add_argument("--teacher-model", default="gpt-4o-mini")
    ap.add_argument("--verify-model", default="gpt-4o")
    args = ap.parse_args()

    sessions = []
    for p in sorted(Path(args.corpus).rglob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if d.get("private_items") and d.get("conversation"):
            sessions.append(d)
    sessions = sessions[: args.limit]
    print(f"sessions: {len(sessions)}", flush=True)
    if not sessions:
        return

    gen = LLM(model=args.gen_model, max_concurrent=args.concurrency)
    teacher = LLM(model=args.teacher_model, max_concurrent=args.concurrency)
    verifier = LLM(model=args.verify_model, max_concurrent=args.concurrency)
    lock = asyncio.Lock()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fout = open(OUT_DIR / "train_grounded.jsonl", "w", encoding="utf-8")
    frej = open(OUT_DIR / "rejected.jsonl", "w", encoding="utf-8")
    stats = {"generated": 0, "no_justify": 0, "untraceable": 0, "weak_support": 0, "kept": 0}

    async def one(s, idx):
        conv = s["conversation"]
        history = format_conversation(conv)
        private = "\n".join(f"- {x}" for x in s["private_items"])
        raw = await gen.chat(
            [
                {
                    "role": "user",
                    "content": GEN_PROMPT.format(
                        private=private[:4000], history=history[-10000:], n=args.per_conv
                    ),
                }
            ],
            temperature=0.7,
            max_tokens=1800,
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
            async with lock:
                stats["generated"] += 1
            opts = [correct] + dis
            rot = (idx + j) % 4
            order = list(range(4))
            order = order[rot:] + order[:rot]
            shuffled = [opts[o] for o in order]
            gold = LETTERS[order.index(0)]
            opt_txt = "\n".join(f"{LETTERS[k]}) {o}" for k, o in enumerate(shuffled))

            just = await teacher.chat(
                [
                    {
                        "role": "user",
                        "content": JUSTIFY_PROMPT.format(
                            history=history[-10000:],
                            question=q,
                            options=opt_txt,
                            gold=gold,
                        ),
                    }
                ],
                temperature=0.4,
                max_tokens=900,
            )
            if not just or "<think>" not in just:
                async with lock:
                    stats["no_justify"] += 1
                continue
            think = just.split("<think>", 1)[1].split("</think>")[0].strip()

            vraw = await verifier.chat(
                [
                    {
                        "role": "user",
                        "content": VERIFY_PROMPT.format(
                            history=history[-10000:],
                            question=q,
                            options=opt_txt,
                            reasoning=think[:4000],
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=500,
                json_mode=True,
            )
            try:
                v = json.loads(vraw)
            except (json.JSONDecodeError, TypeError):
                continue

            rec = {
                "persona_id": s["persona_id"],
                "question": q,
                "gold": gold,
                "traceable": bool(v.get("all_claims_traceable")),
                "supports": bool(v.get("reasoning_supports_answer")),
                "reachable": bool(v.get("a_careful_reader_could_reach_this")),
                "unsupported": v.get("unsupported_claims") or [],
            }
            if not rec["traceable"]:
                async with lock:
                    stats["untraceable"] += 1
                    frej.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            if not (rec["supports"] and rec["reachable"]):
                async with lock:
                    stats["weak_support"] += 1
                    frej.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            async with lock:
                stats["kept"] += 1
                fout.write(
                    json.dumps(
                        {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a personalized AI assistant. Infer what the "
                                    "user has not said from what they have, then answer.",
                                },
                                {"role": "user", "content": f"{q}\n\n{opt_txt}"},
                                {
                                    "role": "assistant",
                                    "content": f"<think>\n{think}\n</think>\nAnswer: {gold}",
                                },
                            ],
                            "metadata": {
                                "source": "qa_grounded",
                                "persona_id": s["persona_id"],
                                "gold": gold,
                                "evidence": (it.get("evidence_in_conversation") or "")[:400],
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    await asyncio.gather(*(one(s, i) for i, s in enumerate(sessions)))
    fout.close()
    frej.close()
    g = max(stats["generated"], 1)
    print(json.dumps(stats, indent=1))
    print(f"\ngroundedness yield: {stats['kept']}/{stats['generated']} = {stats['kept']/g:.1%}")
    print(f"  dropped as untraceable (would train confabulation): {stats['untraceable']/g:.1%}")
    print(f"  dropped as unreachable from evidence: {stats['weak_support']/g:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
