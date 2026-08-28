"""Three-arm supervision for the shared-mind ablation.

  shared    teacher reads the simulator's TRUE private context
  inferred  teacher reads a private-context ESTIMATE produced by a blind model
            from the conversation alone
  blind     teacher reads only the conversation

`inferred` is the arm that tests the paper's actual claim. `shared` vs `blind`
only shows that a latent state helps; it cannot distinguish "sharing one mind"
from "any decent guess about the user". If shared ≈ inferred, the Oracle does not
need privileged access — it can infer what it needs, and the shared-mind framing
is unnecessary. Only shared > inferred supports the title.

Every arm answers on its own (never shown gold), so per-arm label accuracy is
measured rather than assumed. Every arm's reasoning then passes the same
groundedness verifier: a blind checker holding only the conversation rules on
whether each claim is traceable. Items whose reasoning asserts unsupported facts
are dropped from that arm — otherwise the privileged arm wins by citing things
the student can never see, which trains confabulation rather than inference.

Usage:
  python scripts/build_three_arm_supervision.py --corpus output/latent_full --limit 900
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

OUT_DIR = ROOT / "output/three_arm_para"
LETTERS = ["A", "B", "C", "D"]
ANS_RE = re.compile(r"Answer:\s*\(?([A-D])\)?", re.I)
ARMS = ["shared", "inferred", "blind"]

ESTIMATE_PROMPT = """Read this conversation between a user and an assistant.

<conversation>
{history}
</conversation>

<persona_profile>
{profile}
</persona_profile>

Infer what this user has NOT said out loud but is probably true of them: constraints
they are under, things they have already tried, and suggestions they would refuse.
Base it only on what is visible — what they asked, avoided, accepted, or pushed back on.

Strict JSON:
{{"hidden_constraints": ["..."], "prior_experience": ["..."], "deal_breakers": ["..."]}}"""

GEN_PROMPT = """A user is talking to an assistant. Here is what the user has NOT said out loud:

<private_facts>
{private}
</private_facts>

<conversation>
{history}
</conversation>

Write {n} multiple-choice questions about this user's hidden state.

CRITICAL: the correct answer must be REACHABLE from the conversation — there must be
real evidence in what the user said, chose, avoided, or asked about that a careful reader
could follow. The private facts tell you what is TRUE; the conversation must contain the
trail. If a fact leaves no trace, do not write a question about it.

Distractors must be equally plausible to a careless reader and ruled out by that evidence.

Strict JSON:
{{"items": [{{"question": "...", "correct": "...", "distractors": ["...","...","..."]}}]}}"""

ANSWER_PROMPT = """<persona_profile>
{profile}
</persona_profile>
{extra}
<conversation>
{history}
</conversation>

{question}

{options}

Work out the answer inside <think></think>. You are NOT told which option is correct.

HARD CONSTRAINT on the reasoning: cite only what is visible in the conversation — what
the user said, asked, chose, avoided, or reacted to. Every inference must start from
something quotable. Do not assert any fact about the user that has no trace in the
conversation, and never refer to knowing anything privately.

Format exactly:
<think>
your reasoning
</think>
Answer: X"""

SHARED_BLOCK = """
<what_you_privately_understand_about_this_user>
{p}
</what_you_privately_understand_about_this_user>
Use this to work out WHICH option is right, but justify it only from the conversation.
"""

INFERRED_BLOCK = """
<your_read_on_this_user_inferred_from_the_conversation>
{p}
</your_read_on_this_user_inferred_from_the_conversation>
"""


PARAPHRASE_PROMPT = """Rewrite this multiple-choice question and its four options in
different words. Keep the meaning, the difficulty, and which option is correct exactly
the same. Do not add or remove information, and do not make any option easier to spot.

{question}

{options}

Strict JSON: {{"question": "...", "options": ["A text", "B text", "C text", "D text"]}}"""

VERIFY_PROMPT = """<conversation>
{history}
</conversation>

{question}

{options}

<reasoning>
{reasoning}
</reasoning>

You can see ONLY the conversation. Strict JSON:
{{"all_claims_traceable": true/false,
  "unsupported_claims": ["claims about the user the conversation does not support"],
  "reasoning_supports_answer": true/false}}"""


def split_think(t):
    if not t or "<think>" not in t:
        return None, None
    think = t.split("<think>", 1)[1].split("</think>")[0].strip()
    m = ANS_RE.search(t.split("</think>")[-1] if "</think>" in t else t)
    return think, (m.group(1).upper() if m else None)


def mcnemar(recs, k1, k2):
    b = sum(1 for r in recs if r.get(k1) and not r.get(k2))
    c = sum(1 for r in recs if not r.get(k1) and r.get(k2))
    if b + c == 0:
        return b, c, float("nan")
    chi = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, math.erfc(math.sqrt(chi / 2))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="output/latent_full")
    ap.add_argument("--limit", type=int, default=900)
    ap.add_argument("--per-conv", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=18)
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
    files = {a: open(OUT_DIR / f"train_{a}.jsonl", "w", encoding="utf-8") for a in ARMS}
    recs = []
    kept = {a: 0 for a in ARMS}
    dropped = {a: 0 for a in ARMS}

    async def one(s, idx):
        conv = s["conversation"]
        history = format_conversation(conv)
        profile = s.get("profile_summary", "")
        true_private = "\n".join(f"- {x}" for x in s["private_items"])

        est_raw = await teacher.chat(
            [
                {
                    "role": "user",
                    "content": ESTIMATE_PROMPT.format(
                        history=history[-10000:], profile=profile[:2000]
                    ),
                }
            ],
            temperature=0.5,
            max_tokens=700,
            json_mode=True,
        )
        try:
            e = json.loads(est_raw)
            est_items = (
                list(e.get("hidden_constraints", []))
                + list(e.get("prior_experience", []))
                + list(e.get("deal_breakers", []))
            )
        except (json.JSONDecodeError, TypeError, AttributeError):
            est_items = []
        est_private = "\n".join(f"- {x}" for x in est_items) or "(nothing inferred)"

        raw = await gen.chat(
            [
                {
                    "role": "user",
                    "content": GEN_PROMPT.format(
                        private=true_private[:4000], history=history[-10000:], n=args.per_conv
                    ),
                }
            ],
            temperature=0.7,
            max_tokens=2200,
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

            praw = await gen.chat(
                [{"role": "user",
                  "content": PARAPHRASE_PROMPT.format(question=q, options=opt_txt)}],
                temperature=0.5, max_tokens=700, json_mode=True,
            )
            try:
                pv = json.loads(praw)
                pq, popts = pv["question"], pv["options"]
                assert pq and len(popts) == 4
                q = pq
                opt_txt = "\n".join(f"{LETTERS[k]}) {o}" for k, o in enumerate(popts))
            except (json.JSONDecodeError, KeyError, AssertionError, TypeError):
                pass

            async def run_arm(arm):
                extra = {
                    "shared": SHARED_BLOCK.format(p=true_private[:3000]),
                    "inferred": INFERRED_BLOCK.format(p=est_private[:3000]),
                    "blind": "",
                }[arm]
                out = await teacher.chat(
                    [
                        {
                            "role": "user",
                            "content": ANSWER_PROMPT.format(
                                profile=profile[:2000],
                                extra=extra,
                                history=history[-10000:],
                                question=q,
                                options=opt_txt,
                            ),
                        }
                    ],
                    temperature=0.3,
                    max_tokens=900,
                )
                return split_think(out)

            results = await asyncio.gather(*(run_arm(a) for a in ARMS))
            rec = {"persona_id": s["persona_id"], "gold": gold}
            verify_jobs = []
            for arm, (think, letter) in zip(ARMS, results):
                rec[f"{arm}_correct"] = letter == gold if letter else False
                rec[f"{arm}_answered"] = letter is not None
                if think and letter:
                    verify_jobs.append((arm, think, letter))

            async def verify(arm, think, letter):
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
                    max_tokens=400,
                    json_mode=True,
                )
                try:
                    v = json.loads(vraw)
                except (json.JSONDecodeError, TypeError):
                    return arm, False
                ok = bool(v.get("all_claims_traceable")) and bool(
                    v.get("reasoning_supports_answer")
                )
                if ok:
                    async with lock:
                        kept[arm] += 1
                        files[arm].write(
                            json.dumps(
                                {
                                    "messages": [
                                        {
                                            "role": "system",
                                            "content": "You are a personalized AI assistant. Infer "
                                            "what the user has not said from what they have, then "
                                            "answer.",
                                        },
                                        {"role": "user", "content": f"{q}\n\n{opt_txt}"},
                                        {
                                            "role": "assistant",
                                            "content": f"<think>\n{think}\n</think>\nAnswer: {letter}",
                                        },
                                    ],
                                    "metadata": {
                                        "source": "qa_grounded",
                                        "arm": arm,
                                        "persona_id": s["persona_id"],
                                        "gold": gold,
                                        "answer": letter,
                                        "correct": letter == gold,
                                    },
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                else:
                    async with lock:
                        dropped[arm] += 1
                return arm, ok

            vres = await asyncio.gather(
                *(verify(arm, think, letter) for arm, think, letter in verify_jobs)
            )
            for arm, ok in vres:
                rec[f"{arm}_grounded"] = ok
            async with lock:
                recs.append(rec)

    await asyncio.gather(*(one(s, i) for i, s in enumerate(sessions)))
    for f in files.values():
        f.close()

    n = len(recs)
    if not n:
        print("no records")
        return
    print(f"\n=== per-arm label accuracy (n={n}, teacher never shown gold) ===")
    for a in ARMS:
        c = sum(1 for r in recs if r.get(f"{a}_correct"))
        print(f"  {a:<10}: {c}/{n} = {c/n:.1%}")
    print("\nMcNemar (paired):")
    for a, b_ in [("blind", "inferred"), ("inferred", "shared"), ("blind", "shared")]:
        x, y, p = mcnemar(recs, f"{a}_correct", f"{b_}_correct")
        print(f"  {a} vs {b_}: {a}-only={x} {b_}-only={y} p={p:.4f}")
    print("\n=== groundedness (fraction of reasoning a blind checker accepts) ===")
    for a in ARMS:
        k, d = kept[a], dropped[a]
        tot = k + d
        print(f"  {a:<10}: kept {k}/{tot} = {k/max(tot,1):.1%}  -> train_{a}.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
