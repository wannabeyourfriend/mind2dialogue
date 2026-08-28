"""Evaluate whether the latent-cause state is non-trivial.

Three questions, in order of how decisive they are:

A. CONTAINMENT — is the private context actually absent from the transcript?
   Lexical overlap of each private item against everything the user said. If the
   items leak, the state is a summary again and nothing else matters.

B. CAUSAL LOAD-BEARING — does the private context change behaviour? A judge that
   sees a deal_breaker checks whether the user accepted or deflected when the
   assistant proposed something matching it. If users accept deal-breakers, the
   private context is decoration.

C. PRIVILEGE VALUE — 4-arm QA, questions generated from the private context:
     blind         persona + first half of transcript
     history_only  persona + the FULL transcript (every user turn)
     privileged    persona + first half + private context
     both          everything
   `history_only` is the control that killed the previous design: there the
   state was a summary of the transcript, so full-transcript access matched it.
   Here the private context is never stated, so if privilege still buys nothing
   the design has failed for a different reason and we should know.

Usage: python scripts/evaluate_latent_corpus.py --corpus output/latent --limit 120
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

WORD = re.compile(r"[a-z]{5,}")
LETTERS = ["A", "B", "C", "D"]
PICK_RE = re.compile(r"\b([A-D])\b")
OUT = ROOT / "output/relabel_blind/latent_eval.jsonl"

GEN_PROMPT = """A user is talking to an AI assistant. Here is what the user has NOT told the assistant.

<private_facts>
{private}
</private_facts>

<what_the_user_actually_said>
{user_turns}
</what_the_user_actually_said>

<persona_profile_the_assistant_can_see>
{profile}
</persona_profile_the_assistant_can_see>

Write {n} multiple-choice questions whose correct answer follows from the private facts.

Requirements:
- The correct answer must NOT be stated in what the user actually said, and must NOT follow
  from the persona profile.
- CRITICAL: the three distractors must be exactly as plausible as the correct answer to
  someone who has read everything the user said and the whole profile. Same kind of thing,
  same length, same specificity. Someone without the private facts must be guessing.

Return strict JSON:
{{"items": [{{"question": "...", "correct": "...", "distractors": ["...","...","..."]}}]}}"""

ANSWER_PROMPT = """You are assisting a user.

<persona_profile>
{profile}
</persona_profile>
{extra}
<conversation>
{history}
</conversation>

{question}

{options}

Reply with only the letter."""

PRIV_BLOCK = "\n<what_you_privately_know_about_this_user>\n{p}\n</what_you_privately_know_about_this_user>\n"
FULL_BLOCK = "\n<complete_transcript_of_this_conversation>\n{h}\n</complete_transcript_of_this_conversation>\n"

BEHAVIOR_PROMPT = """Here is a deal-breaker for this user — something they will refuse:

<deal_breaker>{db}</deal_breaker>

Here is a conversation. The user never states this deal-breaker out loud.

<conversation>
{conv}
</conversation>

Answer strict JSON:
{{"assistant_proposed_it": true/false,
  "user_accepted_it": true/false,
  "user_deflected": true/false,
  "evidence": "the turn where this is visible, quoted briefly"}}

`assistant_proposed_it` = did the assistant suggest something matching the deal-breaker.
`user_accepted_it` = did the user agree to it. `user_deflected` = did the user change
the subject, narrow the request, or push back instead of accepting."""


def toks(s):
    return set(WORD.findall((s or "").lower()))


def containment(session):
    """Fraction of each private item's content words that appear in user turns."""
    user_txt = " ".join(m["content"] for m in session["conversation"] if m["role"] == "user")
    ut = toks(user_txt)
    out = []
    for item in session.get("private_items", []):
        it = toks(item)
        out.append(len(it & ut) / len(it) if it else 1.0)
    return out


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
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--per-session", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=14)
    ap.add_argument("--gen-model", default="gpt-4o")
    ap.add_argument("--answer-model", default="gpt-4o-mini")
    ap.add_argument("--skip-behavior", action="store_true")
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
    print(f"sessions: {len(sessions)}")
    if not sessions:
        return

    # ---------- A. containment ----------
    all_c = [c for s in sessions for c in containment(s)]
    hi = sum(1 for c in all_c if c > 0.6)
    print("\n=== A. containment (private items vs what the user actually said) ===")
    print(f"items={len(all_c)}  mean_overlap={sum(all_c)/len(all_c):.3f}  leaked(>0.6)={hi} ({hi/len(all_c):.1%})")

    gen_llm = LLM(model=args.gen_model, max_concurrent=args.concurrency)
    ans_llm = LLM(model=args.answer_model, max_concurrent=args.concurrency)
    lock = asyncio.Lock()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(OUT, "w").close()
    qa_recs, beh_recs = [], []

    # ---------- B. behavioural load-bearing ----------
    async def behavior(s):
        conv = format_conversation(s["conversation"])[-14000:]
        for db in (s.get("private_context") or {}).get("deal_breakers", [])[:2]:
            raw = await gen_llm.chat(
                [{"role": "user", "content": BEHAVIOR_PROMPT.format(db=db, conv=conv)}],
                temperature=0.0,
                max_tokens=300,
                json_mode=True,
            )
            try:
                v = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            async with lock:
                beh_recs.append(
                    {
                        "persona_id": s["persona_id"],
                        "proposed": bool(v.get("assistant_proposed_it")),
                        "accepted": bool(v.get("user_accepted_it")),
                        "deflected": bool(v.get("user_deflected")),
                    }
                )

    # ---------- C. privilege value ----------
    async def qa(s, idx):
        conv = s["conversation"]
        cut = max(2, len(conv) // 2)
        half = format_conversation(conv[:cut])
        full = format_conversation(conv)
        user_turns = "\n".join(m["content"] for m in conv if m["role"] == "user")
        profile = s.get("profile_summary", "")
        private = "\n".join(f"- {x}" for x in s["private_items"])

        raw = await gen_llm.chat(
            [
                {
                    "role": "user",
                    "content": GEN_PROMPT.format(
                        private=private[:4000],
                        user_turns=user_turns[-8000:],
                        profile=profile[:2000],
                        n=args.per_session,
                    ),
                }
            ],
            temperature=0.7,
            max_tokens=1600,
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
            # differential leak guard (the fix from the cross-session run)
            lc = len(toks(correct) & toks(full)) / max(len(toks(correct)), 1)
            ld = [len(toks(d) & toks(full)) / max(len(toks(d)), 1) for d in dis]
            if lc - sum(ld) / len(ld) > 0.15:
                continue
            opts = [correct] + dis
            rot = (idx + j) % 4
            order = list(range(4))
            order = order[rot:] + order[:rot]
            shuffled = [opts[o] for o in order]
            gold = LETTERS[order.index(0)]
            opt_txt = "\n".join(f"{LETTERS[k]}) {o}" for k, o in enumerate(shuffled))

            async def ask(arm):
                extra = {
                    "blind": "",
                    "history_only": FULL_BLOCK.format(h=full[-14000:]),
                    "privileged": PRIV_BLOCK.format(p=private[:4000]),
                    "both": FULL_BLOCK.format(h=full[-14000:]) + PRIV_BLOCK.format(p=private[:4000]),
                }[arm]
                out = await ans_llm.chat(
                    [
                        {
                            "role": "user",
                            "content": ANSWER_PROMPT.format(
                                profile=profile[:2000],
                                extra=extra,
                                history=half[-6000:],
                                question=q,
                                options=opt_txt,
                            ),
                        }
                    ],
                    temperature=0.0,
                    max_tokens=8,
                )
                m = PICK_RE.search((out or "").strip().upper())
                return m.group(1) if m else None

            picks = await asyncio.gather(
                ask("blind"), ask("history_only"), ask("privileged"), ask("both")
            )
            if any(p is None for p in picks):
                continue
            rec = {
                "persona_id": s["persona_id"],
                "blind": picks[0] == gold,
                "history_only": picks[1] == gold,
                "privileged": picks[2] == gold,
                "both": picks[3] == gold,
            }
            async with lock:
                qa_recs.append(rec)
                with open(OUT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    jobs = [qa(s, i) for i, s in enumerate(sessions)]
    if not args.skip_behavior:
        jobs += [behavior(s) for s in sessions]
    await asyncio.gather(*jobs)

    if beh_recs:
        prop = [r for r in beh_recs if r["proposed"]]
        print("\n=== B. deal-breakers (does the private context change behaviour?) ===")
        print(f"checks={len(beh_recs)}  assistant proposed a deal-breaker in {len(prop)} ({len(prop)/len(beh_recs):.1%})")
        if prop:
            acc = sum(1 for r in prop if r["accepted"])
            defl = sum(1 for r in prop if r["deflected"])
            print(f"  of those: user ACCEPTED {acc} ({acc/len(prop):.1%}) | DEFLECTED {defl} ({defl/len(prop):.1%})")
            print("  (low acceptance = the private context is causally load-bearing)")

    if qa_recs:
        print(f"\n=== C. privilege value (n={len(qa_recs)}, chance = 25%) ===")
        accs = {}
        for arm in ["blind", "history_only", "privileged", "both"]:
            accs[arm] = sum(1 for r in qa_recs if r[arm]) / len(qa_recs)
            print(f"{arm:<14}: {accs[arm]:.1%}")
        print(f"\nprivileged - history_only (THE number): {accs['privileged'] - accs['history_only']:+.1%}")
        for a, b_ in [("blind", "history_only"), ("history_only", "privileged"), ("blind", "privileged")]:
            b, c, pv = mcnemar(qa_recs, a, b_)
            print(f"  McNemar {a} vs {b_}: {a}-only={b} {b_}-only={c} p={pv:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
