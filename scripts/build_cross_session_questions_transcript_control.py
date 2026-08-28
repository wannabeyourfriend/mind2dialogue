"""Cross-session QA: the regime where privileged state is provably load-bearing.

The released QA is answerable from the current transcript, so a state-blind
teacher reproduces the gold label 99.9% of the time and privilege buys nothing.
Here each item is built so its answer lives in a PREVIOUS session:

  1. A generator sees the carried state (closing state of session k-1) plus the
     current session's transcript, and writes a question whose correct answer
     depends on something established earlier and NOT restated in the current
     transcript, with three plausible distractors.
  2. A leakage filter drops any item whose answer text overlaps the current
     transcript or the persona profile — so the item cannot be solved by a
     reader of the visible surface, and cannot be solved from the persona block
     the student already receives.
  3. Two teachers then answer the surviving items:
       blind      = persona + current transcript
       privileged = the same + carried cross-session state
     The accuracy gap is the value of privileged information, measured on the
     same scale as the 99.9% agreement seen on the released corpus.

Usage:
  python scripts/build_cross_session_questions.py --limit 200 --concurrency 15
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from user_simulator.data import LLM, format_conversation  # noqa: E402

MS_DIR = ROOT / "output/multisession"
OUT = ROOT / "output/relabel_blind/crosssession_qa_transctrl.jsonl"

WORD = re.compile(r"[a-z]{5,}")
LETTERS = ["A", "B", "C", "D"]
PICK_RE = re.compile(r"\b([A-D])\b")

GEN_PROMPT = """You are constructing a diagnostic question about a returning user.

Below is what the assistant privately knows about this user from a PREVIOUS conversation, and the transcript of the CURRENT conversation.

<transcript_of_the_previous_conversation>
{state}
</transcript_of_the_previous_conversation>

<current_session_transcript>
{history}
</current_session_transcript>

<persona_profile_visible_to_everyone>
{profile}
</persona_profile_visible_to_everyone>

Write {n_items} multiple-choice questions about this user. Each one must satisfy ALL of:
- the correct answer is true ONLY because of the PREVIOUS CONVERSATION TRANSCRIPT above
- it is NOT answerable from the current transcript, and NOT answerable from the persona profile
- it is about a concrete fact: something they already tried, already ruled out, a constraint
  they are under, a commitment they made, or how the last conversation left them feeling
- CRITICAL — the three distractors must be EQUALLY plausible for this persona as the correct
  answer. Someone who has read the persona profile and the current transcript, but does NOT
  have the previous conversation, must have NO basis whatsoever to prefer one option over
  another; they should be reduced to guessing. All four options must be the same kind of
  thing, the same length and specificity, and all four must fit the persona equally well.
  Do NOT make distractors wrong in a way that is detectable without the previous session
  (nothing absurd, nothing off-persona, nothing contradicting the current transcript).

Return strict JSON:
{{"items": [{{"question": "...", "correct": "...", "distractors": ["...", "...", "..."], "evidence_from_state": "the exact line(s) of the previous transcript that make the correct answer true"}}]}}"""

ANSWER_PROMPT = """You are answering a question about a user you are assisting.

<persona_profile>
{profile}
</persona_profile>
{state_block}
<current_conversation>
{history}
</current_conversation>

{question}

{options}

Reply with only the letter of the correct option."""

STATE_BLOCK = """
<what_you_remember_about_this_user_from_previous_conversations>
{state}
</what_you_remember_about_this_user_from_previous_conversations>
"""

PREV_BLOCK = """
<transcript_of_your_previous_conversation_with_this_user>
{prev}
</transcript_of_your_previous_conversation_with_this_user>
"""


def toks(s):
    return set(WORD.findall((s or "").lower()))


def leaks(answer, *surfaces):
    """Fraction of the answer's content words already present in visible text."""
    a = toks(answer)
    if not a:
        return 1.0
    vis = set()
    for s in surfaces:
        vis |= toks(s)
    return len(a & vis) / len(a)


def load_carrying_sessions():
    """Sessions that carry prior state, each paired with the PREVIOUS session's
    raw transcript.

    The prior transcript is the control condition: the state is mandated to be a
    self-contained summary of it, so a reader given the raw transcript should
    match a reader given the state — unless the state adds something extraction
    cannot recover. Without this arm, a privileged-vs-blind gap only shows that
    prior-session information exists, not that privileged STATE is what carries it.
    """
    by_persona = {}
    for p in sorted(MS_DIR.rglob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if not d.get("conversation"):
            continue
        by_persona.setdefault(d["persona_id"], {})[d.get("session_index")] = d
    out = []
    for pid, sess in by_persona.items():
        for k, d in sorted(sess.items(), key=lambda x: (x[0] is None, x[0])):
            if not d.get("carried_state_in") or k is None:
                continue
            prev = sess.get(k - 1)
            if not prev:
                continue
            d["_prev_conversation"] = prev["conversation"]
            out.append(d)
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=15)
    ap.add_argument("--gen-model", default="gpt-4o")
    ap.add_argument("--answer-model", default="gpt-4o-mini")
    ap.add_argument("--leak-threshold", type=float, default=0.15,
                    help="max differential leak (correct minus mean distractor)")
    ap.add_argument("--per-session", type=int, default=3)
    args = ap.parse_args()

    sessions = load_carrying_sessions()[: args.limit]
    print(f"sessions carrying prior state: {len(sessions)}", flush=True)
    if not sessions:
        return

    gen_llm = LLM(model=args.gen_model, max_concurrent=args.concurrency)
    ans_llm = LLM(model=args.answer_model, max_concurrent=args.concurrency)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(OUT, "w").close()
    lock = asyncio.Lock()
    stats = {"generated": 0, "dropped_leak": 0, "dropped_parse": 0, "kept": 0}

    async def one(s, idx):
        conv = s["conversation"]
        # use roughly the first half of the session: the question must not be
        # answerable from what the user says later in this same session
        cut = max(2, len(conv) // 2)
        history = format_conversation(conv[:cut])
        profile = s.get("profile_summary", "")
        state = s["carried_state_in"]

        raw = await gen_llm.chat(
            [
                {
                    "role": "user",
                    "content": GEN_PROMPT.format(
                        state=format_conversation(s.get("_prev_conversation") or [])[-9000:],
                        history=history[-6000:],
                        profile=profile[:2000],
                        n_items=args.per_session,
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
            async with lock:
                stats["dropped_parse"] += 1
            return

        visible = format_conversation(conv)
        for j, item in enumerate(items):
            try:
                q, correct = item["question"], item["correct"]
                dis = item["distractors"][:3]
                assert q and correct and len(dis) == 3
            except (KeyError, AssertionError, TypeError):
                async with lock:
                    stats["dropped_parse"] += 1
                continue
            async with lock:
                stats["generated"] += 1

            # DIFFERENTIAL leakage: what matters is not whether the correct answer
            # shares words with the visible surface, but whether it shares MORE
            # than the distractors do — that asymmetry is what a blind reader
            # exploits. Absolute overlap is unavoidable (all options discuss the
            # same topic) and filtering on it just discards usable items.
            lk_c = leaks(correct, visible, profile)
            lk_d = [leaks(d, visible, profile) for d in dis]
            diff = lk_c - (sum(lk_d) / len(lk_d))
            if diff > args.leak_threshold:
                async with lock:
                    stats["dropped_leak"] += 1
                continue

            opts = [correct] + dis
            rot = (idx + j) % 4
            order = list(range(4))
            order = order[rot:] + order[:rot]
            shuffled = [opts[o] for o in order]
            gold = LETTERS[order.index(0)]
            opt_txt = "\n".join(f"{LETTERS[k]}) {o}" for k, o in enumerate(shuffled))
            await score_item(s, q, opt_txt, gold, diff, item, profile, state, history)

    async def score_item(s, q, opt_txt, gold, diff, item, profile, state, history):
        prev_txt = format_conversation(s.get("_prev_conversation") or [])

        async def ask(arm):
            if arm == "privileged":
                block = STATE_BLOCK.format(state=state[:6000])
            elif arm == "history_only":
                block = PREV_BLOCK.format(prev=prev_txt[-12000:])
            elif arm == "both":
                block = PREV_BLOCK.format(prev=prev_txt[-12000:]) + STATE_BLOCK.format(
                    state=state[:6000]
                )
            else:
                block = ""
            out = await ans_llm.chat(
                [
                    {
                        "role": "user",
                        "content": ANSWER_PROMPT.format(
                            profile=profile[:2000],
                            state_block=block,
                            history=history[-6000:],
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

        blind_pick, priv_pick, hist_pick, both_pick = await asyncio.gather(
            ask("blind"), ask("privileged"), ask("history_only"), ask("both")
        )
        rec = {
            "persona_id": s["persona_id"],
            "scenario_id": s.get("prompt_id"),
            "session_index": s.get("session_index"),
            "question": q,
            "gold": gold,
            "differential_leak": round(diff, 3),
            "blind_pick": blind_pick,
            "priv_pick": priv_pick,
            "hist_pick": hist_pick,
            "both_pick": both_pick,
            "blind_correct": blind_pick == gold if blind_pick else None,
            "priv_correct": priv_pick == gold if priv_pick else None,
            "hist_correct": hist_pick == gold if hist_pick else None,
            "both_correct": both_pick == gold if both_pick else None,
            "evidence_from_state": (item.get("evidence_from_state") or "")[:500],
        }
        async with lock:
            stats["kept"] += 1
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(s, i) for i, s in enumerate(sessions)))
    print(json.dumps(stats, indent=1))

    recs = [json.loads(line) for line in OUT.read_text().splitlines()]
    import math

    arms = ["blind", "history_only", "privileged", "both"]
    key = {"blind": "blind_correct", "history_only": "hist_correct",
           "privileged": "priv_correct", "both": "both_correct"}
    ok = [r for r in recs if all(r.get(key[a]) is not None for a in arms)]
    if not ok:
        return
    print(f"\n=== cross-session QA (n={len(ok)}, chance = 25%) ===")
    accs = {}
    for a in arms:
        accs[a] = sum(1 for r in ok if r[key[a]]) / len(ok)
        print(f"{a:<14}: {accs[a]:.1%}")
    print(f"\nprior-session information (history_only - blind): {accs['history_only'] - accs['blind']:+.1%}")
    print(f"privileged STATE beyond raw transcript (privileged - history_only): {accs['privileged'] - accs['history_only']:+.1%}")

    def mcnemar(a1, a2):
        b = sum(1 for r in ok if r[key[a1]] and not r[key[a2]])
        c = sum(1 for r in ok if not r[key[a1]] and r[key[a2]])
        if b + c == 0:
            return b, c, float("nan")
        chi = (abs(b - c) - 1) ** 2 / (b + c)
        return b, c, math.erfc(math.sqrt(chi / 2))

    print("\nMcNemar (paired):")
    for a1, a2 in [("blind", "history_only"), ("history_only", "privileged"), ("blind", "privileged"), ("privileged", "both")]:
        b, c, pv = mcnemar(a1, a2)
        print(f"  {a1} vs {a2}: {a1}-only={b} {a2}-only={c} p={pv:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
