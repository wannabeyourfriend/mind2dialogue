"""Non-circular probe: does the latent state carry predictive power beyond the surface?

Task (next-turn identification): given the conversation up to turn t, pick the
user's REAL next turn from 4 candidates. Gold is what the simulator actually
generated — observable, not state-defined — so the probe is not circular.

Two readers on identical items:
  blind      — conversation history only
  privileged — history + the latent user state active at turn t

If the state is epiphenomenal (fully verbalized), both readers score the same.
If it carries non-verbalized information (unexpressed tension, trust decay,
concerns the assistant missed), the privileged reader wins — and the gap is the
measured value of privilege.

Distractors are the same persona's real turns from other depths of the SAME
conversation, so they match voice, topic and register; only the state-dependent
continuation is correct.

Usage:
  python scripts/probe_state_predictive_power.py --conv-dir output/friction/conversations/deep_no_privilege \
      --tag friction [--min-depth 3] [--per-conv 6]
"""

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from user_simulator.data import LLM, format_conversation  # noqa: E402

LETTERS = ["A", "B", "C", "D"]
PICK_RE = re.compile(r"\b([A-D])\b")

# Graded state views. The full state contains "### Next action plan: what I will
# say or ask next", which would make prediction near-tautological, so the
# load-bearing conditions strip it. `affect_only` keeps just the non-verbalized
# affective signal (emotion, internal tension, trust evaluation) — if that alone
# beats blind, the latent state carries genuine unspoken predictive content.
SECTION_RE = re.compile(r"^(#{2,3}) (.+)$", re.M)


def slice_state(state: str, view: str) -> str:
    if view == "full":
        return state
    blocks, cur, keep = [], [], False
    wanted = {
        "implicit": {"Stable state", "Dynamic state", "Evaluation of Last assistant turn"},
        "affect_only": {"Evaluation of Last assistant turn"},
    }[view]
    lines = state.splitlines()
    for ln in lines:
        m = re.match(r"^#{2,3} (.+)$", ln)
        if m:
            if keep and cur:
                blocks.append("\n".join(cur))
            title = m.group(1).strip()
            keep = title in wanted
            cur = [ln] if keep else []
            continue
        if keep:
            cur.append(ln)
    if keep and cur:
        blocks.append("\n".join(cur))
    out = "\n".join(blocks)
    if view == "affect_only":
        # additionally pull only the emotion / internal-tension lines from Dynamic state
        dyn = re.search(r"### Dynamic state(.*?)(?=\n#{2,3} |\Z)", state, re.S)
        if dyn:
            picks = [
                line
                for line in dyn.group(1).splitlines()
                if re.search(r"^\s*\d*\.?\s*(Emotion|Internal tension)", line)
            ]
            if picks:
                out = "### Affective state\n" + "\n".join(picks) + "\n" + out
    return out.strip()

PROMPT = """Below is a conversation between a user and an assistant.

<conversation_so_far>
{history}
</conversation_so_far>
{state_block}
Four candidate next messages from the user are listed. Exactly one is what this user actually said next.

{options}

Which candidate is the user's real next message? Reply with only the letter."""

STATE_BLOCK = """
The user's private internal state at this moment (not visible to the assistant):
<user_state>
{state}
</user_state>
"""


def load_sessions(conv_dir: Path):
    out = []
    for p in sorted(conv_dir.rglob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if d.get("conversation") and d.get("user_state_trajectory"):
            out.append(d)
    return out


def state_for_turn(session, k):
    """State active when the user produced their (k+1)-th message.

    user_state_trajectory[turn == k] is the state the user emitted after
    assistant turn k, i.e. immediately before their next message.
    """
    for e in session["user_state_trajectory"]:
        if e.get("turn") == k:
            return e.get("user_state") or ""
    return ""


def build_items(sessions, per_conv, min_depth, rng, state_source="current"):
    items = []
    for s in sessions:
        if state_source == "carried":
            # Only sessions after the first carry anything; the carried state is
            # the closing state of the PREVIOUS session, so nothing in it can be
            # a summary of the transcript the blind reader is given.
            if not s.get("carried_state_in"):
                continue
        conv = s["conversation"]
        user_idx = [i for i, m in enumerate(conv) if m["role"] == "user"]
        # depth d = index into user_idx; the state after assistant turn d exists for d>=1
        eligible = [d for d in range(min_depth, len(user_idx)) if state_for_turn(s, d)]
        if len(eligible) < 2:
            continue
        rng.shuffle(eligible)
        for d in eligible[:per_conv]:
            i = user_idx[d]
            gold = conv[i]["content"]
            pool = [conv[j]["content"] for j in user_idx if j != i and conv[j]["content"] != gold]
            if len(pool) < 3:
                continue
            distractors = rng.sample(pool, 3)
            opts = [gold] + distractors
            order = list(range(4))
            rng.shuffle(order)
            shuffled = [opts[o] for o in order]
            gold_letter = LETTERS[order.index(0)]
            items.append(
                {
                    "persona_id": s["persona_id"],
                    "scenario_id": s.get("prompt_id"),
                    "depth": d,
                    "n_user_turns": len(user_idx),
                    "history": conv[:i],
                    "state": (
                        s["carried_state_in"]
                        if state_source == "carried"
                        else state_for_turn(s, d)
                    ),
                    "session_index": s.get("session_index"),
                    "options": shuffled,
                    "gold_letter": gold_letter,
                }
            )
    return items


async def run_condition(items, view, llm, out_path, lock):
    async def one(it):
        opts = "\n\n".join(
            f"{LETTERS[k]}) {o[:1200]}" for k, o in enumerate(it["options"])
        )
        st = "" if view == "blind" else slice_state(it["state"], view)
        prompt = PROMPT.format(
            history=format_conversation(it["history"])[-9000:],
            state_block=STATE_BLOCK.format(state=st) if st else "",
            options=opts,
        )
        out = await llm.chat(
            [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=8
        )
        m = PICK_RE.search((out or "").strip().upper())
        pick = m.group(1) if m else None
        rec = {
            "persona_id": it["persona_id"],
            "scenario_id": it["scenario_id"],
            "depth": it["depth"],
            "n_user_turns": it["n_user_turns"],
            "arm": view,
            "pick": pick,
            "gold": it["gold_letter"],
            "correct": (pick == it["gold_letter"]) if pick else None,
        }
        async with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    return await asyncio.gather(*(one(it) for it in items))


def summarize(recs, label, indent="    "):
    ok = [r for r in recs if r["correct"] is not None]
    if not ok:
        return
    bins = {"early (d<=4)": lambda d: d <= 4, "mid (5-9)": lambda d: 5 <= d <= 9, "late (d>=10)": lambda d: d >= 10}
    for name, fn in bins.items():
        sub = [r for r in ok if fn(r["depth"])]
        if sub:
            print(f"{indent}{name}: n={len(sub)} acc={sum(1 for r in sub if r['correct'])/len(sub):.1%}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--per-conv", type=int, default=6)
    ap.add_argument("--min-depth", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=15)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--state-source", choices=["current", "carried"], default="current")
    args = ap.parse_args()

    sessions = load_sessions(Path(args.conv_dir))
    print(f"sessions with state trajectories: {len(sessions)}")
    if not sessions:
        return
    rng = random.Random(42)
    items = build_items(sessions, args.per_conv, args.min_depth, rng, args.state_source)
    print(f"probe items: {len(items)}")
    if not items:
        return

    out_path = ROOT / f"output/relabel_blind/probe_predictive_{args.tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    open(out_path, "w").close()
    llm = LLM(model=args.model, max_concurrent=args.concurrency)
    lock = asyncio.Lock()

    views = ["blind", "affect_only", "implicit", "full"]
    results = {}
    for v in views:
        results[v] = await run_condition(items, v, llm, out_path, lock)

    print(f"\n=== next-turn identification ({args.tag}, state={args.state_source}, chance = 25%) ===")
    base = None
    for v in views:
        ok = [r for r in results[v] if r["correct"] is not None]
        if not ok:
            continue
        acc = sum(1 for r in ok if r["correct"]) / len(ok)
        if v == "blind":
            base = acc
        delta = f"  ({acc - base:+.1%} vs blind)" if base is not None and v != "blind" else ""
        print(f"{v:<12}: n={len(ok)} acc={acc:.1%}{delta}")
        summarize(ok, f"  {v} by depth")


if __name__ == "__main__":
    asyncio.run(main())
