"""Manipulation check: did the friction / guarded conditions actually create a
latent state that varies and is under-reported at the surface?

Without this the predictive probe is meaningless: if `Internal tension` stays
"none" and `Confidence in assistant` stays "raised" (as in the shipped samples),
there is nothing latent to read and no gap to measure.

Reports per condition:
  tension_nonnull   — fraction of turns whose Internal tension is not none/empty
  conf_lowered      — fraction of turns where confidence in assistant fell
  conf_variance     — distinct confidence values seen (raised/lowered/unchanged)
  plan_leak         — content-word recall of "Next action plan" into the user's
                      actual next message (the tautology measure; lower = the
                      state is less of a script for the surface)
  state_len         — mean characters of the state report

Usage: python scripts/run_manipulation_check.py output/control output/friction output/guarded
"""

import json
import re
import sys
from pathlib import Path

WORD = re.compile(r"\w{5,}")
NULL_TENSION = re.compile(r"^\s*(none|n/?a|no\b|nothing)\b", re.I)


def sections(state: str):
    out, cur, title = {}, [], None
    for ln in state.splitlines():
        m = re.match(r"^#{2,4}\s+(.+)$", ln)
        if m:
            if title:
                out[title] = "\n".join(cur)
            title, cur = m.group(1).strip(), []
        elif title:
            cur.append(ln)
    if title:
        out[title] = "\n".join(cur)
    return out


def field(state: str, name: str) -> str:
    m = re.search(rf"^\s*(?:[-*]\s*)?\d*\.?\s*{name}\s*:\s*(.+)$", state, re.M | re.I)
    return m.group(1).strip() if m else ""


def analyze(root: Path):
    files = sorted(root.rglob("*.json"))
    n_conv = n_turn = 0
    tension_nonnull = conf_lowered = 0
    conf_vals = {}
    leaks, lens, turns_per = [], [], []
    for p in files:
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        traj = d.get("user_state_trajectory") or []
        conv = d.get("conversation") or []
        if not traj:
            continue
        n_conv += 1
        turns_per.append(d.get("num_turns", 0))
        user_msgs = [m["content"] for m in conv if m["role"] == "user"]
        for e in traj:
            st = e.get("user_state") or ""
            if not st:
                continue
            n_turn += 1
            lens.append(len(st))
            t = field(st, "Internal tension")
            if t and not NULL_TENSION.match(t):
                tension_nonnull += 1
            c = field(st, "Confidence in assistant").lower()
            key = (
                "lowered"
                if "lower" in c
                else "raised"
                if "rais" in c
                else "unchanged"
                if "unchanged" in c
                else "other/none"
            )
            conf_vals[key] = conf_vals.get(key, 0) + 1
            if key == "lowered":
                conf_lowered += 1
            # plan leak: content-word recall of the plan into the next user msg
            sec = sections(st)
            plan = next((v for k, v in sec.items() if k.startswith("Next action plan")), "")
            k = e.get("turn")
            if plan and isinstance(k, int) and k < len(user_msgs):
                pw = set(w.lower() for w in WORD.findall(plan))
                mw = set(w.lower() for w in WORD.findall(user_msgs[k]))
                if pw:
                    leaks.append(len(pw & mw) / len(pw))
    if not n_turn:
        return None
    return {
        "conversations": n_conv,
        "state_turns": n_turn,
        "mean_user_turns": round(sum(turns_per) / max(len(turns_per), 1), 1),
        "tension_nonnull": round(tension_nonnull / n_turn, 3),
        "conf_lowered": round(conf_lowered / n_turn, 3),
        "conf_distribution": conf_vals,
        "plan_leak": round(sum(leaks) / len(leaks), 3) if leaks else None,
        "mean_state_chars": round(sum(lens) / len(lens)),
    }


def main():
    roots = [Path(a) for a in sys.argv[1:]] or [Path("output/control")]
    rows = {}
    for r in roots:
        res = analyze(r)
        if res:
            rows[r.name] = res
    print(json.dumps(rows, indent=2))
    if len(rows) > 1:
        keys = ["conversations", "state_turns", "tension_nonnull", "conf_lowered", "plan_leak"]
        names = list(rows)
        print("\n| metric | " + " | ".join(names) + " |")
        print("|---|" + "---|" * len(names))
        for k in keys:
            print(f"| {k} | " + " | ".join(str(rows[n].get(k)) for n in names) + " |")


if __name__ == "__main__":
    main()
