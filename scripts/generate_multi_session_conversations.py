"""Generate multi-session conversations: the one non-tautological source of privilege.

The state schema declares it "persists and evolves across sessions", but every
released rollout is a single session, so the state never carries anything the
current transcript lacks — which is why a state-blind reader loses nothing.

Here each persona runs a chain of sessions on DIFFERENT scenarios. Session k
starts with the closing state of session k-1, so the user enters already holding
constraints, preferences and history that session k's transcript never restates.
A reader of session k alone cannot recover them; a reader of the state can.

Session 1 is discarded from evaluation (it has no carried state); sessions 2+
are the probe material.

Usage:
  python scripts/generate_multi_session_conversations.py --personas profile_259 profile_360 \
      --sessions 3 --max-turns 12 --concurrency 20
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from user_simulator.ablation import AblationConfig  # noqa: E402
from user_simulator.data import LLM, SIM_MODEL, load_json, save_json  # noqa: E402
from user_simulator.simulator import rollout_conversation  # noqa: E402

SCEN_DIR = ROOT / "data/deep_scenarios"
OUT_ROOT = ROOT / "output/multisession"


def load_personas(path: Path):
    from user_simulator.data import Persona

    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        pid = d.get("persona_id")
        out[pid] = Persona(
            id=pid,
            summary=d.get("summary", ""),
            metadata={
                "refined_summary": d.get("refined_summary") or d.get("summary", ""),
                "behavioral_metadata": d.get("behavioral_metadata") or {},
            },
        )
    return out


def scenarios_for(pid: str):
    """Reuse cached deep scenarios so each session has a distinct topic."""
    hits = sorted(SCEN_DIR.glob(f"{pid}__*.json"))
    out = []
    for h in hits:
        try:
            data = load_json(h)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            out.extend(data)
    return out


async def run_chain(persona, scens, llm, config, n_sessions, max_turns, min_turns, out_dir):
    carried = None
    sessions = []
    for k in range(n_sessions):
        if k >= len(scens):
            break
        sc = scens[k]
        prompt = sc.get("initial_prompt") or ""
        sid = sc.get("scenario_id") or f"{persona.id}_s{k}"
        if not prompt:
            continue
        sess = await rollout_conversation(
            persona,
            prompt,
            sid,
            llm,
            max_turns=max_turns,
            min_turns=min_turns,
            config=config,
            initial_state=carried,
        )
        sess["session_index"] = k
        sess["carried_state_in"] = carried or ""
        sess["persona_id"] = persona.id
        sess["profile_summary"] = persona.metadata.get("refined_summary", "")
        sess["behavioral_metadata"] = persona.metadata.get("behavioral_metadata", {})
        traj = sess.get("user_state_trajectory") or []
        carried = traj[-1]["user_state"] if traj else carried
        p = out_dir / persona.id / f"session_{k}_{sid}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        save_json(sess, p)
        sessions.append(sess)
    return sessions


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", nargs="+", required=True)
    ap.add_argument("--profiles", default="data/refined_persona_profiles/summary_refined_profiles_us.jsonl")
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--min-turns", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--ablation", default="guarded")
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()

    personas = load_personas(ROOT / args.profiles)
    config = AblationConfig.from_name(args.ablation)
    llm = LLM(model=SIM_MODEL, max_concurrent=args.concurrency)
    out_dir = Path(args.out)

    chains = []
    for pid in args.personas:
        p = personas.get(pid)
        if not p:
            print(f"skip {pid}: not in profiles")
            continue
        scens = scenarios_for(pid)
        if len(scens) < 2:
            print(f"skip {pid}: only {len(scens)} cached scenarios (need >=2)")
            continue
        chains.append(run_chain(p, scens, llm, config, args.sessions, args.max_turns, args.min_turns, out_dir))

    print(f"running {len(chains)} persona chains x {args.sessions} sessions (ablation={args.ablation})")
    results = await asyncio.gather(*chains)
    n = sum(len(r) for r in results)
    print(f"DONE: {n} sessions written to {out_dir} | calls={llm.calls} tokens={llm.tokens}")


if __name__ == "__main__":
    asyncio.run(main())
