"""Generate a corpus whose latent state is a CAUSE of the dialogue, not a summary.

Per conversation:
  1. Sample a private context from the persona BEFORE any dialogue exists —
     hidden constraints, a prior bad experience, hard deal-breakers, a trust
     prior and a patience budget. These are facts about the user that the
     transcript will never state wholesale.
  2. Seed the rollout's initial state with that private context plus an empty
     disclosure ledger, via rollout_conversation(initial_state=...).
  3. Roll out with user_s4_latent, which carries the private context forward
     verbatim, maintains the ledger, and moves trust/patience numerically.

The measurable claim this sets up: at turn t the `undisclosed` set is, by
construction, absent from the transcript — so privileged access to it is not a
re-encoding of the surface, which is what every previous design failed on.

Usage:
  python scripts/generate_latent_corpus.py --personas-per-region 20 --scenarios 2 \
      --max-turns 14 --concurrency 24
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from user_simulator.ablation import AblationConfig  # noqa: E402
from user_simulator.data import LLM, Persona, SIM_MODEL, load_json, save_json  # noqa: E402
from user_simulator.simulator import rollout_conversation  # noqa: E402

SCEN_DIR = ROOT / "data/deep_scenarios"
OUT_ROOT = ROOT / "output/latent"

PRIVATE_PROMPT = """Here is a person's profile.

<profile>
{profile}
</profile>

They are about to start this conversation with an AI assistant:
<opening_message>
{opening}
</opening_message>

Invent the part of their situation they would NOT volunteer to an assistant. It must be
specific, ordinary, and genuinely consequential — the kind of thing a real person keeps to
themselves out of embarrassment, privacy, or because it feels irrelevant to explain.

Requirements:
- Every item must be CHECKABLE: a concrete fact, not a mood or a personality trait.
- None of it may be inferable from the profile above. Do not restate the profile.
- The deal_breakers must be things a well-meaning assistant would plausibly SUGGEST for
  this opening message — so that a assistant who does not know them will walk into one.

Return strict JSON:
{{"hidden_constraints": ["2-3 concrete limits they will not spell out (money, time, health, family, legal, workplace)"],
  "prior_experience": ["1-2 specific things they already tried or already went through that shape what they will accept"],
  "deal_breakers": ["2-3 concrete suggestions that this person will refuse, each a thing an assistant might well propose"],
  "why_unsaid": "one sentence: why this person does not just say all this up front",
  "trust_prior": 5.5,
  "patience_budget": 4}}"""

STATE_TEMPLATE = """# User State Report

<private_context>
hidden_constraints:
{constraints}
prior_experience:
{experience}
deal_breakers:
{dealbreakers}
why_unsaid: {why}
trust_prior: {trust}
patience_budget: {patience}
</private_context>

## Disclosure ledger
- disclosed: []
- undisclosed: {all_items}
- last turn revealed: nothing

## Working state
1. trust: {trust} (was {trust}) because no exchange yet
2. patience: {patience} of {patience} remaining
3. Emotion: mild uncertainty about whether this is worth explaining fully
4. Internal tension: wants help with the opening request, but the honest version of the
   problem involves things listed in private_context that they are not ready to say
5. Short-term intent: get a useful answer without having to explain the whole situation
6. Assistant's last turn: none yet

## Cross turn memory
Nothing yet.

## Next action plan
Open with the request as written, holding back private_context.
"""


def bullets(xs):
    return "\n".join(f"  - {x}" for x in xs) if xs else "  - (none)"


def load_personas(path: Path):
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
    out = []
    for h in sorted(SCEN_DIR.glob(f"{pid}__*.json")):
        try:
            data = load_json(h)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            out.extend(data)
    return out


async def one_conversation(persona, scen, llm, config, max_turns, min_turns, out_dir, region):
    opening = scen.get("initial_prompt") or ""
    sid = scen.get("scenario_id") or f"{persona.id}_s"
    if not opening:
        return None
    profile = persona.metadata.get("refined_summary", "")
    raw = await llm.chat(
        [
            {
                "role": "user",
                "content": PRIVATE_PROMPT.format(profile=profile[:2500], opening=opening[:1200]),
            }
        ],
        temperature=0.9,
        max_tokens=800,
        json_mode=True,
    )
    try:
        pc = json.loads(raw)
        items = (
            list(pc.get("hidden_constraints", []))
            + list(pc.get("prior_experience", []))
            + list(pc.get("deal_breakers", []))
        )
        assert items
    except (json.JSONDecodeError, AssertionError, TypeError, AttributeError):
        return None

    seed_state = STATE_TEMPLATE.format(
        constraints=bullets(pc.get("hidden_constraints", [])),
        experience=bullets(pc.get("prior_experience", [])),
        dealbreakers=bullets(pc.get("deal_breakers", [])),
        why=pc.get("why_unsaid", ""),
        trust=pc.get("trust_prior", 5.0),
        patience=pc.get("patience_budget", 4),
        all_items=json.dumps(items, ensure_ascii=False),
    )

    sess = await rollout_conversation(
        persona,
        opening,
        sid,
        llm,
        max_turns=max_turns,
        min_turns=min_turns,
        config=config,
        initial_state=seed_state,
    )
    sess["persona_id"] = persona.id
    sess["region"] = region
    sess["profile_summary"] = profile
    sess["behavioral_metadata"] = persona.metadata.get("behavioral_metadata", {})
    sess["private_context"] = pc
    sess["private_items"] = items
    sess["seed_state"] = seed_state
    p = out_dir / persona.id / f"{sid}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    save_json(sess, p)
    return sess


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=["us", "cn", "jp"])
    ap.add_argument("--personas-per-region", type=int, default=20)
    ap.add_argument("--scenarios", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=14)
    ap.add_argument("--min-turns", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--ablation", default="latent")
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()

    config = AblationConfig.from_name(args.ablation)
    llm = LLM(model=SIM_MODEL, max_concurrent=args.concurrency)
    out_dir = Path(args.out)
    tasks = []
    for region in args.regions:
        pf = ROOT / f"data/refined_persona_profiles/summary_refined_profiles_{region}.jsonl"
        if not pf.exists():
            print(f"skip region {region}: no profile file")
            continue
        personas = load_personas(pf)
        picked = 0
        for pid, persona in personas.items():
            scens = scenarios_for(pid)
            if not scens:
                continue
            for sc in scens[: args.scenarios]:
                tasks.append(
                    one_conversation(
                        persona, sc, llm, config, args.max_turns, args.min_turns, out_dir, region
                    )
                )
            picked += 1
            if picked >= args.personas_per_region:
                break
    print(f"launching {len(tasks)} conversations (ablation={args.ablation})", flush=True)
    res = await asyncio.gather(*tasks, return_exceptions=True)
    ok = sum(1 for r in res if isinstance(r, dict))
    err = sum(1 for r in res if isinstance(r, Exception))
    print(f"DONE: {ok} written, {err} errors | calls={llm.calls} tokens={llm.tokens}")


if __name__ == "__main__":
    asyncio.run(main())
