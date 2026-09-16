# M2D-Sim generation inputs

The checked-in inputs are restricted to personas and seeds represented in the released dialogues. Generated conversations are available on [Hugging Face](https://huggingface.co/datasets/wannabeyourfriend-hf/mind2dialogue).

| Directory | Contents |
|---|---|
| [`behavior_modes/`](behavior_modes/README.md) | 16 behavior records: 14 TUNA-derived behaviors and two fallbacks |
| [`refined_persona_profiles/`](refined_persona_profiles/README.md) | 289 profiles across US, CN, DE, IN, and JP |
| [`rewritten_prompts/`](rewritten_prompts/README.md) | 1,240 seeds across 62 US personas; every `(persona_id, prompt_id)` matches a released rewritten-query dialogue |
| `deep_scenarios/` | 911 scenario seeds in 188 cached files: 871 lifelong and 40 affective; all scenario IDs occur in the released dialogues |

The local scenario cache is a subset of the 1,757 conversation-linked scenario seeds available on Hugging Face. A scenario seed can produce several dialogues, so seed counts differ from conversation counts. The paper's analysis subset contains 6,330 deep-scenario conversations; the broader release adds 1,240 rewritten-query dialogues.

Rewritten prompts retain the original upstream prompt text and source identifier for provenance. The raw upstream staging collection is omitted because it is not required to run these seeds.

Use `uv run python mind2dialogue.py --help` from the repository root. The [main README](../README.md) shows generation and quality-control commands. The exact 8,244-example paper training mixture and its QA artifacts are not included in this cleaned release.
