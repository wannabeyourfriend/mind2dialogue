# original_rewritten_selected_prompts_us.jsonl

M2D-Sim seeds for `mind2dialogue.py generate-conversations`: one JSON object per line, with 1,240 prompts across 62 US personas (20 each).

The `original` text and source-anchored `prompt_id` retain the upstream prompt provenance; `rewritten` is the persona-grounded user utterance used to start a new synthetic dialogue. These are generation seeds, not the released dialogue or QA training examples.

## Fields

| Field | Type | Description |
|---|---|---|
| `persona_id` | string | Profile this prompt belongs to, e.g. `profile_259` |
| `prompt_id` | string | Source dataset identifier, e.g. `open-r1/OpenR1-Math-220k_102002` |
| `original` | string | Original prompt text from source dataset |
| `rewritten` | string | Persona-voice rewritten prompt grounded in the profile's background and goals |