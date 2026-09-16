# M2D-Sim behavior catalog

`behavior_modes.jsonl` contains 16 behavior records used by the simulator: 14 TUNA-derived behaviors and two fallbacks (`default_behavior` and `compound_request`). The [paper](https://arxiv.org/abs/2609.15972v1) groups the 14 displayed behaviors into six TUNA families. The catalog is loaded by `user_simulator.simulator.behavior.library` to guide the next user turn.

TUNA is the Taxonomy of User Needs and Actions by Renée Shelby, Fernando Diaz, and Vinodkumar Prabhakaran (2025), cited in the paper. These behavior controls are separate from the user's evolving mental-state document.

## Fields

| Field | Type | Description |
|---|---|---|
| `behavior_id` | string | Identifier, such as `analysis`, `clarification`, or `default_behavior` |
| `name` | string | Display name |
| `tuna_mode` | string | Parent family, or `Mixed` / `Multiple (blended)` for fallbacks |
| `tuna_strategy` | string | Strategy within the family, when applicable |
| `cognitive_delegation_level` | string | Description of how much cognitive work is delegated to the assistant |
| `description` | string | Purpose and framing of the behavior |
| `guidance_template` | string | Instructions injected into the simulator prompt |
| `few_shot_examples` | list | Optional objects with `request_type` and `user_turn` fields |

## Families

1. Information Seeking
2. Information Processing and Synthesis
3. Procedural Guidance and Execution
4. Content Creation and Transformation
5. Social Interaction
6. Meta-Conversation

The controller selects behaviors turn by turn; the catalog does not prescribe a fixed conversation sequence.
