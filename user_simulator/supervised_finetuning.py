"""Build student-view SFT records from the paper's generated conversations.

The system prompt includes the persona; messages contain observable dialogue.
The simulator's latent-state document is never included in the training input.
"""

from __future__ import annotations

import json

from user_simulator.ablation import AblationConfig

BASE_SYSTEM_INSTRUCTION = (
    "You are a personalized AI assistant. Given the conversation so far, "
    "reason about the user's state and provide a helpful response."
)


def build_training_system_prompt(
    profile_summary: str = "", behavior_metadata: str = "", include_profile: bool = True
) -> str:
    """Build the SFT system prompt, optionally injecting user profile."""
    parts = [BASE_SYSTEM_INSTRUCTION]
    if include_profile and profile_summary:
        parts.append(f"\n<user_profile>\n{profile_summary}\n</user_profile>")
    if include_profile and behavior_metadata:
        parts.append(f"\n<behavior_metadata>\n{behavior_metadata}\n</behavior_metadata>")
    return "\n".join(parts)


def build_training_instance(session: dict, config: AblationConfig) -> dict | None:
    """Assemble one SFT JSONL line from a completed rollout session.

    Returns None for empty conversations so the caller can skip the write.
    """
    conversation = session.get("conversation", [])
    if not conversation:
        return None

    behavior_metadata_raw = session.get("behavioral_metadata") or {}
    behavior_metadata_str = (
        json.dumps(behavior_metadata_raw, indent=2, ensure_ascii=False)
        if behavior_metadata_raw
        else ""
    )
    system_msg = build_training_system_prompt(
        profile_summary=session.get("profile_summary", ""),
        behavior_metadata=behavior_metadata_str,
        include_profile=config.sft_include_profile,
    )

    messages: list[dict] = [{"role": "system", "content": system_msg}]
    for msg in conversation:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    metadata: dict = {
        "persona_id": session.get("persona_id", ""),
        "scenario_id": session.get("prompt_id", ""),
    }
    if "scenario_category" in session:
        metadata["scenario_category"] = session.get("scenario_category", "")
    metadata["num_turns"] = session.get("num_turns", 0)
    metadata["termination"] = session.get("termination", "")
    metadata["ablation"] = config.name
    source = session.get("source")
    if source:
        metadata["source"] = source

    return {"messages": messages, "metadata": metadata}
