"""Construct the paper's three synthetic QA formats from simulated dialogues.

PersonaMem-format multiple choice, open-ended preference QA, and
preference classification are generated from independently simulated content.
Outputs use the same messages/metadata schema as conversation SFT records.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from enum import Enum

from user_simulator.data import LLM, Persona, format_conversation
from user_simulator.prompts import load_prompt, render

logger = logging.getLogger(__name__)


class QuestionAnsweringStyle(str, Enum):
    PERSONAMEM_MULTIPLE_CHOICE = "personamem_mcq"
    PREFEVAL_GENERATION = "prefeval_gen"
    LAMP_CLASSIFICATION = "lamp_cls"


@dataclass
class QuestionAnsweringItem:
    style: QuestionAnsweringStyle
    persona_id: str
    scenario_id: str
    user_query: str
    answer_text: str
    profile_block: str = ""
    options: list[str] | None = None
    correct_letter: str | None = None
    extra: dict = field(default_factory=dict)


_PERSONAMEM_RECALL_SUFFIX = (
    " Please recall my related preferences from our conversation history "
    "to give personalized responses."
)


def _det_seed(persona_id: str, scenario_id: str, style: str) -> int:
    """Deterministic per-(persona, scenario, style) seed for reproducible shuffle."""
    h = hashlib.sha256(f"{persona_id}|{scenario_id}|{style}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _conversation_excerpt(conv: list[dict], max_turns: int = 6) -> str:
    """Take the first `max_turns` messages for prompt context (keeps prompts cheap)."""
    return format_conversation(conv[:max_turns])


def _last_user_state(session: dict) -> str:
    traj = session.get("user_state_trajectory") or []
    if not traj:
        return ""
    return traj[-1].get("user_state", "") or ""


def _profile_block(session: dict) -> str:
    return session.get("profile_summary", "") or ""


def _behavior_metadata_text(session: dict) -> str:
    bm = session.get("behavioral_metadata") or {}
    if not bm:
        return "N/A"
    return json.dumps(bm, indent=2, ensure_ascii=False)


def _persona_attributes_text(persona: Persona | None) -> str:
    if persona is None:
        return "{}"
    if persona.attributes:
        return json.dumps(persona.attributes, indent=2, ensure_ascii=False)

    return json.dumps(persona.behavioral_metadata or {}, indent=2, ensure_ascii=False)


_PERSONAMEM_TMPL = load_prompt("question_answering_personamem")
_PREFEVAL_TMPL = load_prompt("question_answering_prefeval")
_LAMP_TMPL = load_prompt("question_answering_lamp")
_MCQ_REASONING_TMPL = load_prompt("question_answering_multiple_choice_reasoning")


async def _write_multiple_choice_reasoning(
    context: str,
    question_block: str,
    correct_letter: str,
    final_answer_format: str,
    llm: LLM,
) -> str | None:
    """Generate chain-of-thought reasoning that lands on `correct_letter`.

    Returns the full assistant text including the trailing final-answer line,
    or None on failure (caller falls back to the plain final-answer string).
    """
    prompt = render(
        _MCQ_REASONING_TMPL,
        context=context,
        question_block=question_block,
        correct_letter=correct_letter,
        final_answer_format=final_answer_format,
    )
    try:
        text = await llm.chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Write the reasoning now."},
            ],
            temperature=0.5,
            max_tokens=400,
            call_type="question_answering_multiple_choice_reasoning",
        )
    except Exception as e:
        logger.warning("MCQ reasoning generation failed: %s", e)
        return None
    if not text or not text.strip():
        return None
    text = text.strip()

    if final_answer_format not in text:
        text = f"{text}\n\n{final_answer_format}"
    return text


async def _build_personamem_multiple_choice(persona: Persona | None, session: dict, llm: LLM) -> QuestionAnsweringItem | None:
    persona_id = session.get("persona_id", "")
    scenario_id = session.get("prompt_id", "")
    prompt = render(
        _PERSONAMEM_TMPL,
        profile_summary=_profile_block(session),
        behavior_metadata=_behavior_metadata_text(session),
        user_state=_last_user_state(session),
        conversation_excerpt=_conversation_excerpt(session.get("conversation", [])),
    )
    try:
        data = await llm.chat_json(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Generate the QA item now."},
            ],
            temperature=0.7,
            max_tokens=1500,
            call_type="question_answering_personamem",
        )
    except Exception as e:
        logger.warning("personamem_mcq generation failed for %s/%s: %s", persona_id, scenario_id, e)
        return None

    required = ("user_query", "correct", "stereotypical", "random", "generic")
    if not isinstance(data, dict) or any(k not in data for k in required):
        logger.warning(
            "personamem_mcq missing required keys for %s/%s: %s",
            persona_id,
            scenario_id,
            list(data.keys()) if isinstance(data, dict) else type(data),
        )
        return None

    user_query = str(data["user_query"]).strip()
    correct = str(data["correct"]).strip()
    distractors = [
        str(data["stereotypical"]).strip(),
        str(data["random"]).strip(),
        str(data["generic"]).strip(),
    ]
    if not user_query or not correct or any(not d for d in distractors):
        return None

    rng = random.Random(_det_seed(persona_id, scenario_id, QuestionAnsweringStyle.PERSONAMEM_MULTIPLE_CHOICE.value))
    options = [correct] + distractors
    rng.shuffle(options)
    correct_letter = chr(ord("A") + options.index(correct))

    final_answer = f"Final Answer: {correct_letter}"
    reasoning_text = await _write_multiple_choice_reasoning(
        context=_conversation_excerpt(session.get("conversation", [])),
        question_block=(f"Question: {user_query}\n\n" + _personamem_options_block(options)),
        correct_letter=correct_letter,
        final_answer_format=final_answer,
        llm=llm,
    )
    answer_text = reasoning_text or final_answer

    return QuestionAnsweringItem(
        style=QuestionAnsweringStyle.PERSONAMEM_MULTIPLE_CHOICE,
        persona_id=persona_id,
        scenario_id=scenario_id,
        user_query=user_query,
        answer_text=answer_text,
        profile_block=_profile_block(session),
        options=options,
        correct_letter=correct_letter,
        extra={
            "ground_truth_preference": data.get("ground_truth_preference", ""),
            "has_cot": reasoning_text is not None,
        },
    )


async def _build_prefeval_gen(persona: Persona | None, session: dict, llm: LLM) -> QuestionAnsweringItem | None:
    persona_id = session.get("persona_id", "")
    scenario_id = session.get("prompt_id", "")
    prompt = render(
        _PREFEVAL_TMPL,
        profile_summary=_profile_block(session),
        behavior_metadata=_behavior_metadata_text(session),
        user_state=_last_user_state(session),
        conversation_excerpt=_conversation_excerpt(session.get("conversation", [])),
    )
    try:
        data = await llm.chat_json(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Generate the QA item now."},
            ],
            temperature=0.7,
            max_tokens=1500,
            call_type="question_answering_prefeval",
        )
    except Exception as e:
        logger.warning("prefeval_gen generation failed for %s/%s: %s", persona_id, scenario_id, e)
        return None

    required = ("preference", "question", "assistant_response")
    if not isinstance(data, dict) or any(k not in data for k in required):
        return None

    preference = str(data["preference"]).strip()
    question = str(data["question"]).strip()
    response = str(data["assistant_response"]).strip()
    ack = str(data.get("acknowledge_quote", "")).strip()
    if not preference or not question or not response:
        return None

    if not _prefeval_acknowledges(preference, response, ack):
        logger.info("prefeval_gen dropped (no acknowledgement) for %s/%s", persona_id, scenario_id)
        return None

    return QuestionAnsweringItem(
        style=QuestionAnsweringStyle.PREFEVAL_GENERATION,
        persona_id=persona_id,
        scenario_id=scenario_id,
        user_query=question,
        answer_text=response,
        profile_block="",
        extra={"preference": preference, "acknowledge_quote": ack},
    )


def _prefeval_acknowledges(preference: str, response: str, ack_quote: str) -> bool:
    """Return True if `response` plausibly acknowledges `preference`."""
    if not preference or not response:
        return False
    resp_lower = response.lower()
    if ack_quote and ack_quote.lower() in resp_lower:
        return True

    pref_tokens = {t.lower().strip(".,!?;:\"'") for t in preference.split() if len(t) >= 4}
    hits = sum(1 for t in pref_tokens if t and t in resp_lower)
    return hits >= 2


async def _build_lamp_cls(persona: Persona | None, session: dict, llm: LLM) -> QuestionAnsweringItem | None:
    persona_id = session.get("persona_id", "")
    scenario_id = session.get("prompt_id", "")
    prompt = render(
        _LAMP_TMPL,
        profile_summary=_profile_block(session),
        persona_attributes_json=_persona_attributes_text(persona),
        conversation_excerpt=_conversation_excerpt(session.get("conversation", [])),
    )
    try:
        data = await llm.chat_json(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Generate the QA item now."},
            ],
            temperature=0.7,
            max_tokens=1200,
            call_type="question_answering_lamp",
        )
    except Exception as e:
        logger.warning("lamp_cls generation failed for %s/%s: %s", persona_id, scenario_id, e)
        return None

    required = ("task_family", "profile_items", "query", "target")
    if not isinstance(data, dict) or any(k not in data for k in required):
        return None
    items = data.get("profile_items") or []
    if not isinstance(items, list) or len(items) < 2:
        return None
    query = str(data["query"]).strip()
    target = str(data["target"]).strip()
    if not query or not target:
        return None

    item_lines = []
    for it in items:
        if not isinstance(it, dict):
            continue
        ip = str(it.get("input", "")).strip()
        op = str(it.get("output", "")).strip()
        if ip and op:
            item_lines.append(f"- input: {ip} | output: {op}")
    if not item_lines:
        return None

    user_text = (
        "Here are some past items belonging to this user:\n"
        + "\n".join(item_lines)
        + f"\n\nNew input: {query}\nProduce the output."
    )

    return QuestionAnsweringItem(
        style=QuestionAnsweringStyle.LAMP_CLASSIFICATION,
        persona_id=persona_id,
        scenario_id=scenario_id,
        user_query=user_text,
        answer_text=target,
        profile_block="",
        extra={"task_family": str(data["task_family"])},
    )


_BUILDERS = {
    QuestionAnsweringStyle.PERSONAMEM_MULTIPLE_CHOICE: _build_personamem_multiple_choice,
    QuestionAnsweringStyle.PREFEVAL_GENERATION: _build_prefeval_gen,
    QuestionAnsweringStyle.LAMP_CLASSIFICATION: _build_lamp_cls,
}


async def generate_for_conversation(
    persona: Persona | None, session: dict, style: QuestionAnsweringStyle, llm: LLM
) -> QuestionAnsweringItem | None:
    """Build one QA item of the given style from one conversation session."""
    builder = _BUILDERS[style]
    return await builder(persona, session, llm)


_BASE_SFT_INSTRUCTION = (
    "You are a personalized AI assistant. Use the conversation context to "
    "give a response that reflects the user's preferences."
)


def _personamem_options_block(options: list[str]) -> str:
    """Mirror p13n-eval-harness/personamem/inference.py:create_mcq_options output."""
    parts = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
    return (
        "Please choose the best answer from the following options:\n\n"
        + "\n".join(parts)
        + "\n\nThink step by step about which answer best fits the user's "
        "query and conversation context. Provide your reasoning first, "
        "then give your final answer as 'Final Answer: [Letter]'"
    )


def question_answering_item_to_training_line(item: QuestionAnsweringItem, source_session: dict) -> dict:
    """Render a QuestionAnsweringItem to one SFT JSONL line.

    Output schema is identical to `user_simulator.supervised_finetuning.build_training_instance`:
        {"messages": [...], "metadata": {...}}
    so the existing trainer can ingest QA and conversational SFT lines
    interchangeably. Two extra metadata fields: `qa_style` and `source: "qa"`.

    Each style has its own student-visible prompt format; the latent-state
    document used during generation is withheld.
    """
    metadata = {
        "persona_id": item.persona_id,
        "scenario_id": item.scenario_id,
        "qa_style": item.style.value,
        "source": "qa",
        "ablation": source_session.get("ablation", ""),
    }

    if item.style is QuestionAnsweringStyle.PERSONAMEM_MULTIPLE_CHOICE:
        history = source_session.get("conversation", []) or []

        history = history[-8:]
        user_msg = (
            item.user_query.rstrip()
            + _PERSONAMEM_RECALL_SUFFIX
            + "\n\n"
            + _personamem_options_block(item.options or [])
        )
        system_msg = f"{_BASE_SFT_INSTRUCTION}\n\n<persona>\n{item.profile_block}\n</persona>"
        messages = [{"role": "system", "content": system_msg}]
        for m in history:
            r = m.get("role")
            if r in ("user", "assistant") and (m.get("content") or "").strip():
                messages.append({"role": r, "content": m["content"]})
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": item.answer_text})
        metadata["correct_letter"] = item.correct_letter
        return {"messages": messages, "metadata": metadata}

    if item.style is QuestionAnsweringStyle.PREFEVAL_GENERATION:
        preference = item.extra.get("preference", "")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": preference},
            {
                "role": "assistant",
                "content": "Got it — I'll keep that in mind for our conversation.",
            },
            {"role": "user", "content": item.user_query},
            {"role": "assistant", "content": item.answer_text},
        ]
        metadata["preference"] = preference
        return {"messages": messages, "metadata": metadata}

    if item.style is QuestionAnsweringStyle.LAMP_CLASSIFICATION:
        system_msg = "You are a helpful assistant. Use the user's past items as context to produce the requested output."
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": item.user_query},
            {"role": "assistant", "content": item.answer_text},
        ]
        metadata["task_family"] = item.extra.get("task_family", "")
        return {"messages": messages, "metadata": metadata}

    raise ValueError(f"unknown QA style: {item.style}")


__all__ = [
    "QuestionAnsweringStyle",
    "QuestionAnsweringItem",
    "generate_for_conversation",
    "question_answering_item_to_training_line",
]
