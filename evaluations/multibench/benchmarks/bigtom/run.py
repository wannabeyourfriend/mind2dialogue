"""BigToM Forward Belief MCQ evaluation.

Upstream: https://github.com/cicl-stanford/procedural-evals-tom

Evaluation semantics are preserved: 200 stories, each story rendered with
either `true_belief` or `false_belief` percept, asked as a two-way MCQ with
randomised (seed=0) option order, scored by 'a)' / 'b)' substring match
with a content-match fallback.

Parallelized across items; both-condition runs also report paired TB & FB accuracy.
Forward Action is not implemented in this retained adapter.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from ...core import add_common_llm_args, atomic_write_json, client_from_args, ensure_dir

PROMPT_PATH = Path(__file__).with_name("prompt_evaluate.txt")


def _load_instruction() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _find_csv(user_path: str | None) -> Path:
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    # default: repo data symlink
    from ...core import benchmark_data_dir
    root = benchmark_data_dir("bigtom")
    # upstream structure: data/bigtom/bigtom.csv
    cand = root / "bigtom" / "bigtom.csv"
    if cand.exists():
        return cand
    cand = root / "bigtom.csv"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"bigtom.csv not found under {root}")


def _build_story(row: list[str], condition: str) -> tuple[str, str, str, str]:
    story_base = row[0]
    if condition == "true_belief":
        percept, question = row[1], row[5]
        true_answer, false_answer = row[8], row[11]
    elif condition == "false_belief":
        percept, question = row[2], row[5]
        true_answer, false_answer = row[11], row[8]
    else:
        raise ValueError(f"Unsupported condition: {condition}")
    return f"{story_base} {percept}", question, true_answer, false_answer


def _prepare_items(rows: list[list[str]], condition: str, instruction: str, seed: int):
    rng = random.Random(seed)
    items = []
    for row in rows:
        story, question, true_ans, false_ans = _build_story(row, condition)
        answers = [true_ans, false_ans]
        rng.shuffle(answers)
        answer_key = "a)" if answers[0] == true_ans else "b)"
        mcq_q = f"{question}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}"
        prompt = f"{instruction}\n\nStory: {story}\nQuestion: {mcq_q}\nAnswer:"
        items.append({
            "prompt": prompt,
            "story": story,
            "question": question,
            "true_answer": true_ans,
            "answer_key": answer_key,
        })
    return items


def _score(predicted: str, true_answer: str, answer_key: str) -> bool:
    p = predicted.lower()
    other = "b)" if answer_key == "a)" else "a)"
    if answer_key in p:
        return True
    if other in p:
        return False
    return true_answer.lower()[:20] in p


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="BigToM Forward Belief MCQ (parallel).")
    add_common_llm_args(ap)
    ap.add_argument("--csv", default=None, help="Path to bigtom.csv (default: data/bigtom/bigtom/bigtom.csv).")
    ap.add_argument("--condition", default="both",
                    choices=["true_belief", "false_belief", "both"])
    return ap


def _run_condition(client, rows, condition, instruction, args, out_dir):
    items = _prepare_items(rows, condition, instruction, args.seed)
    predictions_text = client.chat_batch(
        items,
        build_messages=lambda it: it["prompt"],
        workers=args.workers,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        desc=f"BigTom {condition}",
    )

    predictions = []
    correct = 0
    for it, pred in zip(items, predictions_text):
        if isinstance(pred, Exception):
            is_correct = False
            pred_str = f"<error: {pred}>"
        else:
            pred_str = pred
            is_correct = _score(pred, it["true_answer"], it["answer_key"])
        correct += int(is_correct)
        predictions.append({
            "story": it["story"][:100],
            "question": it["question"],
            "true_answer": it["true_answer"],
            "predicted": pred_str,
            "correct": is_correct,
        })

    total = len(predictions)
    accuracy = correct / max(total, 1)
    model_safe = args.model.replace("/", "_")
    atomic_write_json(out_dir / f"results_{model_safe}_{condition}.json", {
        "model": args.model,
        "condition": condition,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    })
    atomic_write_json(out_dir / f"predictions_{model_safe}_{condition}.json", predictions)
    print(f"BigTom {condition}: accuracy={accuracy:.2%} ({correct}/{total})")
    return predictions


def paired_accuracy(true_belief, false_belief):
    """Require both answers to be correct for each CSV row."""
    if len(true_belief) != len(false_belief):
        raise ValueError("Paired conditions must contain the same number of rows")
    correct = sum(t["correct"] and f["correct"] for t, f in zip(true_belief, false_belief))
    return correct / max(len(true_belief), 1)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out_dir = ensure_dir(args.output_dir)
    instruction = _load_instruction()
    csv_path = _find_csv(args.csv)
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter=";"))
    if args.max_items:
        rows = rows[: args.max_items]

    client = client_from_args(args)
    conditions = ["true_belief", "false_belief"] if args.condition == "both" else [args.condition]
    predictions = {
        c: _run_condition(client, rows, c, instruction, args, out_dir)
        for c in conditions
    }
    summary = {
        "model": args.model,
        "task": "forward_belief",
        "accuracy": {
            c: sum(p["correct"] for p in items) / max(len(items), 1)
            for c, items in predictions.items()
        },
    }
    if args.condition == "both":
        summary["paired_tb_fb_accuracy"] = paired_accuracy(
            predictions["true_belief"], predictions["false_belief"]
        )
        print(f"BigToM Forward Belief TB & FB: {summary['paired_tb_fb_accuracy']:.2%}")
    atomic_write_json(out_dir / f"summary_{args.model.replace('/', '_')}.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
