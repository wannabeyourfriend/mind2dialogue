"""Mind2Dialogue response-only LoRA trainer (Unsloth + TRL).

Single-file entry point for the Mind2Dialogue training pipeline. Loads a YAML
run config, fine-tunes a model with LoRA using
response-only loss masking, logs the number of optimized tokens per step to
Weights & Biases, and writes a reproducible ``run_meta.json`` next to the
checkpoints.

Usage:
    python train.py --config configs/<run>.yaml
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import yaml

DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_training_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def truncate_first_n_assistant(messages, n):
    if not n or n <= 0:
        return messages
    out, count = [], 0
    for m in messages:
        out.append(m)
        if m["role"] == "assistant":
            count += 1
            if count >= n:
                break
    return out


def resolve_device_map(cfg):
    if "device_map" in cfg:
        return cfg["device_map"]
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return {"": int(local_rank)}
    return "sequential"


def _patch_datasets():
    import datasets
    from datasets import Dataset
    from datasets.fingerprint import Hasher

    datasets.disable_caching()

    hash_counter = {"n": 0}

    def stable_hash(value):
        hash_counter["n"] += 1
        return f"unhashable-{hash_counter['n']:08d}"

    Hasher.hash = staticmethod(stable_hash)

    orig_map = Dataset.map

    def map_no_mp(self, *args, **kwargs):
        kwargs["num_proc"] = None
        kwargs["load_from_cache_file"] = False
        return orig_map(self, *args, **kwargs)

    Dataset.map = map_no_mp
    return Dataset


def _patch_training_args_to_dict():
    from transformers import TrainingArguments

    orig_to_dict = TrainingArguments.to_dict

    def unredacted_to_dict(self):
        d = orig_to_dict(self)
        for k in ("eos_token", "pad_token"):
            if k in d:
                d[k] = getattr(self, k)
        return d

    TrainingArguments.to_dict = unredacted_to_dict


def build_dataset(Dataset, data_path, max_turns):
    rows = read_training_rows(data_path)
    samples, turn_hist = [], Counter()
    for r in rows:
        msgs = truncate_first_n_assistant(r["messages"], max_turns)
        n_assist = sum(1 for m in msgs if m["role"] == "assistant")
        if n_assist == 0:
            continue
        turn_hist[n_assist] += 1
        samples.append({"messages": msgs})
    return Dataset.from_list(samples), turn_hist


class TokenCountingCollator:
    def __init__(self, base):
        self.base = base
        self.last_n = 0
        self.total = 0

    def __call__(self, features):
        batch = self.base(features)
        n = int((batch["labels"] != -100).sum())
        self.last_n = n
        self.total += n
        return batch

    def __getattr__(self, name):
        return getattr(self.base, name)


def make_token_log_callback():
    from transformers import TrainerCallback

    class TokenLogCallback(TrainerCallback):
        def __init__(self, collator):
            self.c = collator

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                logs["tokens/step"] = self.c.last_n
                logs["tokens/total_optimized"] = self.c.total

    return TokenLogCallback


def train(cfg):
    data_path = Path(cfg["data"])
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Training data not found: {data_path}. Supply your training file; "
            "the exact 8,244-example paper mixture is not bundled."
        )
    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

    import torch
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    Dataset = _patch_datasets()
    _patch_training_args_to_dict()

    run_name = cfg["run_name"]
    out_dir = Path(cfg["output_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WANDB_PROJECT", cfg.get("wandb_project", "sft"))
    os.environ["WANDB_NAME"] = run_name

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"],
        max_seq_length=cfg.get("max_seq_length", 16384),
        load_in_4bit=cfg.get("load_in_4bit", True),
        use_gradient_checkpointing=cfg.get("gradient_checkpointing", "unsloth"),
        device_map=resolve_device_map(cfg),
        dtype=None,
    )

    lora = cfg.get("lora", {})
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora.get("r", 64),
        lora_alpha=lora.get("alpha", 64),
        lora_dropout=lora.get("dropout", 0.0),
        target_modules=lora.get("target_modules", DEFAULT_TARGET_MODULES),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.get("seed", 42),
    )

    if cfg.get("chat_template"):
        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(tokenizer, chat_template=cfg["chat_template"])

    data_path = cfg["data"]
    dataset, turn_hist = build_dataset(Dataset, data_path, cfg.get("max_turns"))

    meta = {
        "run_name": run_name,
        "model": cfg["model"],
        "data_path": str(Path(data_path).resolve()),
        "data_sha256": file_sha256(data_path),
        "num_samples": len(dataset),
        "assistant_turn_distribution": dict(sorted(turn_hist.items())),
        "max_turns": cfg.get("max_turns"),
        "loss_mode": "assistant_only",
        "tuning_mode": "lora",
        "config": cfg,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    sft_args = SFTConfig(
        output_dir=str(out_dir),
        run_name=run_name,
        per_device_train_batch_size=cfg.get("batch_size", 2),
        gradient_accumulation_steps=cfg.get("grad_accum", 4),
        num_train_epochs=cfg.get("epochs", 1),
        learning_rate=cfg.get("lr", 2e-4),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=cfg.get("scheduler", "cosine"),
        logging_steps=cfg.get("logging_steps", 1),
        save_steps=cfg.get("save_steps", 200),
        save_total_limit=cfg.get("save_total_limit", 3),
        seed=cfg.get("seed", 42),
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim=cfg.get("optim", "adamw_8bit"),
        weight_decay=cfg.get("weight_decay", 0.0),
        max_length=cfg.get("max_seq_length", 16384),
        dataset_num_proc=1,
        eos_token=tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id),
        report_to=cfg.get("report_to", "none"),
        dataset_kwargs={"skip_prepare_dataset": False},
        ddp_find_unused_parameters=cfg.get("ddp_find_unused_parameters"),
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_args,
    )

    instr = cfg.get("instruction_part", "<|im_start|>user\n")
    resp = cfg.get("response_part", "<|im_start|>assistant\n")
    trainer = train_on_responses_only(
        trainer, instruction_part=instr, response_part=resp,
    )

    counting = TokenCountingCollator(trainer.data_collator)
    trainer.data_collator = counting
    trainer.add_callback(make_token_log_callback()(counting))

    trainer.train()

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if cfg.get("save_merged_16bit"):
        model.save_pretrained_merged(str(out_dir / "merged_16bit"), tokenizer, save_method="merged_16bit")
    meta["total_optimized_tokens"] = counting.total
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Multi-turn SFT trainer (Unsloth + TRL).")
    ap.add_argument("--config", required=True, help="Path to the YAML run config.")
    args = ap.parse_args()
    train(load_yaml(args.config))


if __name__ == "__main__":
    main()
