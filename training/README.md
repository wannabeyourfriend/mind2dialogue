# Training M2D-Chat

`train.py` trains an assistant with 4-bit LoRA and loss on assistant responses. Inputs are chat records with a `messages` field, stored as JSONL, JSON, or Parquet. Persona and visible dialogue belong in student inputs; the simulator's latent-state document does not.

The supplied Qwen2.5-7B-Instruct configuration preserves the historical `mix_v6` recipe: rank/alpha 64, learning rate 2e-4, two epochs, batch size 1 with eight accumulation steps, length 16,384, AdamW 8-bit, and cosine scheduling. The exact **8,244-example paper mixture is not bundled**. Supply the original file to reproduce that run; substituting the complete Hugging Face release or a similarly sized sample does not reproduce it. The reported mixture contains 3,312 dialogue and 4,932 QA examples (2,052 persona-memory MCQ, 1,442 open-ended preference QA, and 1,438 preference-classification QA).

From this directory, install dependencies on a supported CUDA machine, set `data` in the config to your file, then run:

```bash
uv sync
uv run python train.py --config configs/qwen25_7b_instruct_mix_v6_r64_lr2e-4.yaml
```

The config uses the public model ID `Qwen/Qwen2.5-7B-Instruct`; replace it with a local checkpoint path if needed. Outputs go to `outputs/<run_name>/`, including the resolved config, data SHA-256, example count, and assistant-turn distribution in `run_meta.json`. Logging to an external service is disabled by default.

The paper also evaluates Llama and OLMo. Their exact run configurations and the data-fraction manifests are not supplied here. This folder does not claim a complete reproduction of every reported run.

This trainer is adapted from [p13n-training at 36a1a14](https://github.com/wannabeyourfriend/p13n-training/tree/36a1a14ce4c42708c13c73e2612b1fdce70431d3). The release removes unrelated experiment presets, full-sequence/full-parameter tuning, adapter continuation, and GGUF export; adds Parquet input support; and replaces a machine-specific model path. Mind2Dialogue-owned training code uses the repository's [MIT license](../LICENSE).
