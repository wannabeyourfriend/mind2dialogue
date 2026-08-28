# Mind2Dialogue: Training Human-Aware Language Models through Shared-State User Simulation

[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb.svg)](https://wannabeyourfriend.github.io/mind2dialogue/)
[![arXiv](https://img.shields.io/badge/arXiv-2026.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow.svg)](https://huggingface.co/datasets/wannabeyourfriend-hf/mind2dialogue)
[![License](https://img.shields.io/badge/License-MIT%20%C2%B7%20CC--BY--4.0-blue.svg)](LICENSE)
![idea-promotion](assets/idea-promotion.png)

A user simulator and an assistant share one structured user state. The simulator updates that state as the conversation proceeds and writes each user turn from it; the assistant answers while reading the same state, so its replies are grounded in what the simulated user actually holds. A student model is then fine-tuned on those replies with the state withheld, and has to recover state-dependent behavior from the persona and the dialogue alone.

This repository runs the data pipeline in six stages:

- generate persona-seeded scenarios in scale
- generate conversations in scale
- llm-as-judge based corpus quality auditing
- build question answering data from conversational corpus
- corpus rewrite and augmentation
- llm fine-tuning and evaluation

## Setup

Requirements:

- Python 3.10 or newer
- `uv`
- an endpoint that speaks the OpenAI protocol, such as vLLM, OpenAI itself, or a
  gateway in front of another provider

Install the package and its submodules:

```bash
git clone --recursive https://github.com/wannabeyourfriend/mind2dialogue.git
cd mind2dialogue
uv sync                   
cp .env.example .env
```

If you already cloned without `--recursive`, pull the submodules with
`git submodule update --init`.

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4o-mini"
```

Splitting the roles across endpoints is the recommended way to avoid a model judging its own output, and to put a stronger model:

```bash
export SIM_MODEL="gpt-4o-mini"      # writes the user turns
export ORACLE_MODEL="gpt-4o"        # writes the assistant turns
export JUDGE_MODEL="gpt-4.1-mini"   # scores conversations and question answers
```


Every stage is a subcommand of `mind2dialogue.py`. Stages 3 to 6 consume the output of the stage above them, so run them in order.

### 1. generate conversations

One conversation per rewritten persona prompt. `--ablation` selects which parts of the pipeline are active: `full`, `no_privilege`, `no_behavior`, `no_state`, `oracle_profile_only`, `guarded`, `guarded_oracle`, `latent`, `latent_oracle`.

```bash
python mind2dialogue.py generate-conversations --ablation full --concurrency 80
```

### 2. generate scenarios

- Construct one scenario set per persona, then generate a deeper multi-session dialogue from each.

- Scenario families are `lifelong`, `high_frequency`, `affective` and `concerning`. 

- Scenarios are cached under `data/deep_scenarios/` and reused unless you pass `--force-reconstruct`.

```bash
python mind2dialogue.py generate-scenarios --constructor lifelong --ablation full --concurrency 40
```

### 3. check quality

Score every conversation on six dimensions and sort it into tier A, B or C. Dimensions one to four are programmatic and free; five and six call a judge model, which `--skip-judges` turns off.

```bash
python mind2dialogue.py check-quality --conversations-dir output/conversations/full --output-dir output/quality_control/v1
```

### 4. build question answering data

Turn the tier A conversations into training items in four styles:
`personamem_mcq`, `prefeval_gen`, `bigtom_tom` and `lamp_cls`. Passing
`--quality-control-results` restricts the input to tier A.

```bash
python mind2dialogue.py build-question-answering-data --conversations-dir output/conversations/full --quality-control-results output/quality_control/v1/qc_results.jsonl --output-dir output/question_answering/v1
```

### 5. rewrite question answering data

Rewrite the version 1 items into harder, more persona-grounded version 2 items.

```bash
python mind2dialogue.py rewrite-question-answering-data --question-answering-dir output/question_answering/v1 --conversations-dir output/conversations/full --output-dir output/question_answering/v2
```

### 6. evaluate question answering models

Benchmark any set of models reachable through `OPENAI_BASE_URL` on the items
built above.

```bash
python mind2dialogue.py evaluate-question-answering-models --question-answering-dir output/question_answering/v2 --models gpt-4o-mini gpt-4o --output-dir output/evaluation/v2
```

## Training and Evaluation

`training/` is a submodule holding a single-file trainer: it reads a YAML run config, fine-tunes with low-rank adaptation or full fine-tuning under response-only loss masking, and writes a reproducible `run_meta.json` beside the checkpoints.

```bash
git submodule update --init training
cd training && uv sync
python train.py --config configs/example.yaml
bash scripts/serve_qwen3_4b_no_think.sh
```

`evaluations/` is a submodule holding one command line interface over six personalization and theory-of-mind benchmarks: PersonaMem, PrefEval, BigToM, LaMP, PersonaLens and Sotopia.

```bash
git submodule update --init evaluations
pip install -e evaluations
multibench run personamem --api-base <server_endpoint> --model <run_name> --workers 64 --output-dir results/<run_name>/PersonaMem
```

## Dataset

The released artifacts live at
**https://huggingface.co/datasets/wannabeyourfriend-hf/mind2dialogue**.

`samples/` holds three small files checked into the repository, so you can see the schemas without downloading anything: one conversation, one deep scenario conversation, and one training line.

## Contact Us

Feel free to find more details in the paper. This is still a work in progress; if you have any questions or comments, or notice any issues, contact us at `ziw178@ucsd.edu`. If you found this work helpful, please consider starring the repository and citing us:

```bibtex
@article{mind2dialogue2026,
  title  = {Mind2Dialogue: Training Human-Aware Language Models through Shared-State User Simulation},
  author = {Wang, Zixuan and Zhou, Yufan and Tang, Jinzhou and Wu, Chengjun and Yu, Xinle and Ye, Lyumanshan and Feng, Zhaoxiang and Peng, Letian and Patra, Adyasha and Bai, Fan and Ma, Enze and Hu, Zhengding and Gu, Jianyang and Wang, Zhao and Ding, Yufei and Shang, Jingbo and Shu, Tianmin and Hu, Zhiting and Wang, Zhen},
  year   = {2026},
}
```
