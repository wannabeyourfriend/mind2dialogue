# Evaluation

The paper evaluates PersonaMem-v1/v2, PrefEval, ToMi, and BigToM. This folder retains the available paper-related adapters; it is **not a complete reproduction harness**.

| Command | Implemented coverage |
|---|---|
| `prefeval` | Explicit-preference generation, classification, response judging, and preference-adherence aggregation |
| `bigtom` | Forward Belief MCQ, separate true/false-belief scores, and paired TB ∧ FB accuracy |
| `personamem` | Launcher for a separately obtained official PersonaMem-v2 checkout; MCQ or generation |

PersonaMem-v1, ToMi, BigToM Forward Action, and PrefEval implicit-preference adapters are not included. The paper's exact benchmark snapshots, all run settings, and result manifests are also not bundled. Use the [paper's evaluation protocol](https://arxiv.org/html/2609.15972v1#S4.SS3) and the official benchmark repositories to establish matching conditions before comparing results.

```bash
uv pip install -e evaluations
uv run multibench list
uv run multibench run bigtom -- \
  --model your-served-model --api-base http://localhost:8000/v1 \
  --csv /path/to/bigtom.csv --condition both --output-dir results/bigtom
```

Obtain benchmark data separately; none is packaged with training data. BigToM expects the original semicolon-delimited CSV. Both conditions use the same row ordering, and `paired_tb_fb_accuracy` counts a row only when both answers are correct. Per-condition scores are diagnostic and are not the paper's paired metric. Forward Action is not produced by this adapter.

For PrefEval, place or symlink the upstream `benchmark_dataset` directory at `evaluations/data/prefeval`, including `explicit_preference/`, `mcq_options/`, and `filtered_inter_turns.json`:

```bash
uv run multibench run prefeval -- \
  --model your-served-model --api-base http://localhost:8000/v1 \
  --topic travel_restaurant --stage all --inter-turns 2 \
  --judge-model your-judge-model --judge-api-base http://localhost:8001/v1 \
  --output-dir results/prefeval
```

This is a usage example, not a claim that two intervening turns or these endpoints reproduce the paper's runs. `--stage all` runs generation, classification, judging, and aggregation. Configure a judge deliberately; otherwise the served model also judges its responses.

PersonaMem-v2's upstream implementation is not redistributed. Obtain its official checkout and dependencies separately, then launch it through:

```bash
uv run multibench run personamem -- \
  --upstream-dir /path/to/PersonaMem-v2 --python /path/to/upstream-venv/bin/python \
  --model your-served-model --api-base http://localhost:8000/v1 \
  --benchmark-file /path/to/benchmark.csv --eval-mode mcq \
  --output-dir results/personamem
```

The upstream checkout controls inference and judging behavior. Resolve its history paths, endpoint settings, configuration, and dependency versions there. The launcher does not provide a generation judge score; use upstream instructions for that stage.

## Sources and licenses

The retained adapters were selected from [p13n-eval-harness at f793763](https://github.com/wannabeyourfriend/p13n-eval-harness/tree/f793763f6089d91784e1d9673801a00b087a0ec8). The release removes unrelated adapters and unused provider/experiment utilities, fixes the CLI entry point and package assets, and adds paired Forward Belief aggregation. Prediction prompts and retained scoring logic originate in that implementation.

- [PrefEval](https://github.com/amazon-science/PrefEval): the retained prompt helpers and judge templates are adapted from Amazon Science's implementation under [CC-BY-NC-4.0](multibench/benchmarks/prefeval/LICENSE). These files are not covered by the root MIT license. This release retains the explicit zero-shot path only.
- [BigToM](https://github.com/cicl-stanford/procedural-evals-tom): [MIT](multibench/benchmarks/bigtom/LICENSE), copyright Kanishk Gandhi, Jan-Philipp Fränken, Tobias Gerstenberg, and Noah D. Goodman (2023).
- [PersonaMem-v2](https://github.com/bowen-upenn/PersonaMem-v2): official evaluator obtained separately; only the Mind2Dialogue launcher is included.
- Mind2Dialogue-owned adapter and launcher code: repository [MIT license](../LICENSE).
