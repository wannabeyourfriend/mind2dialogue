# Corpus utilities

Use [`mind2dialogue.py`](../mind2dialogue.py) for the paper's generation, quality-control, and QA-construction pipeline.

| Script | Purpose |
|---|---|
| `deduplicate_and_reorder_jsonl.py` | Deduplicates a supplied chat JSONL by message content and applies deterministic ordering. Run with `--help` for input/output options. |
| `plot_corpus_composition.py` | Regenerates the scenario and specialization composition plots from `corpus_composition_data.json`; both panels describe the paper's 6,330-conversation analysis subset. Requires `matplotlib` and writes `donuts_scenario.pdf` and `donuts_specialization.pdf` beside the script. |

These utilities operate on supplied inputs; they do not reconstruct the paper's historical training mixture.
