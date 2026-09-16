import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


spec = importlib.util.spec_from_file_location("m2d_train", Path(__file__).parents[1] / "train.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def test_training_formats_preserve_messages(tmp_path):
    rows = [{"messages": [{"role": "user", "content": "A synthetic request"},
                          {"role": "assistant", "content": "A synthetic answer"}]}]
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text(json.dumps(rows[0]) + "\n")
    parquet = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    assert trainer.read_training_rows(jsonl) == rows
    assert trainer.read_training_rows(parquet) == rows


def test_missing_paper_mixture_fails_before_loading_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="not bundled"):
        trainer.train({"data": str(tmp_path / "missing.parquet")})
