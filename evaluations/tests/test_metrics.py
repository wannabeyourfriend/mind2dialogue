import json
from types import SimpleNamespace

import pytest

from multibench.benchmarks.bigtom.run import paired_accuracy
from multibench.benchmarks.prefeval.run import _paths, stage_accuracy


def test_bigtom_requires_both_variants_correct():
    true_belief = [{"correct": True}, {"correct": True}, {"correct": False}]
    false_belief = [{"correct": True}, {"correct": False}, {"correct": True}]
    assert paired_accuracy(true_belief, false_belief) == pytest.approx(1 / 3)


def test_bigtom_rejects_unpaired_lengths():
    with pytest.raises(ValueError, match="same number"):
        paired_accuracy([{"correct": True}], [])


def test_prefeval_missing_judgments_cannot_count_as_success(tmp_path):
    args = SimpleNamespace(output_dir=str(tmp_path), pref_form="explicit", task="zero-shot",
                           topic="synthetic", model="test", inter_turns=0)
    _paths(args, "judged_results").write_text(json.dumps([{
        "evaluation_error_analysis": {"acknow": {"answer": "yes"}}
    }]))
    with pytest.raises(ValueError, match="Incomplete judge outputs"):
        stage_accuracy(args)
