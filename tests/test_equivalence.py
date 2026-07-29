import json

import pytest

pytest.importorskip("torch")

from dvcl_bench.equivalence import compare_metrics


def test_metric_equivalence_uses_absolute_tolerance(tmp_path):
    reference = tmp_path / "reference.json"
    current = tmp_path / "current.json"
    reference.write_text(json.dumps({"accuracy": 0.8, "micro_f1": 0.8, "macro_f1": 0.7}))
    current.write_text(json.dumps({"metrics": {"accuracy": 0.801, "micro_f1": 0.8, "macro_f1": 0.7}}))
    assert compare_metrics(reference, current, 0.005)["ok"]
    assert not compare_metrics(reference, current, 0.0001)["ok"]
