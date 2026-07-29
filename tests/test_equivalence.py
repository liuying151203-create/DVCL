import json
import subprocess
import sys
from pathlib import Path

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


def test_compare_cli_creates_output_parent(tmp_path):
    reference = tmp_path / "reference.json"
    current = tmp_path / "current.json"
    output = tmp_path / "reports" / "metrics.json"
    reference.write_text(json.dumps({"accuracy": 0.8, "micro_f1": 0.8, "macro_f1": 0.7}))
    current.write_text(json.dumps({"accuracy": 0.8, "micro_f1": 0.8, "macro_f1": 0.7}))
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "compare_equivalence.py"),
            "--kind",
            "metrics",
            "--reference",
            str(reference),
            "--current",
            str(current),
            "--output",
            str(output),
        ],
        check=True,
    )
    assert json.loads(output.read_text())["ok"]
