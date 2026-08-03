import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from dvcl_bench.equivalence import compare_legacy_training_log, compare_metrics


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


def test_compare_legacy_training_log(tmp_path):
    reference = tmp_path / "legacy.log"
    reference.write_text(
        "0 |VAL Micro-F1: 0.5 , Macro-F1: 0.4\n"
        "1 |VAL Micro-F1: 0.6 , Macro-F1: 0.5\n"
        "@@@@test: 0.7 0.7 0.6\n",
        encoding="utf-8",
    )
    history = tmp_path / "history.csv"
    history.write_text(
        "epoch,val_micro_f1,val_macro_f1\n0,0.5,0.4\n1,0.6,0.5\n",
        encoding="utf-8",
    )
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps({"metrics": {"accuracy": 0.7, "micro_f1": 0.7, "macro_f1": 0.6}}),
        encoding="utf-8",
    )
    report = compare_legacy_training_log(reference, history, metrics, tolerance=0)
    assert report["ok"]
    assert report["epochs_compared"] == 2
