import importlib.util
from pathlib import Path

import pytest

from dvcl_bench.paths import ExperimentLayout


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_attack_effectiveness",
    ROOT / "scripts" / "analyze_attack_effectiveness.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_input_paths_use_manifest_artifacts(tmp_path: Path):
    paths = {
        "attack": tmp_path / "factor.pt",
        "clean": tmp_path / "clean.pt",
        "split": tmp_path / "split.pt",
    }
    samples = [{
        "micro_f1": 0.5,
        "inputs": {
            name: {"path": str(path), "sha256": "hash"}
            for name, path in paths.items()
        },
    }]
    actual = MODULE._input_paths(
        samples, ExperimentLayout(tmp_path), "acm", "prbcd", 5
    )
    assert actual == (paths["attack"], paths["clean"], paths["split"])


def test_input_paths_reject_mixed_artifacts(tmp_path: Path):
    samples = [
        {"micro_f1": 0.5, "inputs": {"attack": {"path": str(tmp_path / name)}}}
        for name in ("one.pt", "two.pt")
    ]
    with pytest.raises(RuntimeError, match="multiple attack artifacts"):
        MODULE._input_paths(
            samples, ExperimentLayout(tmp_path), "acm", "prbcd", 5
        )
