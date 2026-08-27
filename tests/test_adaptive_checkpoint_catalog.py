import json
from pathlib import Path

import pytest

from scripts.prepare_adaptive_checkpoint_catalog import (
    materialize_checkpoint,
    source_protocol,
    validate_source,
)


def test_source_protocol_covers_all_model_families():
    assert source_protocol("acm", "han") == "acm_poisoning_main_v1"
    assert source_protocol("dblp", "heteroguard") == "robust_baselines_poisoning_v1"
    assert source_protocol("acm", "hgt") == "openhgnn_baselines_poisoning_v1"
    assert source_protocol("aminer", "hgt") == "aminer_poisoning_main_v1"


def test_validate_and_materialize_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "checkpoint.pt").write_bytes(b"checkpoint")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"state": "completed"}), encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(json.dumps({
        "experiment": {
            "dataset": "acm",
            "model": {"name": "han"},
            "seeds": {"train": 1},
            "attack": {"name": "clean", "rate": 0},
        }
    }), encoding="utf-8")

    validate_source(run_dir, "acm", "han", 1)
    destination = tmp_path / "catalog" / "checkpoint.pt"
    assert materialize_checkpoint(run_dir / "checkpoint.pt", destination, False) in {
        "hardlink", "copy"
    }
    assert destination.read_bytes() == b"checkpoint"
    assert materialize_checkpoint(run_dir / "checkpoint.pt", destination, False) == "existing"


def test_validate_source_rejects_wrong_identity(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in ("checkpoint.pt", "metrics.json"):
        (run_dir / name).write_bytes(b"{}")
    (run_dir / "status.json").write_text(
        json.dumps({"state": "completed"}), encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(json.dumps({
        "experiment": {
            "dataset": "dblp",
            "model": {"name": "han"},
            "seeds": {"train": 1},
            "attack": {"name": "clean", "rate": 0},
        }
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="identity mismatch"):
        validate_source(run_dir, "acm", "han", 1)
