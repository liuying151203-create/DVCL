import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_attack_artifacts", ROOT / "scripts" / "prepare_dblp_attack_pilot.py"
)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def test_provenance_validation_includes_dataset_identity():
    provenance = {
        "dataset": "acm",
        "attack": "HetePRBCD",
        "rate": 5,
        "seed": 1,
        "constrained": True,
        "biased": True,
    }
    PREPARE.validate_provenance(provenance, "acm", "heteprbcd", 5, 1)
    with pytest.raises(ValueError, match="dataset"):
        PREPARE.validate_provenance(provenance, "dblp", "heteprbcd", 5, 1)
