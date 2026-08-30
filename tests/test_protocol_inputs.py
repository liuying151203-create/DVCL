import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_protocol_inputs", ROOT / "scripts" / "check_protocol_inputs.py"
)
CHECK_INPUTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK_INPUTS)


def test_main_protocol_has_expected_unique_inputs():
    config = CHECK_INPUTS.load_config(ROOT / "configs" / "protocols" / "dvcl_main.yaml")
    requirements = list(CHECK_INPUTS.protocol_requirements(config, ROOT))
    assert len(requirements) == 24
    assert sum(value["kind"] == "clean" for value in requirements) == 2
    assert sum(value["kind"] == "split" for value in requirements) == 2
    assert sum(value["kind"] == "attack" for value in requirements) == 20
    assert any(
        value["kind"] == "attack"
        and value["dataset"] == "dblp"
        and value["attack"] == "heteprbcd"
        and value["rate"] == 25
        for value in requirements
    )


def test_adaptive_formal_protocol_includes_pattern_inputs_and_checkpoints():
    config = CHECK_INPUTS.load_config(
        ROOT / "configs" / "protocols" / "adaptive_target_evasion_v1.yaml"
    )
    requirements = list(CHECK_INPUTS.protocol_requirements(config, ROOT))
    assert len(requirements) == 114
    assert sum(value["kind"] == "clean" for value in requirements) == 3
    assert sum(value["kind"] == "split" for value in requirements) == 3
    assert sum(value["kind"] == "attack" for value in requirements) == 9
    assert sum(value["kind"] == "checkpoint" for value in requirements) == 99
    assert any(
        value["kind"] == "attack"
        and value["path"] == ROOT / (
            "outputs/attacks/adaptive_requests_v1/cand_64/aminer/"
            "adaptive_query/rate_5/seed_3/attack.pt"
        )
        for value in requirements
    )
    assert any(
        value["kind"] == "checkpoint"
        and value["model"] == "dvcl"
        and value["train_seed"] == 3
        and value["path"] == ROOT / (
            "outputs/checkpoints/adaptive_clean_v1/dblp/dvcl/"
            "train_seed_3/checkpoint.pt"
        )
        for value in requirements
    )
    assert not any(
        value["kind"] == "checkpoint" and value["train_seed"] in {4, 5}
        for value in requirements
    )


def test_variant_protocol_audits_paired_variant_checkpoints_and_attacks(tmp_path):
    config = {
        "protocol": "variant_audit",
        "dataset": "dblp",
        "model": "dvcl",
        "variants": [{"name": "graph_hard"}, {"name": "han_semantic"}],
        "checkpoint_pattern": (
            "checkpoints/{dataset}/{model_variant}/attack_{attack_seed}/"
            "train_{train_seed}.pt"
        ),
        "attacks": [{
            "name": "adaptive_query",
            "variant": "cand_64",
            "rates": [5],
            "path_pattern": (
                "attacks/{dataset}/{model_variant}/attack_{attack_seed}/"
                "train_{train_seed}.pt"
            ),
        }],
        "seeds": {
            "split": [1],
            "pairs": [
                {"attack": 1, "train": 1},
                {"attack": 2, "train": 2},
            ],
        },
    }
    requirements = list(CHECK_INPUTS.protocol_requirements(config, tmp_path))
    checkpoints = [
        value for value in requirements if value["kind"] == "checkpoint"
    ]
    attacks = [value for value in requirements if value["kind"] == "attack"]
    assert len(checkpoints) == 4
    assert len(attacks) == 4
    assert {
        (value["model_variant"], value["attack_seed"], value["train_seed"])
        for value in checkpoints
    } == {
        ("graph_hard", 1, 1),
        ("graph_hard", 2, 2),
        ("han_semantic", 1, 1),
        ("han_semantic", 2, 2),
    }
