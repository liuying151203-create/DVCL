from pathlib import Path

from dvcl_bench.golden import (
    GoldenCase,
    build_reference_command,
    build_spec,
    load_golden_cases,
)


def test_golden_cases_reject_duplicate_ids():
    config = {
        "cases": [
            {"id": "same", "model": "hseco", "dataset": "acm"},
            {"id": "same", "model": "dvcl", "dataset": "dblp"},
        ]
    }
    try:
        load_golden_cases(config)
    except ValueError as exc:
        assert "Duplicate golden case id" in str(exc)
    else:
        raise AssertionError("duplicate golden ids must fail")


def test_reference_command_uses_frozen_attack_artifact():
    case = GoldenCase("case", "hseco", "acm", "prbcd", 5, 1, 7, 1)
    config = {"negative_noise_rate": 0.01, "legacy_checkpoint_semantics": True}
    spec = build_spec(case, config, "golden", "cuda:0", 200, 100)
    inputs = {
        "clean": Path("data/processed/acm/clean.pt"),
        "split": Path("data/splits/acm/paper_seed_1.pt"),
        "attack": Path("data/attacks/acm/prbcd/rate_5/seed_7/attack.pt"),
    }
    command = build_reference_command(
        spec, Path("reference"), "python", inputs, Path("metrics.log")
    )
    assert command[command.index("--seed") + 1] == "1"
    assert command[command.index("--attack_artifact_path") + 1] == str(inputs["attack"])
    assert command[command.index("--neg_noise_rate") + 1] == "0.01"
