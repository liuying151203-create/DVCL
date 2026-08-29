import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_attack_seed_artifacts.py"
SPEC = importlib.util.spec_from_file_location("prepare_attack_seed_artifacts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_generator_command_freezes_attack_semantics(tmp_path):
    prbcd = MODULE.generator_command("acm", "prbcd", 15, 2, 3, 100000, tmp_path / "p.pt")
    hete = MODULE.generator_command(
        "dblp", "heteprbcd", 25, 3, 4, 200000, tmp_path / "h.pt"
    )
    assert "--constrained" in prbcd
    assert "--biased" not in prbcd
    assert "--constrained" in hete
    assert "--biased" in hete
    assert hete[hete.index("--seed") + 1] == "3"
    assert hete[hete.index("--block-size") + 1] == "200000"
    assert "--relation-scope" not in prbcd


def test_relation_scope_uses_isolated_paths_and_command(tmp_path):
    source = MODULE.source_path("aminer", "prbcd", 15, 1, "pr")
    artifact = MODULE.artifact_path("aminer", "prbcd", 15, 1, "pr")
    command = MODULE.generator_command(
        "aminer", "prbcd", 15, 1, 0, 50000, tmp_path / "source.pt", "pr"
    )
    assert "f1_relations/pr/prbcd/rate_15/seed_1/source.pt" in str(source)
    assert "f1_relations/pr/prbcd/rate_15/seed_1/attack.pt" in str(artifact)
    assert command[command.index("--relation-scope") + 1] == "pr"


def test_generator_command_can_freeze_data_root(tmp_path):
    command = MODULE.generator_command(
        "aminer", "prbcd", 15, 1, 0, 50000, tmp_path / "source.pt",
        "joint", tmp_path / "data",
    )
    assert command[command.index("--data-root") + 1] == str(
        (tmp_path / "data").resolve()
    )


def test_validate_provenance_rejects_mislabeled_attack():
    provenance = {
        "dataset": "acm", "attack": "PRBCD", "rate": 5, "seed": 2,
        "constrained": True, "biased": True,
    }
    try:
        MODULE.validate_provenance(provenance, "acm", "prbcd", 5, 2)
    except ValueError as exc:
        assert "biased" in str(exc)
    else:
        raise AssertionError("mislabeled PRBCD provenance must fail")


def test_validate_provenance_checks_relation_budget():
    provenance = {
        "dataset": "aminer", "attack": "PRBCD", "rate": 15, "seed": 1,
        "constrained": True, "biased": False, "relation_scope": "pr",
        "budget": [["paper", "pa", "author"]],
    }
    try:
        MODULE.validate_provenance(
            provenance, "aminer", "prbcd", 15, 1, "pr"
        )
    except ValueError as exc:
        assert "budget" in str(exc)
    else:
        raise AssertionError("mislabeled relation budget must fail")
