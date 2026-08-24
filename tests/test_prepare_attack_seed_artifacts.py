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
