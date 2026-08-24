import pytest

from dvcl_bench.specs import AttackSpec, ExperimentSpec, ModelSpec


def test_clean_attack_rejects_nonzero_rate():
    with pytest.raises(ValueError):
        AttackSpec(name="clean", rate=5)


def test_seeds_are_independent_in_mapping():
    spec = ExperimentSpec.from_mapping(
        {
            "dataset": "acm",
            "seeds": {"split": 1, "attack": 2, "train": 3},
            "attack": {"name": "prbcd", "rate": 5},
            "model": {"name": "dvcl", "backend": "legacy"},
        }
    )
    assert (spec.seeds.split, spec.seeds.attack, spec.seeds.train) == (1, 2, 3)


def test_hg_baseline_requires_target_evasion():
    with pytest.raises(ValueError, match="hg_baseline"):
        AttackSpec(name="hg_baseline", rate=3)
    value = AttackSpec(
        name="hg_baseline", rate=3, threat_model="evasion", scope="target"
    )
    assert value.threat_model == "evasion"


def test_dvcl_query_requires_adaptive_target_evasion():
    with pytest.raises(ValueError, match="adaptive target evasion"):
        AttackSpec(name="dvcl_adaptive_query", rate=1)
    value = AttackSpec(
        name="dvcl_adaptive_query", rate=1,
        threat_model="evasion", scope="target", adaptive=True,
    )
    assert value.adaptive is True


def test_openhgnn_models_use_strict_allowlist():
    assert ModelSpec("hgt", "openhgnn").name == "hgt"
    with pytest.raises(ValueError, match="OpenHGNN backend supports only"):
        ModelSpec("hgsl", "openhgnn")
