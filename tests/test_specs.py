import pytest

from dvcl_bench.specs import AttackSpec, ExperimentSpec


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
