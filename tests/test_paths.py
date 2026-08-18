from pathlib import Path

from dvcl_bench.paths import ExperimentLayout
from dvcl_bench.specs import AttackSpec, ExperimentSpec, ModelSpec, SeedSpec


def test_run_path_contains_all_seed_dimensions():
    layout = ExperimentLayout(Path("code"))
    spec = ExperimentSpec(
        protocol="dvcl_main",
        dataset="acm",
        split_name="seed_1",
        seeds=SeedSpec(split=1, attack=2, train=3),
        attack=AttackSpec(name="prbcd", rate=5),
        model=ModelSpec(name="dvcl", backend="legacy"),
    )
    value = str(layout.run_dir(spec)).replace("\\", "/")
    assert "split_seed_1/attack_seed_2/train_seed_3" in value


def test_attack_variant_isolated_without_changing_default_paths():
    layout = ExperimentLayout(Path("code"))
    common = dict(
        protocol="diagnostic",
        dataset="acm",
        split_name="paper_seed_1",
        seeds=SeedSpec(split=1, attack=1, train=1),
        model=ModelSpec(name="dvcl", backend="native"),
    )
    default = ExperimentSpec(
        **common, attack=AttackSpec(name="prbcd", rate=5)
    )
    diagnostic = ExperimentSpec(
        **common,
        attack=AttackSpec(name="prbcd", rate=5, variant="unconstrained"),
    )
    assert "/prbcd/rate_5/" in str(layout.run_dir(default)).replace("\\", "/")
    assert "/prbcd_unconstrained/rate_5/" in str(layout.run_dir(diagnostic)).replace("\\", "/")
