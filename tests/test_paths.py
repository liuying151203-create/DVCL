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
