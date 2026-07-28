from pathlib import Path

from dvcl_bench.legacy import build_legacy_command
from dvcl_bench.paths import ExperimentLayout
from dvcl_bench.specs import AttackSpec, ExperimentSpec, ModelSpec, SeedSpec


def test_legacy_command_separates_attack_and_train_seed():
    spec = ExperimentSpec(
        protocol="dvcl_main",
        dataset="acm",
        split_name="seed_4",
        seeds=SeedSpec(split=4, attack=7, train=9),
        attack=AttackSpec(name="prbcd", rate=5),
        model=ModelSpec(name="dvcl", backend="legacy"),
    )
    command = build_legacy_command(spec, ExperimentLayout(Path("code")), "python")
    assert command[command.index("--seed") + 1] == "9"
    assert command[command.index("--attack-seed") + 1] == "7"
