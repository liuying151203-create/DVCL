import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.legacy import build_legacy_command
from dvcl_bench.paths import ExperimentLayout
from dvcl_bench.specs import AttackSpec, ExperimentSpec, ModelSpec, SeedSpec


def main() -> int:
    try:
        AttackSpec(name="clean", rate=5)
    except ValueError:
        pass
    else:
        raise AssertionError("clean attack accepted a non-zero rate")

    spec = ExperimentSpec(
        protocol="dvcl_main",
        dataset="acm",
        split_name="seed_4",
        seeds=SeedSpec(split=4, attack=7, train=9),
        attack=AttackSpec(name="prbcd", rate=5),
        model=ModelSpec(name="dvcl", backend="legacy"),
    )
    layout = ExperimentLayout(ROOT)
    run_path = str(layout.run_dir(spec)).replace("\\", "/")
    assert "split_seed_4/attack_seed_7/train_seed_9" in run_path
    command = build_legacy_command(spec, layout, "python")
    assert command[command.index("--seed") + 1] == "9"
    assert command[command.index("--attack-seed") + 1] == "7"
    native = ExperimentSpec(
        protocol="dvcl_main",
        dataset="acm",
        split_name="paper_seed_1",
        seeds=SeedSpec(split=1, attack=1, train=3),
        attack=AttackSpec(name="clean", rate=0),
        model=ModelSpec(name="dvcl", backend="native", config={"variant": "no_cl"}),
    )
    native_path = str(layout.run_dir(native)).replace("\\", "/")
    assert "/dvcl/no_cl/clean/" in native_path
    assert "split_seed_1/attack_seed_1/train_seed_3" in native_path
    print("DVCL experiment contracts: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
