import argparse
import json
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.legacy import build_legacy_command, run_legacy
from dvcl_bench.manifest import build_manifest, save_json
from dvcl_bench.paths import ExperimentLayout
from dvcl_bench.specs import AttackSpec, ExperimentSpec, ModelSpec, SeedSpec


def parse_args():
    parser = argparse.ArgumentParser(description="Run one DVCL benchmark experiment.")
    parser.add_argument("--protocol", default="dvcl_main")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="legacy", choices=["legacy", "native", "openhgnn"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--attack", default="clean")
    parser.add_argument("--rate", type=float, default=0)
    parser.add_argument("--threat-model", default="poisoning", choices=["poisoning", "evasion"])
    parser.add_argument("--scope", default="global", choices=["global", "target"])
    parser.add_argument("--split-name", default="seed_1")
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--attack-seed", type=int, default=1)
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--model-config-json", default="{}")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    return args, extra


def main() -> int:
    args, extra = parse_args()
    spec = ExperimentSpec(
        protocol=args.protocol,
        dataset=args.dataset.lower(),
        split_name=args.split_name,
        seeds=SeedSpec(args.split_seed, args.attack_seed, args.train_seed),
        attack=AttackSpec(args.attack.lower(), args.rate, args.threat_model, args.scope),
        model=ModelSpec(args.model.lower(), args.backend, json.loads(args.model_config_json)),
        device=args.device,
        epochs=args.epochs,
        patience=args.patience,
        extra_args=tuple(extra),
    )
    layout = ExperimentLayout(ROOT)
    if spec.model.backend != "legacy":
        raise SystemExit(
            f"Backend '{spec.model.backend}' has a registered integration boundary but no frozen "
            "training adapter yet. Use legacy for HSeCo/DVCL until the native/OpenHGNN audit is complete."
        )

    command = build_legacy_command(spec, layout, args.python_bin)
    print(shlex.join(command))
    if args.dry_run:
        return 0

    inputs = {
        "clean": layout.clean_path(spec.dataset),
        "split": layout.split_path(spec.dataset, spec.split_name),
    }
    if spec.attack.name != "clean":
        inputs["attack"] = layout.attack_path(
            spec.dataset, spec.attack.name, spec.attack.rate, spec.seeds.attack
        )
    run_dir = layout.run_dir(spec)
    manifest = build_manifest(spec, ROOT.parent, inputs)
    save_json(manifest, run_dir / "manifest.json")
    returncode = run_legacy(command, layout.legacy_hseco)
    save_json({"returncode": returncode}, run_dir / "status.json")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
