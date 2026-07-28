import argparse
import itertools
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Expand and run a DVCL experiment suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def load_config(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def commands(config, python_bin):
    protocol = config.get("protocol", "dvcl_main")
    training = config.get("training", {})
    seeds = config.get("seeds", {})
    split_seeds = seeds.get("split", [1])
    attack_seeds = seeds.get("attack", [1])
    train_seeds = seeds.get("train", [1])
    for dataset, model, attack, split_seed, attack_seed, train_seed in itertools.product(
        config["datasets"],
        config["models"],
        config["attacks"],
        split_seeds,
        attack_seeds,
        train_seeds,
    ):
        rates = [0] if attack["name"] == "clean" else attack.get("rates", [])
        for rate in rates:
            yield [
                python_bin,
                str(ROOT / "scripts" / "run_experiment.py"),
                "--protocol", protocol,
                "--model", model["name"],
                "--backend", model.get("backend", "legacy"),
                "--dataset", dataset,
                "--attack", attack["name"],
                "--rate", str(rate),
                "--threat-model", attack.get("threat_model", "poisoning"),
                "--scope", attack.get("scope", "global"),
                "--split-name", f"seed_{split_seed}",
                "--split-seed", str(split_seed),
                "--attack-seed", str(attack_seed),
                "--train-seed", str(train_seed),
                "--device", config.get("device", "cuda:0"),
                "--epochs", str(training.get("epochs", 200)),
                "--patience", str(training.get("patience", 100)),
            ]


def main() -> int:
    args = parse_args()
    config = load_config((ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    failures = 0
    for command in commands(config, args.python_bin):
        if args.dry_run:
            command.append("--dry-run")
        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=str(ROOT), check=False)
        if result.returncode != 0:
            failures += 1
            if not args.continue_on_error:
                return result.returncode
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
