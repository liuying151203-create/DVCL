import argparse
import itertools
import json
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--device")
    return parser.parse_args()


def load_config(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def select_models(config, selected):
    if not selected:
        return config
    if "models" not in config:
        raise ValueError("Model selection is only supported for model-list suites")
    selected = set(selected)
    available = {model["name"] for model in config["models"]}
    missing = selected - available
    if missing:
        raise ValueError(f"Unknown selected models: {sorted(missing)}")
    result = dict(config)
    result["models"] = [model for model in config["models"] if model["name"] in selected]
    return result


def select_variants(config, selected):
    if not selected:
        return config
    if "variants" not in config:
        raise ValueError("Variant selection is only supported for ablation suites")
    selected = set(selected)
    available = {variant["name"] for variant in config["variants"]}
    missing = selected - available
    if missing:
        raise ValueError(f"Unknown selected variants: {sorted(missing)}")
    result = dict(config)
    result["variants"] = [
        variant for variant in config["variants"] if variant["name"] in selected
    ]
    return result


def commands(config, python_bin, base_dir=ROOT):
    if "variants" in config:
        yield from ablation_commands(config, python_bin)
        return
    protocol = config.get("protocol", "dvcl_main")
    training = config.get("training", {})
    seeds = config.get("seeds", {})
    dimensions = itertools.product(
        config["datasets"],
        config["models"],
        config["attacks"],
        seeds.get("split", [1]),
        seeds.get("attack", [1]),
        seeds.get("train", [1]),
    )
    for dataset, model, attack, split_seed, attack_seed, train_seed in dimensions:
        rates = [0] if attack["name"] == "clean" else attack.get("rates", [])
        model_config = resolve_model_config(model, Path(base_dir))
        for rate in rates:
            yield command_for(
                python_bin, protocol, dataset, model["name"], model.get("backend", "native"),
                attack, rate, split_seed, attack_seed, train_seed, training,
                config.get("device", "cuda:0"), model_config,
                config.get("split_name_pattern", "paper_seed_{seed}").format(seed=split_seed),
            )


def ablation_commands(config, python_bin):
    training = config.get("training", {})
    split_seed = int(config.get("split_seed", 1))
    attack_seed = int(config.get("attack_seed", 1))
    for variant, attack, train_seed in itertools.product(
        config["variants"], config["attacks"], config.get("train_seeds", [1])
    ):
        rates = attack.get("rates", [0] if attack["name"] == "clean" else [])
        model_config = dict(variant.get("model_config", {}))
        model_config["variant"] = variant["name"]
        for rate in rates:
            yield command_for(
                python_bin, config.get("protocol", "dvcl_main"), config["dataset"],
                config["model"], config.get("backend", "native"), attack, rate,
                split_seed, attack_seed, train_seed, training,
                config.get("device", "cuda:0"), model_config,
                config.get("split_name", f"paper_seed_{split_seed}"),
            )


def resolve_model_config(model, base_dir: Path):
    values = dict(model.get("config", {}))
    if model.get("config_path"):
        path = Path(model["config_path"])
        if not path.is_absolute():
            path = base_dir / path
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        values = {**loaded.get("config", loaded), **values}
    return values


def command_for(
    python_bin, protocol, dataset, model, backend, attack, rate,
    split_seed, attack_seed, train_seed, training, device, model_config, split_name,
):
    command = [
        python_bin,
        str(ROOT / "scripts" / "run_experiment.py"),
        "--protocol", protocol,
        "--model", model,
        "--backend", backend,
        "--dataset", dataset,
        "--attack", attack["name"],
        "--rate", str(rate),
        "--threat-model", attack.get("threat_model", "poisoning"),
        "--scope", attack.get("scope", "global"),
        "--attack-variant", attack.get("variant", "default"),
        "--split-name", split_name,
        "--split-seed", str(split_seed),
        "--attack-seed", str(attack_seed),
        "--train-seed", str(train_seed),
        "--device", device,
        "--epochs", str(training.get("epochs", 200)),
        "--patience", str(training.get("patience", 100)),
        "--model-config-json", json.dumps(model_config, separators=(",", ":")),
    ]
    if attack.get("path_pattern"):
        command.extend([
            "--attack-path",
            attack["path_pattern"].format(
                dataset=dataset, model=model, attack=attack["name"],
                variant=attack.get("variant", "default"),
                rate=f"{rate:g}", seed=attack_seed
            ),
        ])
    if attack.get("adaptive", False):
        command.append("--adaptive")
    return command


def main() -> int:
    args = parse_args()
    path = Path(args.config)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    config = load_config(path)
    config = select_models(config, args.models)
    config = select_variants(config, args.variants)
    if args.device:
        config = {**config, "device": args.device}
    failures = 0
    for command in commands(config, args.python_bin, ROOT):
        if args.dry_run:
            command.append("--dry-run")
        if args.force:
            command.append("--force")
        print(shlex.join(command), flush=True)
        result = subprocess.run(command, cwd=str(ROOT), check=False)
        if result.returncode:
            failures += 1
            if not args.continue_on_error:
                return result.returncode
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
