import argparse
import itertools
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CUDA_OOM_EXIT_CODE = 75


def parse_args():
    parser = argparse.ArgumentParser(description="Expand and run a DVCL experiment suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--attack-variant", action="append", dest="attack_variants")
    parser.add_argument("--rate", action="append", type=float, dest="rates")
    parser.add_argument("--split-seed", action="append", type=int, dest="split_seeds")
    parser.add_argument("--attack-seed", action="append", type=int, dest="attack_seeds")
    parser.add_argument("--train-seed", action="append", type=int, dest="train_seeds")
    parser.add_argument("--device")
    parser.add_argument("--oom-retries", type=int, default=0)
    parser.add_argument("--oom-retry-delay", type=float, default=60.0)
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


def select_datasets(config, selected):
    if not selected:
        return config
    selected = set(selected)
    available = set(config.get("datasets", [config.get("dataset")]))
    missing = selected - available
    if missing:
        raise ValueError(f"Unknown selected datasets: {sorted(missing)}")
    result = dict(config)
    if "datasets" not in config:
        result["dataset"] = next(iter(selected))
        return result
    result["datasets"] = [
        dataset for dataset in config["datasets"] if dataset in selected
    ]
    return result


def select_attacks(config, selected_variants=None, selected_rates=None):
    if not selected_variants and not selected_rates:
        return config
    selected_variants = set(selected_variants or [])
    selected_rates = set(selected_rates or [])
    available_variants = {
        attack.get("variant", "default") for attack in config.get("attacks", [])
    }
    missing = selected_variants - available_variants
    if missing:
        raise ValueError(f"Unknown selected attack variants: {sorted(missing)}")
    result = dict(config)
    attacks = []
    for attack in config.get("attacks", []):
        if selected_variants and attack.get("variant", "default") not in selected_variants:
            continue
        value = dict(attack)
        if selected_rates:
            rates = [0] if attack["name"] == "clean" else attack.get("rates", [])
            value["rates"] = [rate for rate in rates if float(rate) in selected_rates]
            if not value["rates"]:
                continue
        attacks.append(value)
    result["attacks"] = attacks
    return result


def select_seeds(config, split=None, attack=None, train=None):
    if not split and not attack and not train:
        return config
    if "seeds" not in config and "variants" in config:
        available = {
            "split": {int(config.get("split_seed", 1))},
            "attack": {int(config.get("attack_seed", 1))},
            "train": {int(value) for value in config.get("train_seeds", [1])},
        }
        selected_values = {"split": split, "attack": attack, "train": train}
        for name, selected in selected_values.items():
            if not selected:
                continue
            missing = set(selected) - available[name]
            if missing:
                raise ValueError(f"Unknown selected {name} seeds: {sorted(missing)}")
        result = dict(config)
        if split:
            result["split_seed"] = int(split[0])
        if attack:
            result["attack_seed"] = int(attack[0])
        if train:
            selected = set(train)
            result["train_seeds"] = [
                value for value in config.get("train_seeds", [1])
                if int(value) in selected
            ]
        return result
    result = dict(config)
    seeds = dict(config.get("seeds", {}))
    pairs = seeds.get("pairs")
    available_from_pairs = {}
    if pairs is not None:
        normalized_pairs = _normalize_seed_pairs(pairs)
        available_from_pairs = {
            "attack": {pair["attack"] for pair in normalized_pairs},
            "train": {pair["train"] for pair in normalized_pairs},
        }
    for name, selected in (("split", split), ("attack", attack), ("train", train)):
        if not selected:
            continue
        selected = set(selected)
        available = available_from_pairs.get(name, set(seeds.get(name, [1])))
        missing = selected - available
        if missing:
            raise ValueError(f"Unknown selected {name} seeds: {sorted(missing)}")
        if name == "split" or pairs is None:
            seeds[name] = [value for value in seeds.get(name, [1]) if value in selected]
    if pairs is not None:
        normalized_pairs = [
            pair for pair in normalized_pairs
            if (not attack or pair["attack"] in set(attack))
            and (not train or pair["train"] in set(train))
        ]
        if not normalized_pairs:
            raise ValueError("Seed selection leaves no configured attack/train pairs")
        seeds["pairs"] = normalized_pairs
    result["seeds"] = seeds
    return result


def _normalize_seed_pairs(pairs):
    normalized = []
    seen = set()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict) or "attack" not in pair or "train" not in pair:
            raise ValueError(
                f"seeds.pairs[{index}] must define integer attack and train seeds"
            )
        value = {"attack": int(pair["attack"]), "train": int(pair["train"])}
        key = (value["attack"], value["train"])
        if key in seen:
            raise ValueError(f"Duplicate attack/train seed pair: {key}")
        seen.add(key)
        normalized.append(value)
    if not normalized:
        raise ValueError("seeds.pairs must not be empty")
    return normalized


def seed_dimensions(config):
    seeds = config.get("seeds", {})
    split_seeds = [int(value) for value in seeds.get("split", [1])]
    if "pairs" in seeds:
        if "attack" in seeds or "train" in seeds:
            raise ValueError(
                "Use either seeds.pairs or Cartesian seeds.attack/seeds.train, not both"
            )
        pairs = _normalize_seed_pairs(seeds["pairs"])
        return [
            (split_seed, pair["attack"], pair["train"])
            for split_seed, pair in itertools.product(split_seeds, pairs)
        ]
    return list(itertools.product(
        split_seeds,
        [int(value) for value in seeds.get("attack", [1])],
        [int(value) for value in seeds.get("train", [1])],
    ))


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
        seed_dimensions(config),
    )
    for dataset, model, attack, seed_values in dimensions:
        split_seed, attack_seed, train_seed = seed_values
        rates = [0] if attack["name"] == "clean" else attack.get("rates", [])
        model_config = resolve_model_config(model, Path(base_dir))
        for rate in rates:
            yield command_for(
                python_bin, protocol, dataset, model["name"], model.get("backend", "native"),
                attack, rate, split_seed, attack_seed, train_seed, training,
                config.get("device", "cuda:0"), model_config,
                config.get("split_name_pattern", "paper_seed_{seed}").format(seed=split_seed),
                config.get("checkpoint_pattern"),
            )


def ablation_commands(config, python_bin):
    training = config.get("training", {})
    datasets = config.get("datasets", [config.get("dataset")])
    if any(dataset is None for dataset in datasets):
        raise ValueError("Ablation suites must define dataset or datasets")
    if "seeds" in config:
        seeds = seed_dimensions(config)
    else:
        seeds = list(itertools.product(
            [int(config.get("split_seed", 1))],
            [int(config.get("attack_seed", 1))],
            [int(value) for value in config.get("train_seeds", [1])],
        ))
    model_defaults = resolve_model_config({
        "config_path": config.get("model_config_path"),
        "config": config.get("model_config", {}),
    }, ROOT)
    for dataset, variant, attack, seed_values in itertools.product(
        datasets, config["variants"], config["attacks"], seeds
    ):
        split_seed, attack_seed, train_seed = seed_values
        rates = attack.get("rates", [0] if attack["name"] == "clean" else [])
        model_config = {**model_defaults, **variant.get("model_config", {})}
        model_config["variant"] = variant["name"]
        for rate in rates:
            yield command_for(
                python_bin, config.get("protocol", "dvcl_main"), dataset,
                config["model"], config.get("backend", "native"), attack, rate,
                split_seed, attack_seed, train_seed, training,
                config.get("device", "cuda:0"), model_config,
                config.get(
                    "split_name_pattern", config.get("split_name", "paper_seed_{seed}")
                ).format(seed=split_seed),
                config.get("checkpoint_pattern"),
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
    checkpoint_pattern=None,
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
                model_variant=model_config.get("variant", "default"),
                rate=f"{rate:g}", seed=attack_seed,
                split_seed=split_seed, attack_seed=attack_seed,
                train_seed=train_seed,
            ),
        ])
    if checkpoint_pattern:
        command.extend([
            "--checkpoint-source",
            checkpoint_pattern.format(
                dataset=dataset, model=model, attack=attack["name"],
                variant=attack.get("variant", "default"), rate=f"{rate:g}",
                model_variant=model_config.get("variant", "default"),
                seed=attack_seed, split_seed=split_seed,
                attack_seed=attack_seed, train_seed=train_seed,
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
    config = select_datasets(config, args.datasets)
    config = select_attacks(config, args.attack_variants, args.rates)
    config = select_seeds(
        config, args.split_seeds, args.attack_seeds, args.train_seeds
    )
    if args.device:
        config = {**config, "device": args.device}
    failures = 0
    for command in commands(config, args.python_bin, ROOT):
        if args.dry_run:
            command.append("--dry-run")
        if args.force:
            command.append("--force")
        print(shlex.join(command), flush=True)
        result = run_with_oom_retries(
            command,
            cwd=str(ROOT),
            retries=args.oom_retries,
            delay=args.oom_retry_delay,
        )
        if result.returncode:
            failures += 1
            if not args.continue_on_error:
                return result.returncode
    return 1 if failures else 0


def run_with_oom_retries(
    command, cwd, retries=0, delay=60.0,
    runner=subprocess.run, sleeper=time.sleep,
):
    if retries < 0:
        raise ValueError("OOM retries must be non-negative")
    for attempt in range(retries + 1):
        result = runner(command, cwd=cwd, check=False)
        if result.returncode != CUDA_OOM_EXIT_CODE or attempt == retries:
            return result
        print(
            f"CUDA OOM; retry {attempt + 1}/{retries} after {delay:g}s",
            flush=True,
        )
        sleeper(delay)


if __name__ == "__main__":
    raise SystemExit(main())
