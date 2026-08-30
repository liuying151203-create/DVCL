import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.artifacts import (
    file_sha256,
    load_attack_artifact,
    load_clean_artifact,
    load_split_artifact,
)
from dvcl_bench.attacks import verify_attack
from dvcl_bench.paths import ExperimentLayout


def parse_args():
    parser = argparse.ArgumentParser(description="Audit frozen inputs for an experiment protocol.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--report")
    return parser.parse_args()


def load_config(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def protocol_requirements(config, root=ROOT):
    root = Path(root)
    layout = ExperimentLayout(root)
    datasets = config.get("datasets", [config.get("dataset")])
    if not datasets or datasets == [None]:
        raise ValueError("Protocol config must define dataset or datasets")
    seeds = config.get("seeds", {})
    split_seeds = seeds.get("split", [config.get("split_seed", 1)])
    seed_pairs = _seed_pairs(
        seeds,
        [config.get("attack_seed", 1)],
        config.get("train_seeds", [1]),
    )
    model_dimensions = _model_dimensions(config)
    split_pattern = config.get("split_name_pattern")
    seen = set()
    for dataset in datasets:
        clean_path = layout.clean_path(dataset)
        clean_key = ("clean", str(clean_path))
        if clean_key not in seen:
            seen.add(clean_key)
            yield {"kind": "clean", "dataset": dataset, "path": clean_path}
        for split_seed in split_seeds:
            split_name = (
                split_pattern.format(seed=split_seed)
                if split_pattern
                else config.get("split_name", f"paper_seed_{split_seed}")
            )
            split_path = layout.split_path(dataset, split_name)
            split_key = ("split", str(split_path))
            if split_key not in seen:
                seen.add(split_key)
                yield {
                    "kind": "split",
                    "dataset": dataset,
                    "split_name": split_name,
                    "split_seed": int(split_seed),
                    "path": split_path,
                }
            checkpoint_pattern = config.get("checkpoint_pattern")
            if checkpoint_pattern:
                for model_name, model_variant in model_dimensions:
                    for attack_seed, train_seed in seed_pairs:
                        checkpoint_path = _pattern_path(
                            root,
                            checkpoint_pattern,
                            dataset=dataset,
                            model=model_name,
                            model_variant=model_variant,
                            split_seed=split_seed,
                            attack_seed=attack_seed,
                            train_seed=train_seed,
                        )
                        checkpoint_key = ("checkpoint", str(checkpoint_path))
                        if checkpoint_key in seen:
                            continue
                        seen.add(checkpoint_key)
                        yield {
                            "kind": "checkpoint",
                            "dataset": dataset,
                            "model": model_name,
                            "model_variant": model_variant,
                            "split_seed": int(split_seed),
                            "attack_seed": int(attack_seed),
                            "train_seed": int(train_seed),
                            "path": checkpoint_path,
                        }
            for attack in config.get("attacks", []):
                if attack["name"] == "clean":
                    continue
                for attack_seed, train_seed in seed_pairs:
                    for rate in attack.get("rates", []):
                        for model_name, model_variant in model_dimensions:
                            if attack.get("path_pattern"):
                                attack_path = _pattern_path(
                                    root,
                                    attack["path_pattern"],
                                    dataset=dataset,
                                    model=model_name,
                                    attack=attack["name"],
                                    variant=attack.get("variant", "default"),
                                    model_variant=model_variant,
                                    rate=f"{rate:g}",
                                    seed=attack_seed,
                                    split_seed=split_seed,
                                    attack_seed=attack_seed,
                                    train_seed=train_seed,
                                )
                            else:
                                attack_path = layout.attack_path(
                                    dataset, attack["name"], rate,
                                    int(attack_seed),
                                )
                            attack_key = (
                                "attack", str(attack_path), str(split_path)
                            )
                            if attack_key in seen:
                                continue
                            seen.add(attack_key)
                            yield {
                                "kind": "attack",
                                "dataset": dataset,
                                "attack": attack["name"],
                                "rate": float(rate),
                                "attack_seed": int(attack_seed),
                                "train_seed": int(train_seed),
                                "model": model_name,
                                "model_variant": model_variant,
                                "path": attack_path,
                                "clean_path": clean_path,
                                "split_path": split_path,
                            }


def _seed_pairs(seeds, default_attack, default_train):
    pairs = seeds.get("pairs")
    if pairs is None:
        return [
            (int(attack_seed), int(train_seed))
            for attack_seed in seeds.get("attack", default_attack)
            for train_seed in seeds.get("train", default_train)
        ]
    values = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict) or "attack" not in pair or "train" not in pair:
            raise ValueError(
                f"seeds.pairs[{index}] must define integer attack and train seeds"
            )
        value = (int(pair["attack"]), int(pair["train"]))
        if value not in values:
            values.append(value)
    return values


def _model_dimensions(config):
    if "variants" in config:
        model_name = config.get("model", "")
        return [
            (model_name, variant["name"])
            for variant in config["variants"]
        ]
    models = config.get("models") or [{"name": config.get("model", "")}]
    return [(model["name"], "default") for model in models]


def _pattern_path(root: Path, pattern: str, **values):
    path = Path(pattern.format(**values))
    return path if path.is_absolute() else root / path


def audit_protocol_inputs(config, root=ROOT):
    rows = []
    for requirement in protocol_requirements(config, root):
        row = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in requirement.items()
            if key not in {"clean_path", "split_path"}
        }
        path = requirement["path"]
        try:
            if requirement["kind"] == "clean":
                artifact = load_clean_artifact(path)
                if artifact.dataset != requirement["dataset"]:
                    raise ValueError("clean dataset identity mismatch")
            elif requirement["kind"] == "split":
                artifact = load_split_artifact(path)
                if artifact.dataset != requirement["dataset"]:
                    raise ValueError("split dataset identity mismatch")
                if artifact.split_name != requirement["split_name"]:
                    raise ValueError("split name identity mismatch")
                if artifact.seed != requirement["split_seed"]:
                    raise ValueError("split seed identity mismatch")
            elif requirement["kind"] == "attack":
                clean = load_clean_artifact(requirement["clean_path"])
                split = load_split_artifact(requirement["split_path"])
                artifact = load_attack_artifact(path)
                verification = verify_attack(clean, split, artifact)
                row["verification"] = verification
                if not verification["ok"]:
                    raise ValueError("; ".join(verification["issues"]))
            elif requirement["kind"] == "checkpoint":
                if not path.is_file():
                    raise FileNotFoundError(path)
            else:
                raise ValueError(f"Unsupported input kind: {requirement['kind']}")
            row["sha256"] = file_sha256(path)
            row["ok"] = True
        except Exception as exc:
            row["ok"] = False
            row["error"] = str(exc)
        rows.append(row)
    failed = sum(not row["ok"] for row in rows)
    return {
        "protocol": config.get("protocol", config.get("suite", "unknown")),
        "summary": {"total": len(rows), "passed": len(rows) - failed, "failed": failed},
        "inputs": rows,
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    report = audit_protocol_inputs(load_config(config_path), ROOT)
    output = (
        Path(args.report)
        if args.report
        else ROOT / "outputs" / "audits" / f"{config_path.stem}-inputs.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"Report: {output}")
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
