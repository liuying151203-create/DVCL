"""Configuration and command builders for audited legacy/native golden runs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import yaml

from .artifacts import file_sha256
from .paths import ExperimentLayout
from .specs import AttackSpec, ExperimentSpec, ModelSpec, SeedSpec


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    model: str
    dataset: str
    attack: str
    rate: float
    split_seed: int
    attack_seed: int
    train_seed: int


def load_golden_config(path: Path) -> Dict[str, object]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not payload.get("cases"):
        raise ValueError("Golden config must define at least one case")
    return payload


def load_golden_cases(config: Mapping[str, object]) -> List[GoldenCase]:
    defaults = dict(config.get("seeds", {}))
    cases = []
    seen = set()
    for raw in config["cases"]:
        case = GoldenCase(
            case_id=str(raw["id"]),
            model=str(raw["model"]).lower(),
            dataset=str(raw["dataset"]).lower(),
            attack=str(raw.get("attack", "clean")).lower(),
            rate=float(raw.get("rate", 0)),
            split_seed=int(raw.get("split_seed", defaults.get("split", 1))),
            attack_seed=int(raw.get("attack_seed", defaults.get("attack", 1))),
            train_seed=int(raw.get("train_seed", defaults.get("train", 1))),
        )
        if case.case_id in seen:
            raise ValueError(f"Duplicate golden case id: {case.case_id}")
        seen.add(case.case_id)
        cases.append(case)
    return cases


def resolve_model_config(config: Mapping[str, object], model: str, root: Path):
    definitions = config.get("models", {})
    if model not in definitions:
        raise ValueError(f"Missing golden model configuration: {model}")
    definition = dict(definitions[model])
    values = dict(definition.get("config", {}))
    if definition.get("config_path"):
        path = Path(str(definition["config_path"]))
        if not path.is_absolute():
            path = root / path
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        values = {**loaded.get("config", loaded), **values}
    values["legacy_checkpoint_semantics"] = True
    return values


def build_spec(
    case: GoldenCase,
    model_config: Mapping[str, object],
    protocol: str,
    device: str,
    epochs: int,
    patience: int,
) -> ExperimentSpec:
    return ExperimentSpec(
        protocol=protocol,
        dataset=case.dataset,
        split_name=f"paper_seed_{case.split_seed}",
        seeds=SeedSpec(case.split_seed, case.attack_seed, case.train_seed),
        attack=AttackSpec(case.attack, case.rate),
        model=ModelSpec(case.model, "native", dict(model_config)),
        device=device,
        epochs=epochs,
        patience=patience,
    )


def input_paths(spec: ExperimentSpec, layout: ExperimentLayout):
    paths = {
        "clean": layout.clean_path(spec.dataset),
        "split": layout.split_path(spec.dataset, spec.split_name),
    }
    if spec.attack.name != "clean":
        paths["attack"] = layout.attack_path(
            spec.dataset, spec.attack.name, spec.attack.rate, spec.seeds.attack
        )
    return paths


def build_reference_command(
    spec: ExperimentSpec,
    reference_root: Path,
    python_bin: str,
    inputs: Mapping[str, Path],
    metrics_path: Path,
) -> List[str]:
    entrypoint = {"hseco": "our_global.py", "dvcl": "our_dvcl.py"}.get(spec.model.name)
    if entrypoint is None:
        raise ValueError(f"Golden reference is unsupported for model: {spec.model.name}")
    attack_name = {
        "clean": "RND",
        "rnd": "RND",
        "prbcd": "PRBCD",
        "heteprbcd": "HetePRBCD",
    }[spec.attack.name]
    rate = float(spec.attack.rate)
    command = [
        python_bin,
        str(Path(reference_root) / entrypoint),
        "--use_artifacts",
        "--dataname", spec.dataset,
        "--atk_name", attack_name,
        "--atk_rate", str(int(rate) if rate.is_integer() else rate),
        "--seed", str(spec.seeds.train),
        "--split_name", spec.split_name,
        "--data_root", str(Path(inputs["clean"]).parents[2]),
        "--device", spec.device,
        "--clean_artifact_path", str(inputs["clean"]),
        "--split_artifact_path", str(inputs["split"]),
        "--log_fp", str(metrics_path),
        "--epochs", str(spec.epochs),
        "--patience", str(spec.patience),
    ]
    if "attack" in inputs:
        command.extend(["--attack_artifact_path", str(inputs["attack"])])
    values = spec.model.config
    if spec.model.name == "hseco":
        command.extend(["--neg_noise_rate", str(values["negative_noise_rate"])])
    else:
        option_map = {
            "hidden_dim": "--dvcl_hidden_dim",
            "heads": "--dvcl_num_heads",
            "dropout": "--dropout",
            "feature_mask_rate": "--feature_mask_rate",
            "knn_k": "--knn_k",
            "knn_mode": "--feature_knn_mode",
            "view_mode": "--dvcl_view_mode",
            "fusion_mode": "--dvcl_fusion_mode",
            "temperature": "--dvcl_temperature",
            "lambda_han": "--lambda_han",
            "lambda_dvcl": "--lambda_dvcl",
            "learning_rate": "--lr",
            "weight_decay": "--weight_decay",
            "semantic_hidden_dim": "--hidden_units",
        }
        for name, option in option_map.items():
            command.extend([option, str(values[name])])
        command.extend(["--num_heads", str(values["semantic_heads"])])
    return command


def build_current_command(
    spec: ExperimentSpec, root: Path, python_bin: str, force: bool = False
) -> List[str]:
    command = [
        python_bin,
        str(Path(root) / "scripts" / "run_experiment.py"),
        "--protocol", spec.protocol,
        "--model", spec.model.name,
        "--backend", "native",
        "--dataset", spec.dataset,
        "--attack", spec.attack.name,
        "--rate", str(spec.attack.rate),
        "--split-name", spec.split_name,
        "--split-seed", str(spec.seeds.split),
        "--attack-seed", str(spec.seeds.attack),
        "--train-seed", str(spec.seeds.train),
        "--device", spec.device,
        "--epochs", str(spec.epochs),
        "--patience", str(spec.patience),
        "--model-config-json", json.dumps(spec.model.config, separators=(",", ":")),
    ]
    if force:
        command.append("--force")
    return command


def audit_payload(
    case: GoldenCase,
    spec: ExperimentSpec,
    inputs: Mapping[str, Path],
    reference_root: Path,
    reference_command: List[str],
    current_command: List[str],
):
    return {
        "schema_version": 1,
        "case": asdict(case),
        "spec": asdict(spec),
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for name, path in inputs.items()
        },
        "repositories": {
            "reference": _git_state(reference_root),
            "current": _git_state(Path(__file__).resolve().parents[2]),
        },
        "python": {"executable": sys.executable, "version": sys.version},
        "commands": {"reference": reference_command, "current": current_command},
    }


def resolve_reference_root(explicit: Optional[str], project_root: Path) -> Path:
    value = explicit or os.environ.get("DVCL_PRIVATE_HSECO_ROOT")
    root = Path(value).expanduser() if value else Path(project_root).parent / "HSeCo"
    root = root.resolve()
    if not (root / "our_global.py").is_file() or not (root / "our_dvcl.py").is_file():
        raise FileNotFoundError(f"Compatible HSeCo reference repository not found: {root}")
    return root


def _git_state(root: Path):
    def run(*args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "root": str(Path(root).resolve()),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }
