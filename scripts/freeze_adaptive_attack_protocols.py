import argparse
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL_SPECS = (
    ("han", "native", "configs/models/han.yaml"),
    ("heterosage", "native", "configs/models/heterosage.yaml"),
    ("rohe", "native", "configs/models/rohe.yaml"),
    ("heteroguard", "native", "configs/models/heteroguard.yaml"),
    ("fastrohgcn", "native", "configs/models/fastrohgcn.yaml"),
    ("hgt", "openhgnn", "configs/models/hgt.yaml"),
    ("magnn", "openhgnn", "configs/models/magnn.yaml"),
    ("heco", "openhgnn", "configs/models/heco.yaml"),
    ("simplehgn", "openhgnn", "configs/models/simplehgn.yaml"),
    ("hseco", "native", "configs/models/hseco_native.yaml"),
    ("dvcl", "native", "configs/models/dvcl.yaml"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Freeze adaptive confirmation and formal protocols from pilot selection."
    )
    parser.add_argument(
        "--selection",
        default="outputs/analysis/adaptive_attack_strength_screen_v1/selection.json",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def models(names=None):
    selected = set(names) if names else None
    return [
        {"name": name, "backend": backend, "config_path": config_path}
        for name, backend, config_path in MODEL_SPECS
        if selected is None or name in selected
    ]


def attack(candidate_size: int, request_root: str):
    return {
        "name": "adaptive_query",
        "variant": f"cand_{candidate_size}",
        "rates": [5],
        "threat_model": "evasion",
        "scope": "target",
        "adaptive": True,
        "path_pattern": (
            f"{request_root}/cand_{candidate_size}/{{dataset}}/adaptive_query/"
            "rate_{rate}/seed_{attack_seed}/attack.pt"
        ),
    }


def confirmation_config(candidate_size: int, selection_path: Path):
    return {
        "protocol": "adaptive_attack_strength_confirmation_v1",
        "device": "cuda:0",
        "selection": selection_metadata(selection_path, candidate_size),
        "evaluation_budgets": [1, 3, 5],
        "datasets": ["acm", "dblp"],
        "models": models(("han", "heteroguard", "hseco", "dvcl")),
        "split_name_pattern": "paper_seed_{seed}",
        "checkpoint_pattern": (
            "outputs/checkpoints/adaptive_clean_v1/{dataset}/{model}/"
            "train_seed_{train_seed}/checkpoint.pt"
        ),
        "attacks": [attack(
            candidate_size, "outputs/pilots/adaptive_confirmation_requests"
        )],
        "seeds": {"split": [1], "attack": [1, 2, 3], "train": [1, 2]},
        "training": {"epochs": 200, "patience": 100},
    }


def formal_config(candidate_size: int, selection_path: Path):
    return {
        "protocol": "adaptive_target_evasion_v1",
        "device": "cuda:0",
        "selection": selection_metadata(selection_path, candidate_size),
        "evaluation_budgets": [1, 3, 5],
        "datasets": ["acm", "dblp", "aminer"],
        "models": models(),
        "split_name_pattern": "paper_seed_{seed}",
        "checkpoint_pattern": (
            "outputs/checkpoints/adaptive_clean_v1/{dataset}/{model}/"
            "train_seed_{train_seed}/checkpoint.pt"
        ),
        "attacks": [attack(candidate_size, "outputs/attacks/adaptive_requests_v1")],
        "seeds": {
            "split": [1],
            "pairs": [
                {"attack": 1, "train": 1},
                {"attack": 2, "train": 2},
                {"attack": 3, "train": 3},
            ],
        },
        "training": {"epochs": 200, "patience": 100},
    }


def selection_metadata(path: Path, candidate_size: int):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "candidate_additions": candidate_size,
        "candidate_deletions": candidate_size,
    }


def write_config(path: Path, value, force: bool):
    if path.exists() and not force:
        raise FileExistsError(f"Protocol already exists: {path}; use --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    selection_path = Path(args.selection)
    if not selection_path.is_absolute():
        selection_path = ROOT / selection_path
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    candidate_size = int(selection["selected_candidate_size"])
    outputs = (
        (
            ROOT / "configs" / "protocols" / "adaptive_attack_strength_confirmation_v1.yaml",
            confirmation_config(candidate_size, selection_path),
        ),
        (
            ROOT / "configs" / "protocols" / "adaptive_target_evasion_v1.yaml",
            formal_config(candidate_size, selection_path),
        ),
    )
    for path, value in outputs:
        write_config(path, value, args.force)
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
