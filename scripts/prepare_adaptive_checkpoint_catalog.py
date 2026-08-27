import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("acm", "dblp", "aminer")
CORE_MODELS = ("han", "heterosage", "hseco", "dvcl")
ROBUST_MODELS = ("rohe", "heteroguard", "fastrohgcn")
OPENHGNN_MODELS = ("hgt", "magnn", "heco", "simplehgn")
MODELS = CORE_MODELS + ROBUST_MODELS + OPENHGNN_MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an audited catalog of reusable clean checkpoints."
    )
    parser.add_argument(
        "--output-root",
        default="outputs/checkpoints/adaptive_clean_v1",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def source_protocol(dataset: str, model: str) -> str:
    if dataset == "aminer":
        return "aminer_poisoning_main_v1"
    if model in CORE_MODELS:
        return f"{dataset}_poisoning_main_v1"
    if model in ROBUST_MODELS:
        return "robust_baselines_poisoning_v1"
    if model in OPENHGNN_MODELS:
        return "openhgnn_baselines_poisoning_v1"
    raise ValueError(f"Unsupported model: {model}")


def source_run(root: Path, dataset: str, model: str, train_seed: int) -> Path:
    return (
        root
        / "outputs"
        / "runs"
        / source_protocol(dataset, model)
        / dataset
        / model
        / "default"
        / "clean"
        / "rate_0"
        / "split_seed_1"
        / "attack_seed_1"
        / f"train_seed_{train_seed}"
    )


def validate_source(run_dir: Path, dataset: str, model: str, train_seed: int):
    required = ("checkpoint.pt", "manifest.json", "metrics.json", "status.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete source run {run_dir}: missing {missing}")
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        raise ValueError(f"Source run is not completed: {run_dir}")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    experiment = manifest.get("experiment", {})
    expected = {
        "dataset": dataset,
        "model": model,
        "train_seed": train_seed,
    }
    actual = {
        "dataset": experiment.get("dataset"),
        "model": experiment.get("model", {}).get("name"),
        "train_seed": experiment.get("seeds", {}).get("train"),
    }
    attack = experiment.get("attack", {})
    if actual != expected or attack.get("name") != "clean" or float(
        attack.get("rate", -1)
    ) != 0.0:
        raise ValueError(
            f"Source run identity mismatch: expected={expected}, actual={actual}, "
            f"attack={attack}"
        )
    return manifest


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_checkpoint(source: Path, destination: Path, force: bool) -> str:
    source_hash = file_sha256(source)
    if destination.exists():
        if file_sha256(destination) == source_hash:
            return "existing"
        if not force:
            raise FileExistsError(
                f"Catalog checkpoint differs from source: {destination}; use --force"
            )
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def build_catalog(root: Path, output_root: Path, force: bool = False):
    entries = []
    for dataset in DATASETS:
        for model in MODELS:
            for train_seed in range(1, 6):
                run_dir = source_run(root, dataset, model, train_seed)
                manifest = validate_source(run_dir, dataset, model, train_seed)
                source = run_dir / "checkpoint.pt"
                destination = (
                    output_root
                    / dataset
                    / model
                    / f"train_seed_{train_seed}"
                    / "checkpoint.pt"
                )
                mode = materialize_checkpoint(source, destination, force)
                checksum = file_sha256(source)
                if file_sha256(destination) != checksum:
                    raise RuntimeError(f"Checkpoint hash mismatch: {destination}")
                entries.append({
                    "dataset": dataset,
                    "model": model,
                    "train_seed": train_seed,
                    "source_protocol": source_protocol(dataset, model),
                    "source_run": str(run_dir.resolve()),
                    "source_checkpoint": str(source.resolve()),
                    "catalog_checkpoint": str(destination.resolve()),
                    "sha256": checksum,
                    "materialization": mode,
                    "source_git_commit": manifest.get("git_commit"),
                })
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected": len(DATASETS) * len(MODELS) * 5,
        "entries": entries,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    catalog_path = output_root / "catalog.json"
    catalog_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload, catalog_path


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    payload, catalog_path = build_catalog(ROOT, output_root, args.force)
    print(f"Cataloged {len(payload['entries'])}/{payload['expected']} checkpoints")
    print(f"Wrote {catalog_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
