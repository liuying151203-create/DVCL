import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.artifacts import file_sha256, load_attack_artifact, load_clean_artifact, load_split_artifact
from dvcl_bench.attacks import verify_attack
from dvcl_bench.manifest import save_json
from dvcl_bench.paths import ExperimentLayout


def parse_args():
    parser = argparse.ArgumentParser(description="Promote verified attack artifacts into the formal layout.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--split-name", default="paper_seed_1")
    parser.add_argument("--attack-seed", type=int, default=1)
    parser.add_argument("--attacks", nargs="+", default=["prbcd", "heteprbcd"])
    parser.add_argument("--rates", nargs="+", type=int, default=[5, 10, 15, 20, 25])
    parser.add_argument(
        "--archive-root",
        default=str(ROOT / "outputs" / "archive" / "attacks_pre_protocol_v2"),
    )
    return parser.parse_args()


def git_revision(repository):
    repository = Path(repository).resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repository, text=True
        ).strip()
    )
    if dirty:
        raise RuntimeError(f"Refusing promotion from dirty repository: {repository}")
    return {"path": str(repository), "commit": revision, "dirty": False}


def promote_one(source_dir, destination_dir, archive_dir, audit):
    required = [source_dir / "attack.pt", source_dir / "meta.json", source_dir / "verification.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source artifact files: {missing}")
    if archive_dir.exists():
        raise FileExistsError(f"Archive already exists: {archive_dir}")

    temporary_dir = destination_dir.parent / f".{destination_dir.name}.promoting"
    if temporary_dir.exists():
        raise FileExistsError(f"Temporary promotion directory exists: {temporary_dir}")
    temporary_dir.mkdir(parents=True)
    try:
        for source in required:
            shutil.copy2(source, temporary_dir / source.name)
        save_json(audit, temporary_dir / "promotion.json")
        archived = False
        if destination_dir.exists():
            archive_dir.parent.mkdir(parents=True, exist_ok=True)
            destination_dir.rename(archive_dir)
            archived = True
        try:
            temporary_dir.rename(destination_dir)
        except Exception:
            if archived and not destination_dir.exists():
                archive_dir.rename(destination_dir)
            raise
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def main():
    args = parse_args()
    layout = ExperimentLayout(ROOT)
    source_root = Path(args.source_root).resolve()
    archive_root = Path(args.archive_root).resolve()
    clean_path = layout.clean_path(args.dataset)
    split_path = layout.split_path(args.dataset, args.split_name)
    clean = load_clean_artifact(clean_path)
    split = load_split_artifact(split_path)
    repositories = {
        "dvcl": git_revision(ROOT),
        "hseco": git_revision(ROOT.parent / "HSeCo"),
        "hetero_guard": git_revision(ROOT.parent / "Hetero-Guard"),
    }

    for attack in args.attacks:
        for rate in args.rates:
            source_dir = source_root / args.dataset / attack / f"rate_{rate}"
            artifact_path = source_dir / "attack.pt"
            artifact = load_attack_artifact(artifact_path)
            report = verify_attack(clean, split, artifact)
            stored_report = json.loads((source_dir / "verification.json").read_text())
            if not report["ok"] or not stored_report.get("ok"):
                raise RuntimeError(f"Artifact verification failed: {source_dir}")
            destination_path = layout.attack_path(args.dataset, attack, rate, args.attack_seed)
            destination_dir = destination_path.parent
            archive_dir = (
                archive_root
                / args.dataset
                / attack
                / f"rate_{rate}"
                / f"seed_{args.attack_seed}"
            )
            audit = {
                "schema_version": 1,
                "promoted_at": datetime.now(timezone.utc).isoformat(),
                "dataset": args.dataset,
                "attack": attack,
                "rate": rate,
                "attack_seed": args.attack_seed,
                "split_name": args.split_name,
                "source": str(source_dir),
                "source_attack_sha256": file_sha256(artifact_path),
                "clean_sha256": file_sha256(clean_path),
                "split_sha256": file_sha256(split_path),
                "budget": report["budget"],
                "repositories": repositories,
            }
            promote_one(source_dir, destination_dir, archive_dir, audit)
            print(f"Promoted {attack} {rate}% -> {destination_path}")


if __name__ == "__main__":
    main()
