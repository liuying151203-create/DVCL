import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize DVCL run manifests and statuses.")
    parser.add_argument("--run-root", default=str(ROOT / "outputs" / "runs"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "summaries" / "run_status.csv"))
    return parser.parse_args()


def flatten(manifest, status, run_dir):
    experiment = manifest["experiment"]
    return {
        "protocol": experiment["protocol"],
        "dataset": experiment["dataset"],
        "model": experiment["model"]["name"],
        "backend": experiment["model"]["backend"],
        "attack": experiment["attack"]["name"],
        "rate": experiment["attack"]["rate"],
        "threat_model": experiment["attack"]["threat_model"],
        "split_seed": experiment["seeds"]["split"],
        "attack_seed": experiment["seeds"]["attack"],
        "train_seed": experiment["seeds"]["train"],
        "returncode": status.get("returncode"),
        "git_commit": manifest.get("git_commit"),
        "run_dir": str(run_dir),
    }


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root)
    manifests = run_root.rglob("manifest.json") if run_root.exists() else []
    rows = []
    for manifest_path in manifests:
        run_dir = manifest_path.parent
        status_path = run_dir / "status.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        rows.append(flatten(manifest, status, run_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "protocol", "dataset", "model", "backend", "attack", "rate",
        "threat_model", "split_seed", "attack_seed", "train_seed",
        "returncode", "git_commit", "run_dir",
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
