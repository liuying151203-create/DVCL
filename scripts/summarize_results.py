import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("accuracy", "micro_f1", "macro_f1")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize completed DVCL experiments.")
    parser.add_argument("--run-root", default=str(ROOT / "outputs" / "runs"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "summaries"))
    return parser.parse_args()


def load_rows(run_root: Path):
    rows = []
    for path in run_root.rglob("metrics.json") if run_root.exists() else []:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "metrics" not in payload:
            continue
        row = {
            key: payload.get(key)
            for key in (
                "protocol", "dataset", "model", "variant", "attack", "rate",
                "split_seed", "attack_seed", "train_seed", "best_epoch", "stopped_epoch",
            )
        }
        row.update(payload["metrics"])
        row["run_dir"] = str(path.parent)
        rows.append(row)
    return rows


def aggregate(rows):
    keys = (
        "protocol", "dataset", "model", "variant", "attack", "rate",
        "split_seed", "attack_seed",
    )
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for identity, values in sorted(groups.items()):
        row = dict(zip(keys, identity))
        row["runs"] = len(values)
        row["train_seeds"] = ",".join(str(item["train_seed"]) for item in values)
        for metric in METRICS:
            samples = [float(item[metric]) for item in values]
            row[f"{metric}_mean"] = statistics.mean(samples)
            row[f"{metric}_std"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        result.append(row)
    return result


def attack_averages(rows):
    per_seed = defaultdict(list)
    for row in rows:
        if row["attack"] == "clean":
            continue
        key = (
            row["protocol"], row["dataset"], row["model"], row["variant"],
            row["split_seed"], row["attack_seed"], row["train_seed"],
        )
        per_seed[key].append(row)
    grouped = defaultdict(list)
    condition_counts = defaultdict(list)
    for key, values in per_seed.items():
        seed_row = {metric: statistics.mean(float(item[metric]) for item in values) for metric in METRICS}
        grouped[key[:-1]].append(seed_row)
        condition_counts[key[:-1]].append(len(values))
    result = []
    names = ("protocol", "dataset", "model", "variant", "split_seed", "attack_seed")
    for key, values in sorted(grouped.items()):
        row = dict(zip(names, key))
        row["train_seed_runs"] = len(values)
        row["conditions_per_seed_min"] = min(condition_counts[key])
        row["conditions_per_seed_max"] = max(condition_counts[key])
        for metric in METRICS:
            samples = [item[metric] for item in values]
            row[f"{metric}_mean"] = statistics.mean(samples)
            row[f"{metric}_std"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        result.append(row)
    return result


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        if not rows:
            return
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.run_root))
    output = Path(args.output_dir)
    write_csv(rows, output / "runs.csv")
    write_csv(aggregate(rows), output / "summary.csv")
    write_csv(attack_averages(rows), output / "attack_average.csv")
    print(f"Summarized {len(rows)} completed runs into {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
