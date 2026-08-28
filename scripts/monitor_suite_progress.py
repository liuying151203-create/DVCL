import argparse
from collections import defaultdict
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.analyze_adaptive_pilot import load_run_suite, spec_from_command
from dvcl_bench.paths import ExperimentLayout


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record audited suite progress and ETA at a fixed interval."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=int, default=1800)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--partition-by-attack", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def suite_snapshot(config_path, now=None, workers=1, partition_by_attack=False):
    now = now or datetime.now(timezone.utc)
    run_suite = load_run_suite()
    config = run_suite.load_config(config_path)
    budgets = [int(value) for value in config.get("evaluation_budgets", [])]
    counts = {"completed": 0, "running": 0, "failed": 0, "missing": 0}
    logical_completed = 0
    logical_expected = 0
    active = []
    completed_durations = {}
    pending_groups = []
    layout = ExperimentLayout(ROOT)
    commands = list(run_suite.commands(config, sys.executable, ROOT))
    for command in commands:
        spec = spec_from_command(command)
        logical_units = len(budgets) if spec.attack.adaptive and budgets else 1
        logical_expected += logical_units
        run_dir = layout.run_dir(spec)
        status_path = run_dir / "status.json"
        if not status_path.is_file():
            state = "missing"
            status = {}
        else:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            state = status.get("state", "missing")
            if state not in counts:
                state = "missing"
        counts[state] += 1
        if state == "completed":
            logical_completed += logical_units
            manifest_path = run_dir / "manifest.json"
            if manifest_path.is_file():
                duration = status_path.stat().st_mtime - manifest_path.stat().st_mtime
                if duration > 0:
                    completed_durations.setdefault(
                        (spec.dataset, spec.attack.name), []
                    ).append(duration)
        elif state == "running":
            active.append({
                "dataset": spec.dataset,
                "variant": spec.model.config.get("variant", "default"),
                "attack": spec.attack.name,
                "rate": spec.attack.rate,
                "pid": status.get("pid"),
                "started_at": status.get("started_at"),
                "run_dir": str(run_dir.resolve()),
            })
        if state != "completed":
            pending_groups.append((spec.dataset, spec.attack.name))
    historical_eta = estimate_historical_eta(
        pending_groups,
        completed_durations,
        workers,
        partition_by_attack=partition_by_attack,
    )
    return {
        "timestamp": now.isoformat(),
        "config": str(config_path.resolve()),
        "physical": {"expected": len(commands), **counts},
        "logical": {
            "expected": logical_expected,
            "completed": logical_completed,
        },
        "remaining": len(commands) - counts["completed"],
        "historical_eta_hours": (
            historical_eta / 3600 if historical_eta is not None else None
        ),
        "eta_partition": "attack" if partition_by_attack else "pooled",
        "workers": workers,
        "active": active,
    }


def estimate_historical_eta(
    pending_groups, completed_durations, workers, partition_by_attack=False,
):
    all_durations = [
        duration for values in completed_durations.values() for duration in values
    ]
    fallback = statistics.median(all_durations) if all_durations else None
    if not pending_groups or fallback is None:
        return None
    estimates = []
    for group in pending_groups:
        values = completed_durations.get(group)
        estimates.append((
            group[1], statistics.median(values) if values else fallback
        ))
    if not partition_by_attack:
        return sum(duration for _, duration in estimates) / workers
    partitions = defaultdict(float)
    for attack, duration in estimates:
        partitions[attack] += duration
    return max(partitions.values())


def add_eta(snapshot, history):
    current_time = datetime.fromisoformat(snapshot["timestamp"])
    completed = snapshot["physical"]["completed"]
    previous = next((
        row for row in reversed(history)
        if row.get("physical", {}).get("completed", completed) < completed
    ), None)
    if previous is None:
        snapshot["throughput_per_hour"] = None
        snapshot["eta_hours"] = snapshot.get("historical_eta_hours")
        snapshot["eta_source"] = (
            "historical_run_durations"
            if snapshot["eta_hours"] is not None else None
        )
        return snapshot
    previous_time = datetime.fromisoformat(previous["timestamp"])
    elapsed_hours = (current_time - previous_time).total_seconds() / 3600
    delta = completed - int(previous["physical"]["completed"])
    throughput = delta / elapsed_hours if elapsed_hours > 0 else 0.0
    snapshot["throughput_per_hour"] = throughput or None
    if snapshot.get("eta_partition") == "attack":
        snapshot["eta_hours"] = snapshot.get("historical_eta_hours")
        snapshot["eta_source"] = (
            "historical_attack_partition"
            if snapshot["eta_hours"] is not None else None
        )
        return snapshot
    snapshot["eta_hours"] = (
        snapshot["remaining"] / throughput if throughput > 0 else None
    )
    snapshot["eta_source"] = (
        "observed_throughput" if snapshot["eta_hours"] is not None else None
    )
    return snapshot


def read_history(path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def record(config_path, output_path, workers=1, partition_by_attack=False):
    history = read_history(output_path)
    snapshot = add_eta(suite_snapshot(
        config_path,
        workers=workers,
        partition_by_attack=partition_by_attack,
    ), history)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    eta = snapshot["eta_hours"]
    eta_text = f"{eta:.2f}" if eta is not None else "unknown"
    print(
        f"completed={snapshot['physical']['completed']}/"
        f"{snapshot['physical']['expected']} "
        f"logical={snapshot['logical']['completed']}/"
        f"{snapshot['logical']['expected']} "
        f"running={snapshot['physical']['running']} "
        f"failed={snapshot['physical']['failed']} "
        f"eta_hours={eta_text}",
        flush=True,
    )
    return snapshot


def main():
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    output_path = (ROOT / args.output).resolve()
    if args.interval <= 0:
        raise ValueError("interval must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    while True:
        snapshot = record(
            config_path,
            output_path,
            args.workers,
            partition_by_attack=args.partition_by_attack,
        )
        if args.once or snapshot["physical"]["completed"] == snapshot["physical"]["expected"]:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
