import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.analyze_adaptive_pilot import (
    load_run_suite,
    spec_from_command,
)
from dvcl_bench.paths import ExperimentLayout


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit and aggregate the DVCL view-diagnosis pilot."
    )
    parser.add_argument(
        "--clean-config",
        default="configs/protocols/dvcl_view_diagnosis_clean_pilot_v1.yaml",
    )
    parser.add_argument(
        "--evaluation-config",
        default="configs/protocols/dvcl_view_diagnosis_pilot_v1.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/analysis/dvcl_view_diagnosis_pilot_v1",
    )
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def collect_clean_rows(config_path):
    config, commands = _commands(config_path)
    rows = []
    issues = []
    completed = 0
    for command in commands:
        spec = spec_from_command(command)
        run_dir = ExperimentLayout(ROOT).run_dir(spec)
        payload = _completed_payload(run_dir, issues)
        if payload is None:
            continue
        completed += 1
        rows.append({
            "dataset": spec.dataset,
            "variant": spec.model.config["variant"],
            "train_seed": spec.seeds.train,
            "full_test_micro_f1": float(payload["metrics"]["micro_f1"]),
            "best_epoch": int(payload["best_epoch"]),
            "run_dir": str(run_dir.resolve()),
        })
    return rows, issues, len(commands), completed


def collect_evaluation_rows(config_path):
    config, commands = _commands(config_path)
    evaluation_budgets = [
        int(value) for value in config.get("evaluation_budgets", [])
    ]
    rows = []
    per_target = []
    issues = []
    completed = 0
    expected_logical = 0
    for command in commands:
        spec = spec_from_command(command)
        run_dir = ExperimentLayout(ROOT).run_dir(spec)
        budgets = evaluation_budgets if spec.attack.adaptive else [int(spec.attack.rate)]
        expected_logical += len(budgets)
        payload = _completed_payload(run_dir, issues)
        if payload is None:
            continue
        completed += 1
        diagnostics = payload.get("diagnostics", {})
        if diagnostics.get("checkpoint_reused") is not True:
            issues.append(f"checkpoint not reused: {run_dir}")
            continue
        if diagnostics.get("optimizer_steps") != 0:
            issues.append(f"optimizer steps during evaluation: {run_dir}")
            continue
        if spec.attack.adaptive:
            evaluations = diagnostics.get("budget_evaluations", {})
            for budget in budgets:
                value = evaluations.get(str(budget))
                if value is None:
                    issues.append(f"missing budget {budget}: {run_dir}")
                    continue
                _append_evaluation(
                    rows, per_target, spec, run_dir, budget,
                    value["metrics"], value["diagnostics"],
                )
        else:
            _append_evaluation(
                rows, per_target, spec, run_dir, int(spec.attack.rate),
                payload["metrics"], diagnostics,
            )
    return rows, per_target, issues, len(commands), completed, expected_logical


def _append_evaluation(rows, per_target, spec, run_dir, budget, metrics, diagnostics):
    view = diagnostics.get("view_diagnostics")
    if not view:
        rows.append({
            "_issue": f"missing view diagnostics: {run_dir} rate={budget}"
        })
        return
    clean_micro = float(diagnostics["clean_target_metrics"]["micro_f1"])
    row = {
        "dataset": spec.dataset,
        "variant": spec.model.config["variant"],
        "attack": spec.attack.name,
        "rate": int(budget),
        "attack_seed": spec.seeds.attack,
        "train_seed": spec.seeds.train,
        "clean_target_micro_f1": clean_micro,
        "attacked_target_micro_f1": float(metrics["micro_f1"]),
        "micro_f1_drop": clean_micro - float(metrics["micro_f1"]),
        "attack_success_rate": float(diagnostics["attack_success_rate"]),
        "run_dir": str(run_dir.resolve()),
    }
    for section in ("clean", "attacked", "drift", "gate"):
        for key, value in view.get(section, {}).items():
            if isinstance(value, (int, float)):
                row[f"{section}_{key}"] = value
    rows.append(row)
    base = {
        key: row[key]
        for key in (
            "dataset", "variant", "attack", "rate", "attack_seed",
            "train_seed", "run_dir",
        )
    }
    per_target.extend({**base, **value} for value in view.get("per_target", []))


def aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        if "_issue" not in row:
            grouped[(row["dataset"], row["variant"], row["attack"], row["rate"])].append(row)
    identifiers = {
        "dataset", "variant", "attack", "rate", "attack_seed", "train_seed",
        "run_dir",
    }
    output = []
    for key, values in sorted(grouped.items()):
        row = dict(zip(("dataset", "variant", "attack", "rate"), key))
        row["n"] = len(values)
        numeric_keys = sorted(set.intersection(*(
            {
                name for name, value in item.items()
                if name not in identifiers and isinstance(value, (int, float))
            }
            for item in values
        )))
        for name in numeric_keys:
            samples = [float(value[name]) for value in values]
            row[f"{name}_mean"] = statistics.fmean(samples)
            row[f"{name}_std"] = (
                statistics.stdev(samples) if len(samples) > 1 else 0.0
            )
        output.append(row)
    return output


def _commands(config_path):
    run_suite = load_run_suite()
    config = run_suite.load_config(config_path)
    return config, list(run_suite.commands(config, sys.executable, ROOT))


def _completed_payload(run_dir, issues):
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "metrics.json"
    if not status_path.is_file() or not metrics_path.is_file():
        issues.append(f"missing: {run_dir}")
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        issues.append(f"not completed: {run_dir}")
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    clean_config = (ROOT / args.clean_config).resolve()
    evaluation_config = (ROOT / args.evaluation_config).resolve()
    output_root = (ROOT / args.output_root).resolve()
    clean_rows, clean_issues, clean_expected, clean_completed = collect_clean_rows(
        clean_config
    )
    rows, per_target, issues, expected, completed, expected_logical = (
        collect_evaluation_rows(evaluation_config)
    )
    embedded_issues = [row["_issue"] for row in rows if "_issue" in row]
    rows = [row for row in rows if "_issue" not in row]
    issues.extend(embedded_issues)
    summary = aggregate_rows(rows)
    write_csv(output_root / "clean.csv", clean_rows)
    write_csv(output_root / "runs.csv", rows)
    write_csv(output_root / "summary.csv", summary)
    write_csv(output_root / "per_target.csv", per_target)
    audit = {
        "clean_config": str(clean_config),
        "evaluation_config": str(evaluation_config),
        "clean_physical_runs": f"{clean_completed}/{clean_expected}",
        "evaluation_physical_runs": f"{completed}/{expected}",
        "logical_results": f"{len(rows)}/{expected_logical}",
        "clean_issues": clean_issues,
        "evaluation_issues": issues,
        "ok": (
            clean_completed == clean_expected
            and completed == expected
            and len(rows) == expected_logical
            and not clean_issues
            and not issues
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"clean={clean_completed}/{clean_expected} "
        f"evaluation={completed}/{expected} "
        f"logical={len(rows)}/{expected_logical} "
        f"issues={len(clean_issues) + len(issues)}"
    )
    return 0 if audit["ok"] or args.allow_partial else 1


if __name__ == "__main__":
    raise SystemExit(main())
