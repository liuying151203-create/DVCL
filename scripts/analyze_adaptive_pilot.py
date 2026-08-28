import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import rankdata, t, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.paths import ExperimentLayout
from dvcl_bench.paper_analysis import holm_adjust
from dvcl_bench.specs import AttackSpec, ExperimentSpec, ModelSpec, SeedSpec


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit and aggregate the adaptive attack-strength pilot."
    )
    parser.add_argument(
        "--config",
        default="configs/protocols/adaptive_attack_strength_pilot_v1.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/analysis/adaptive_attack_strength_pilot_v1",
    )
    parser.add_argument("--stability-tolerance", type=float, default=0.02)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def options(values):
    result = {}
    index = 0
    while index < len(values):
        if values[index].startswith("--") and index + 1 < len(values):
            result[values[index]] = values[index + 1]
            index += 2
        else:
            index += 1
    return result


def spec_from_command(command):
    value = options(command[2:])
    return ExperimentSpec(
        protocol=value["--protocol"],
        dataset=value["--dataset"],
        split_name=value["--split-name"],
        seeds=SeedSpec(
            int(value["--split-seed"]),
            int(value["--attack-seed"]),
            int(value["--train-seed"]),
        ),
        attack=AttackSpec(
            value["--attack"], float(value["--rate"]),
            value["--threat-model"], value["--scope"],
            "--adaptive" in command, value["--attack-variant"],
        ),
        model=ModelSpec(
            value["--model"], value["--backend"],
            json.loads(value["--model-config-json"]),
        ),
        device=value["--device"],
        epochs=int(value["--epochs"]),
        patience=int(value["--patience"]),
    )


def collect_rows(config_path: Path):
    run_suite = load_run_suite()
    config = run_suite.load_config(config_path)
    evaluation_budgets = {
        int(value) for value in config.get("evaluation_budgets", [])
    }
    rows = []
    issues = []
    expected_runs = 0
    completed_runs = 0
    for command in run_suite.commands(config, sys.executable, ROOT):
        expected_runs += 1
        spec = spec_from_command(command)
        run_dir = ExperimentLayout(ROOT).run_dir(spec)
        status_path = run_dir / "status.json"
        metrics_path = run_dir / "metrics.json"
        if not status_path.is_file() or not metrics_path.is_file():
            issues.append(f"missing: {run_dir}")
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "completed":
            issues.append(f"not completed: {run_dir}")
            continue
        completed_runs += 1
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        diagnostics = payload.get("diagnostics", {})
        adaptive = diagnostics.get("adaptive_attack", {})
        candidate_size = int(spec.attack.variant.removeprefix("cand_"))
        checks = {
            "victim_model": adaptive.get("victim_model") == spec.model.name,
            "checkpoint_reused": diagnostics.get("checkpoint_reused") is True,
            "optimizer_steps": diagnostics.get("optimizer_steps") == 0,
            "candidate_additions": adaptive.get("candidate_additions") == candidate_size,
            "candidate_deletions": adaptive.get("candidate_deletions") == candidate_size,
            "budget": adaptive.get("budget_per_target") == int(spec.attack.rate),
            "budget_bound": adaptive.get("budget_utilization", 2.0) <= 1.0,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            issues.append(f"invalid {run_dir}: {failed}")
            continue
        budget_evaluations = diagnostics.get("budget_evaluations")
        if budget_evaluations:
            requested_budgets = evaluation_budgets or {int(spec.attack.rate)}
            missing_budgets = requested_budgets - {
                int(value) for value in budget_evaluations
            }
            if missing_budgets:
                issues.append(
                    f"missing budget evaluations {sorted(missing_budgets)}: {run_dir}"
                )
                continue
            for budget in sorted(requested_budgets):
                evaluation = budget_evaluations[str(budget)]
                rows.append(_result_row(
                    spec, run_dir, candidate_size, budget,
                    evaluation["metrics"], evaluation["diagnostics"],
                    evaluation["adaptive_attack"],
                ))
        else:
            rows.append(_result_row(
                spec, run_dir, candidate_size, int(spec.attack.rate),
                payload["metrics"], diagnostics, adaptive,
            ))
    return rows, issues, expected_runs, completed_runs


def _result_row(spec, run_dir, candidate_size, budget, metrics, diagnostics, adaptive):
    clean_micro = float(diagnostics["clean_target_metrics"]["micro_f1"])
    attacked_micro = float(metrics["micro_f1"])
    return {
            "dataset": spec.dataset,
            "model": spec.model.name,
            "candidate_size": candidate_size,
            "rate": int(budget),
            "train_seed": spec.seeds.train,
            "attack_seed": spec.seeds.attack,
            "clean_target_micro_f1": clean_micro,
            "attacked_target_micro_f1": attacked_micro,
            "micro_f1_drop": clean_micro - attacked_micro,
            "attack_success_rate": float(diagnostics["attack_success_rate"]),
            "clean_correct_targets": int(diagnostics["clean_correct_target_count"]),
            "attack_success_count": int(diagnostics["attack_success_count"]),
            "total_changes": int(adaptive["total_changes"]),
            "budget_utilization": float(adaptive["budget_utilization"]),
            "queries": int(adaptive["queries"]),
            "candidate_pool_sha256": adaptive["candidate_pool_sha256"],
            "run_dir": str(run_dir.resolve()),
        }


def aggregate_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"], row["model"], row["candidate_size"], row["rate"]
        )
        grouped[key].append(row)
    output = []
    metrics = (
        "clean_target_micro_f1", "attacked_target_micro_f1", "micro_f1_drop",
        "attack_success_rate", "budget_utilization", "total_changes", "queries",
    )
    for key, values in sorted(grouped.items()):
        row = dict(zip(("dataset", "model", "candidate_size", "rate"), key))
        row["n"] = len(values)
        for metric in metrics:
            samples = [float(value[metric]) for value in values]
            row[f"{metric}_mean"] = statistics.fmean(samples)
            row[f"{metric}_std"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        output.append(row)
    return output


def adaptive_average_ranks(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = (
            row["dataset"], row["rate"], row["attack_seed"], row["train_seed"]
        )
        grouped[key][row["model"]] = row["attacked_target_micro_f1"]
    samples = defaultdict(list)
    for key, values in grouped.items():
        models = sorted(values)
        ranks = rankdata([-values[model] for model in models], method="average")
        for model, rank in zip(models, ranks):
            samples[(key[0], model)].append(float(rank))
    return [
        {
            "dataset": dataset,
            "model": model,
            "conditions": len(values),
            "average_rank": statistics.fmean(values),
        }
        for (dataset, model), values in sorted(samples.items())
    ]


def paired_reference_comparisons(rows, reference="dvcl"):
    by_dataset_seed_model = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"], row["attack_seed"], row["train_seed"], row["model"]
        )
        by_dataset_seed_model[key].append(row["attacked_target_micro_f1"])
    models = sorted({row["model"] for row in rows if row["model"] != reference})
    comparisons = []
    for dataset in sorted({row["dataset"] for row in rows}):
        seeds = sorted({
            (row["attack_seed"], row["train_seed"])
            for row in rows if row["dataset"] == dataset
        })
        dataset_rows = []
        for baseline in models:
            differences = []
            for attack_seed, train_seed in seeds:
                reference_values = by_dataset_seed_model.get(
                    (dataset, attack_seed, train_seed, reference)
                )
                baseline_values = by_dataset_seed_model.get(
                    (dataset, attack_seed, train_seed, baseline)
                )
                if not reference_values or not baseline_values:
                    continue
                differences.append(
                    statistics.fmean(reference_values)
                    - statistics.fmean(baseline_values)
                )
            if not differences:
                continue
            ci_low, ci_high = _mean_t_interval(differences)
            dataset_rows.append({
                "dataset": dataset,
                "reference": reference,
                "baseline": baseline,
                "n": len(differences),
                "effect_pp": 100 * statistics.fmean(differences),
                "effect_ci_low_pp": 100 * ci_low,
                "effect_ci_high_pp": 100 * ci_high,
                "wins": sum(value > 1e-12 for value in differences),
                "ties": sum(abs(value) <= 1e-12 for value in differences),
                "losses": sum(value < -1e-12 for value in differences),
                "p_value": _wilcoxon_pvalue(differences),
            })
        adjusted = holm_adjust([row["p_value"] for row in dataset_rows])
        for row, p_holm in zip(dataset_rows, adjusted):
            row["p_holm"] = p_holm
            row["significant_0_05"] = p_holm < 0.05
        comparisons.extend(dataset_rows)
    return comparisons


def _mean_t_interval(values, confidence=0.95):
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean
    deviation = statistics.stdev(values)
    if deviation == 0:
        return mean, mean
    critical = float(t.ppf((1 + confidence) / 2, len(values) - 1))
    margin = critical * deviation / math.sqrt(len(values))
    return mean - margin, mean + margin


def _wilcoxon_pvalue(differences):
    if all(abs(value) <= 1e-12 for value in differences):
        return 1.0
    return float(wilcoxon(differences, alternative="two-sided").pvalue)


def choose_candidate(summary, tolerance: float):
    sizes = sorted({int(row["candidate_size"]) for row in summary})
    if not sizes:
        raise ValueError("Pilot summary is empty")
    by_condition = {
        (row["dataset"], row["model"], int(row["rate"]), int(row["candidate_size"])): row
        for row in summary
    }
    diagnostics = []
    for size in sizes:
        larger = [value for value in sizes if value > size]
        max_asr_gain = 0.0
        max_micro_drop_gain = 0.0
        missing_comparisons = 0
        for dataset, model, rate, current_size in list(by_condition):
            if current_size != size:
                continue
            current = by_condition[(dataset, model, rate, size)]
            for other_size in larger:
                other = by_condition.get((dataset, model, rate, other_size))
                if other is None:
                    missing_comparisons += 1
                    continue
                max_asr_gain = max(
                    max_asr_gain,
                    other["attack_success_rate_mean"]
                    - current["attack_success_rate_mean"],
                )
                max_micro_drop_gain = max(
                    max_micro_drop_gain,
                    current["attacked_target_micro_f1_mean"]
                    - other["attacked_target_micro_f1_mean"],
                )
        stable = (
            not missing_comparisons
            and max_asr_gain <= tolerance
            and max_micro_drop_gain <= tolerance
        )
        diagnostics.append({
            "candidate_size": size,
            "max_asr_gain_from_larger_pool": max_asr_gain,
            "max_additional_micro_f1_drop_from_larger_pool": max_micro_drop_gain,
            "missing_comparisons": missing_comparisons,
            "stable": stable,
        })
        if stable or not larger:
            return size, diagnostics
    return sizes[-1], diagnostics


def validate_candidate_hashes(rows):
    grouped = defaultdict(set)
    for row in rows:
        key = (
            row["dataset"], row["candidate_size"], row["rate"], row["attack_seed"]
        )
        grouped[key].add(row["candidate_pool_sha256"])
    return [
        {"condition": list(key), "hashes": sorted(values)}
        for key, values in sorted(grouped.items()) if len(values) != 1
    ]


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_run_suite():
    path = ROOT / "scripts" / "run_suite.py"
    spec = importlib.util.spec_from_file_location("adaptive_run_suite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    rows, issues, expected_runs, completed_runs = collect_rows(config_path)
    config = load_run_suite().load_config(config_path)
    evaluation_budgets = config.get("evaluation_budgets")
    expected = expected_runs * len(evaluation_budgets or [None])
    hash_issues = validate_candidate_hashes(rows)
    summary = aggregate_rows(rows)
    ranks = adaptive_average_ranks(rows)
    significance = paired_reference_comparisons(rows)
    write_csv(output_root / "runs.csv", rows)
    write_csv(output_root / "summary.csv", summary)
    write_csv(output_root / "average_ranks.csv", ranks)
    write_csv(output_root / "significance.csv", significance)
    selection = None
    if summary and not issues and not hash_issues:
        selected, selection_diagnostics = choose_candidate(
            summary, args.stability_tolerance
        )
        selection = {
            "selected_candidate_size": selected,
            "stability_tolerance": args.stability_tolerance,
            "diagnostics": selection_diagnostics,
        }
    audit = {
        "config": str(config_path.resolve()),
        "expected": expected,
        "completed": len(rows),
        "expected_physical_runs": expected_runs,
        "completed_physical_runs": completed_runs,
        "issues": issues,
        "candidate_hash_issues": hash_issues,
        "selection": selection,
        "ok": (
            len(rows) == expected
            and completed_runs == expected_runs
            and not issues
            and not hash_issues
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if selection:
        (output_root / "selection.json").write_text(
            json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        f"expected={expected} completed={len(rows)} "
        f"physical_runs={completed_runs}/{expected_runs} issues={len(issues)} "
        f"hash_issues={len(hash_issues)}"
    )
    if selection:
        print(f"selected candidate size: {selection['selected_candidate_size']}+{selection['selected_candidate_size']}")
    print(f"Wrote {output_root}")
    return 0 if audit["ok"] or args.allow_partial else 1


if __name__ == "__main__":
    raise SystemExit(main())
