import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import rankdata, t, wilcoxon


def load_protocol_rows(run_root: Path, protocols):
    rows = []
    for protocol in protocols:
        protocol_root = run_root / protocol
        for path in protocol_root.rglob("metrics.json") if protocol_root.is_dir() else []:
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", {})
            if "micro_f1" not in metrics:
                continue
            row = {
                key: payload.get(key)
                for key in (
                    "protocol", "dataset", "model", "variant", "attack",
                    "attack_variant", "rate", "split_seed", "attack_seed",
                    "train_seed", "best_epoch", "stopped_epoch",
                )
            }
            row["micro_f1"] = float(metrics["micro_f1"])
            row["diagnostics"] = payload.get("diagnostics", {})
            row["run_dir"] = str(path.parent)
            rows.append(row)
    return rows


def summarize(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(float(row["micro_f1"]))
    result = []
    for key, values in sorted(groups.items()):
        item = dict(zip(fields, key))
        item.update({
            "n": len(values),
            "micro_f1_mean": statistics.mean(values),
            "micro_f1_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
        result.append(item)
    return result


def family_averages(rows, protocol, attacks=None):
    selected = [row for row in rows if row["protocol"] == protocol]
    if attacks is not None:
        selected = [row for row in selected if row["attack"] in attacks]
    per_seed = defaultdict(list)
    for row in selected:
        key = (
            row["dataset"], row["model"], row["attack"], row["attack_seed"],
            row["train_seed"],
        )
        per_seed[key].append(row["micro_f1"])
    grouped = defaultdict(list)
    for key, values in per_seed.items():
        grouped[key[:3]].append(statistics.mean(values))
    result = []
    for key, values in sorted(grouped.items()):
        result.append({
            "dataset": key[0],
            "model": key[1],
            "attack": key[2],
            "n": len(values),
            "micro_f1_mean": statistics.mean(values),
            "micro_f1_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    return result


def paired_significance(rows, reference="dvcl"):
    datasets = sorted({row["dataset"] for row in rows})
    attacks = sorted({row["attack"] for row in rows})
    models = sorted({row["model"] for row in rows})
    baselines = [model for model in models if model != reference]
    comparisons = []
    for dataset in datasets:
        for attack in [*attacks, "all"]:
            selected = [
                row for row in rows
                if row["dataset"] == dataset
                and (attack == "all" or row["attack"] == attack)
            ]
            by_unit = defaultdict(lambda: defaultdict(list))
            for row in selected:
                unit = (row["attack_seed"], row["train_seed"])
                by_unit[unit][row["model"]].append(row["micro_f1"])
            for baseline in baselines:
                pairs = [
                    (
                        statistics.mean(values[reference]),
                        statistics.mean(values[baseline]),
                    )
                    for values in by_unit.values()
                    if reference in values and baseline in values
                ]
                if not pairs:
                    continue
                differences = [left - right for left, right in pairs]
                comparison = paired_effect_statistics(differences)
                comparisons.append({
                    "dataset": dataset,
                    "attack": attack,
                    "correction_family": f"{dataset}:{attack}",
                    "reference": reference,
                    "baseline": baseline,
                    **comparison,
                })
    families = defaultdict(list)
    for row in comparisons:
        families[row["correction_family"]].append(row)
    for rows in families.values():
        adjusted = holm_adjust([row["p_value"] for row in rows])
        for row, p_holm in zip(rows, adjusted):
            row["p_holm"] = p_holm
            row["significant_0_05"] = p_holm < 0.05
    return comparisons


def paired_effect_statistics(differences, confidence=0.95):
    values = [float(value) for value in differences]
    if not values:
        raise ValueError("paired differences must not be empty")
    ci_low, ci_high = mean_t_interval(values, confidence)
    return {
        "n": len(values),
        "effect_pp": 100 * statistics.fmean(values),
        "effect_ci_low_pp": 100 * ci_low,
        "effect_ci_high_pp": 100 * ci_high,
        "wins": sum(value > 1e-12 for value in values),
        "ties": sum(abs(value) <= 1e-12 for value in values),
        "losses": sum(value < -1e-12 for value in values),
        "p_value": _wilcoxon_pvalue(values),
    }


def mean_t_interval(values, confidence=0.95):
    values = [float(value) for value in values]
    if not values:
        raise ValueError("values must not be empty")
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean
    deviation = statistics.stdev(values)
    if deviation == 0:
        return mean, mean
    critical = float(t.ppf((1 + confidence) / 2, len(values) - 1))
    margin = critical * deviation / math.sqrt(len(values))
    return mean - margin, mean + margin


def average_ranks(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = (
            row["dataset"], row["attack"], float(row["rate"]),
            row["attack_seed"], row["train_seed"],
        )
        groups[key][row["model"]] = row["micro_f1"]
    rank_samples = defaultdict(list)
    for key, values in groups.items():
        models = sorted(values)
        ranks = rankdata([-values[model] for model in models], method="average")
        for model, rank in zip(models, ranks):
            rank_samples[(key[0], model)].append(float(rank))
            rank_samples[("all", model)].append(float(rank))
    result = []
    for (dataset, model), values in sorted(rank_samples.items()):
        result.append({
            "dataset": dataset,
            "model": model,
            "conditions": len(values),
            "average_rank": statistics.mean(values),
        })
    return result


def target_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        clean_metrics = row["diagnostics"].get("clean_target_metrics", {})
        clean_micro = clean_metrics.get("micro_f1")
        if clean_micro is None:
            continue
        key = (
            row["protocol"], row["dataset"], row["model"], row["attack"],
            float(row["rate"]),
        )
        groups[key].append((float(clean_micro), row["micro_f1"]))
    result = []
    for key, values in sorted(groups.items()):
        clean = [value[0] for value in values]
        attacked = [value[1] for value in values]
        drops = [100 * (left - right) for left, right in values]
        result.append({
            "protocol": key[0],
            "dataset": key[1],
            "model": key[2],
            "attack": key[3],
            "rate": key[4],
            "n": len(values),
            "clean_micro_f1_mean": statistics.mean(clean),
            "clean_micro_f1_std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
            "micro_f1_mean": statistics.mean(attacked),
            "micro_f1_std": statistics.stdev(attacked) if len(attacked) > 1 else 0.0,
            "drop_pp_mean": statistics.mean(drops),
            "drop_pp_std": statistics.stdev(drops) if len(drops) > 1 else 0.0,
        })
    return result


def holm_adjust(p_values):
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 0.0
    for position, index in enumerate(order):
        candidate = min(1.0, (count - position) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _wilcoxon_pvalue(differences):
    if all(abs(value) <= 1e-12 for value in differences):
        return 1.0
    return float(wilcoxon(differences, alternative="two-sided").pvalue)
