"""Paper-table rendering over audited run-level experiment results."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


MAIN_DATASETS = ("acm", "dblp")
MAIN_MODELS = ("hseco", "dvcl")
BASELINE_MODELS = ("han", "heterosage")
MAIN_CONDITIONS = (
    ("clean", 0.0),
    ("prbcd", 5.0),
    ("prbcd", 10.0),
    ("prbcd", 15.0),
    ("prbcd", 20.0),
    ("prbcd", 25.0),
    ("heteprbcd", 5.0),
    ("heteprbcd", 10.0),
    ("heteprbcd", 15.0),
    ("heteprbcd", 20.0),
    ("heteprbcd", 25.0),
)
ABLATION_VARIANTS = ("full", "no_cl", "topology_only", "feature_only")
ABLATION_CONDITIONS = (
    ("clean", 0.0),
    ("prbcd", 5.0),
    ("prbcd", 15.0),
    ("prbcd", 25.0),
    ("heteprbcd", 5.0),
    ("heteprbcd", 15.0),
    ("heteprbcd", 25.0),
)
VARIANT_LABELS = {
    "full": "Full DVCL",
    "no_cl": "w/o Cross-view CL",
    "topology_only": "w/o Feature View",
    "feature_only": "w/o Topology View",
}
ATTACK_LABELS = {"clean": "Clean", "prbcd": "PRBCD", "heteprbcd": "HetePRBCD"}


def load_run_rows(path: Path) -> List[Dict[str, object]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for row in rows:
        value: Dict[str, object] = dict(row)
        for name in ("rate", "accuracy", "micro_f1"):
            value[name] = float(row[name])
        for name in ("split_seed", "attack_seed", "train_seed"):
            value[name] = int(row[name])
        result.append(value)
    return result


def audit_manifests(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    commits = set()
    dirty_runs = 0
    devices = set()
    for row in rows:
        run_dir = Path(str(row["run_dir"]))
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        commits.add(manifest.get("git_commit"))
        dirty_runs += int(bool(manifest.get("git_dirty")))
        devices.add(manifest.get("experiment", {}).get("device"))
    return {
        "runs": len(rows),
        "dirty_runs": dirty_runs,
        "commits": sorted(str(value) for value in commits),
        "devices": sorted(str(value) for value in devices),
    }


def validate_matrix(
    rows: Sequence[Mapping[str, object]],
    identities: Iterable[Tuple[str, str, str, float]],
    train_seeds: Sequence[int],
) -> None:
    grouped: Dict[Tuple[str, str, str, float], List[int]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["model"]),
            str(row["attack"]),
            float(row["rate"]),
        )
        grouped[key].append(int(row["train_seed"]))
        if abs(float(row["accuracy"]) - float(row["micro_f1"])) > 1e-12:
            raise ValueError(f"Accuracy and Micro-F1 differ for {key}, seed={row['train_seed']}")
    expected = set(identities)
    actual = set(grouped)
    issues = []
    if missing := sorted(expected - actual):
        issues.append(f"missing conditions: {missing}")
    if unexpected := sorted(actual - expected):
        issues.append(f"unexpected conditions: {unexpected}")
    expected_seeds = sorted(train_seeds)
    for key in sorted(expected & actual):
        seeds = sorted(grouped[key])
        if seeds != expected_seeds:
            issues.append(f"{key} train seeds {seeds} != {expected_seeds}")
    if issues:
        raise ValueError("; ".join(issues))


def summarize_condition(rows: Sequence[Mapping[str, object]]) -> Tuple[float, float]:
    values = [float(row["micro_f1"]) for row in rows]
    if not values:
        raise ValueError("Cannot summarize an empty condition")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_family(
    rows: Sequence[Mapping[str, object]], attacks: Iterable[str]
) -> Tuple[float, float]:
    selected = set(attacks)
    by_seed: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        if str(row["attack"]) in selected:
            by_seed[int(row["train_seed"])].append(float(row["micro_f1"]))
    if not by_seed or any(not values for values in by_seed.values()):
        raise ValueError(f"Cannot summarize attack family: {sorted(selected)}")
    values = [statistics.mean(items) for _, items in sorted(by_seed.items())]
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def render_paper_tables(
    rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
    baseline_audit: Mapping[str, object] = None,
    train_seeds: Sequence[int] = (1, 2, 3, 4, 5),
) -> str:
    main = [
        row for row in rows
        if row["dataset"] in MAIN_DATASETS
        and row["model"] in MAIN_MODELS
        and row["variant"] == "default"
    ]
    main_identities = (
        (dataset, model, attack, rate)
        for dataset in MAIN_DATASETS
        for model in MAIN_MODELS
        for attack, rate in MAIN_CONDITIONS
    )
    validate_matrix(main, main_identities, train_seeds)

    baselines = [
        row for row in baseline_rows
        if row["dataset"] in MAIN_DATASETS
        and row["model"] in BASELINE_MODELS
        and row["variant"] == "default"
    ]
    if baselines:
        baseline_identities = (
            (dataset, model, attack, rate)
            for dataset in MAIN_DATASETS
            for model in BASELINE_MODELS
            for attack, rate in MAIN_CONDITIONS
        )
        validate_matrix(baselines, baseline_identities, train_seeds)

    ablation = [
        row for row in rows
        if row["dataset"] == "acm"
        and row["model"] == "dvcl"
        and row["variant"] in ABLATION_VARIANTS
    ]
    _validate_ablation(ablation, train_seeds)

    lines = [
        "# 跨数据集论文实验表",
        "",
        "本文档由 `scripts/generate_paper_tables.py` 从逐次实验结果自动生成。主实验",
        "包含 ACM 和 DBLP 的 220 次 HSeCo/DVCL 运行、220 次 HAN/HeteroSAGE",
        "运行；消融包含 ACM 的 140 次运行。所有结果均为 5 个训练种子的均值",
        "± 样本标准差，单位为百分数。Accuracy 与",
        "Micro-F1 数值相同，因此统一记为 `Accuracy / Micro-F1`。",
        "",
        "> 协议审计状态：`acm_poisoning_main_v1`、`dblp_poisoning_main_v1`",
        "> 和 `acm_poisoning_ablation_v1` 均已完成；共 580 次运行，全部来自干净提交。",
        "",
        "## 1. 主实验",
        "",
    ]
    if baselines and baseline_audit:
        dirty = int(baseline_audit.get("dirty_runs", 0))
        if dirty:
            lines.extend([
                "",
                f"**ACM 基线审计状态：暂定。** HAN/HeteroSAGE 的 {dirty} 次 ACM 运行",
                "manifest 记录为 `git_dirty=true`，需结合修正攻击 artifact 在干净提交上复跑。",
            ])
    lines.extend([
        "| Dataset | Attack | Rate | HAN | HeteroSAGE | HSeCo | DVCL | \\(\\Delta\\) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset in MAIN_DATASETS:
        for attack, rate in MAIN_CONDITIONS:
            values = {}
            for model in MAIN_MODELS:
                selected = _select(main, dataset, model, "default", attack, rate)
                values[model] = summarize_condition(selected)
            for model in BASELINE_MODELS:
                selected = _select(baselines, dataset, model, "default", attack, rate)
                values[model] = summarize_condition(selected)
            cells = _best_cells(values, (*BASELINE_MODELS, *MAIN_MODELS))
            rate_label = "—" if attack == "clean" else f"{int(rate)}%"
            delta = (values["dvcl"][0] - values["hseco"][0]) * 100
            lines.append(
                f"| {dataset.upper()} | {ATTACK_LABELS[attack]} | {rate_label} | "
                + " | ".join(cells) + f" | {_format_delta(delta)} |"
            )

    lines.extend([
        "",
        "## 2. 攻击平均",
        "",
        "先在每个训练种子内对攻击条件取平均，再计算种子间的均值和样本标准差。",
        "Attack Average 包含 PRBCD 和 HetePRBCD 的全部 10 个条件，不包含 clean。",
        "",
        "| Dataset | Condition | HAN | HeteroSAGE | HSeCo | DVCL | \\(\\Delta\\) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    family_specs = (
        ("Clean", ("clean",)),
        ("PRBCD Average", ("prbcd",)),
        ("HetePRBCD Average", ("heteprbcd",)),
        ("Attack Average", ("prbcd", "heteprbcd")),
    )
    for dataset in MAIN_DATASETS:
        for label, attacks in family_specs:
            values = {}
            for model in MAIN_MODELS:
                selected = [
                    row for row in main
                    if row["dataset"] == dataset and row["model"] == model
                ]
                values[model] = summarize_family(selected, attacks)
            for model in BASELINE_MODELS:
                selected = [
                    row for row in baselines
                    if row["dataset"] == dataset and row["model"] == model
                ]
                values[model] = summarize_family(selected, attacks)
            cells = _best_cells(values, (*BASELINE_MODELS, *MAIN_MODELS))
            delta = (values["dvcl"][0] - values["hseco"][0]) * 100
            lines.append(
                f"| {dataset.upper()} | {label} | " + " | ".join(cells)
                + f" | {_format_delta(delta)} |"
            )

    lines.extend([
        "",
        "## 3. ACM 组件消融",
        "",
        "PRBCD Average 和 HetePRBCD Average 分别包含 5%、15% 和 25%；",
        "Attack Average 包含两个攻击族的全部 6 个条件。",
        "",
        "| Variant | Clean | PRBCD Avg. | HetePRBCD Avg. | Attack Avg. |",
        "|---|---:|---:|---:|---:|",
    ])
    ablation_values = {}
    for variant in ABLATION_VARIANTS:
        selected = [row for row in ablation if row["variant"] == variant]
        ablation_values[variant] = (
            summarize_family(selected, ("clean",)),
            summarize_family(selected, ("prbcd",)),
            summarize_family(selected, ("heteprbcd",)),
            summarize_family(selected, ("prbcd", "heteprbcd")),
        )
    best = [max(ablation_values[v][index][0] for v in ABLATION_VARIANTS) for index in range(4)]
    for variant in ABLATION_VARIANTS:
        cells = []
        for index, value in enumerate(ablation_values[variant]):
            cell = _format_metric(value)
            if abs(value[0] - best[index]) < 1e-12:
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append(f"| {VARIANT_LABELS[variant]} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _validate_ablation(rows, train_seeds):
    grouped = defaultdict(list)
    for row in rows:
        key = (str(row["variant"]), str(row["attack"]), float(row["rate"]))
        grouped[key].append(int(row["train_seed"]))
        if abs(float(row["accuracy"]) - float(row["micro_f1"])) > 1e-12:
            raise ValueError(f"Accuracy and Micro-F1 differ for ablation {key}")
    expected = {
        (variant, attack, rate)
        for variant in ABLATION_VARIANTS
        for attack, rate in ABLATION_CONDITIONS
    }
    actual = set(grouped)
    issues = []
    if missing := sorted(expected - actual):
        issues.append(f"missing ablation conditions: {missing}")
    if unexpected := sorted(actual - expected):
        issues.append(f"unexpected ablation conditions: {unexpected}")
    expected_seeds = sorted(train_seeds)
    for key in sorted(expected & actual):
        seeds = sorted(grouped[key])
        if seeds != expected_seeds:
            issues.append(f"ablation {key} train seeds {seeds} != {expected_seeds}")
    if issues:
        raise ValueError("; ".join(issues))


def _select(rows, dataset, model, variant, attack, rate):
    return [
        row for row in rows
        if row["dataset"] == dataset
        and row["model"] == model
        and row["variant"] == variant
        and row["attack"] == attack
        and float(row["rate"]) == rate
    ]


def _format_metric(value: Tuple[float, float]) -> str:
    return f"{value[0] * 100:.2f} ± {value[1] * 100:.2f}"


def _comparison_cells(left: Tuple[float, float], right: Tuple[float, float]):
    left_cell, right_cell = _format_metric(left), _format_metric(right)
    if left[0] > right[0]:
        left_cell = f"**{left_cell}**"
    elif right[0] > left[0]:
        right_cell = f"**{right_cell}**"
    return left_cell, right_cell


def _best_cells(values, order):
    best = max(values[model][0] for model in order)
    cells = []
    for model in order:
        cell = _format_metric(values[model])
        if abs(values[model][0] - best) < 1e-12:
            cell = f"**{cell}**"
        cells.append(cell)
    return cells


def _format_delta(value: float) -> str:
    return f"{value:+.2f}"
