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

from scripts.analyze_adaptive_pilot import load_run_suite, spec_from_command
from scripts.analyze_dvcl_view_diagnosis import (
    _completed_payload,
    _manifest_summary,
    write_csv,
)
from dvcl_bench.paper_analysis import load_protocol_rows
from dvcl_bench.paths import ExperimentLayout


VARIANTS = ("full", "no_cl", "topology_only", "feature_only")
VARIANT_LABELS = {
    "full": "Full DVCL",
    "no_cl": "w/o Cross-view CL",
    "topology_only": "w/o Feature View",
    "feature_only": "w/o Topology View",
}
CONDITIONS = (
    ("clean", 0),
    ("prbcd", 5),
    ("prbcd", 15),
    ("prbcd", 25),
    ("heteprbcd", 5),
    ("heteprbcd", 15),
    ("heteprbcd", 25),
)
TRAIN_SEEDS = (1, 2, 3, 4, 5)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit and aggregate the formal DBLP DVCL ablation matrix."
    )
    parser.add_argument(
        "--config",
        default="configs/suites/dblp_poisoning_ablation_v1.yaml",
    )
    parser.add_argument(
        "--baseline-protocol", default="dblp_poisoning_main_v1"
    )
    parser.add_argument(
        "--output-root",
        default="outputs/analysis/dblp_poisoning_ablation_v1",
    )
    parser.add_argument(
        "--report", default="docs/dblp-ablation-results.md"
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def collect_rows(config_path):
    run_suite = load_run_suite()
    config = run_suite.load_config(config_path)
    commands = list(run_suite.commands(config, sys.executable, ROOT))
    rows = []
    issues = []
    manifests = []
    hash_cache = {}
    completed = 0
    for command in commands:
        spec = spec_from_command(command)
        run_dir = ExperimentLayout(ROOT).run_dir(spec)
        payload = _completed_payload(
            run_dir, spec, command, issues, manifests, hash_cache
        )
        if payload is None:
            continue
        completed += 1
        absent = [
            name for name in ("history.csv", "checkpoint.pt")
            if not (run_dir / name).is_file()
        ]
        if absent:
            issues.append(f"missing run artifacts {absent}: {run_dir}")
            continue
        micro_f1 = float(payload["metrics"]["micro_f1"])
        accuracy = float(payload["metrics"]["accuracy"])
        if abs(micro_f1 - accuracy) > 1e-12:
            issues.append(f"accuracy and Micro-F1 differ: {run_dir}")
            continue
        rows.append({
            "variant": spec.model.config["variant"],
            "attack": spec.attack.name,
            "rate": int(spec.attack.rate),
            "train_seed": spec.seeds.train,
            "micro_f1": micro_f1,
            "best_epoch": int(payload["best_epoch"]),
            "run_dir": str(run_dir.resolve()),
        })
    return rows, issues, len(commands), completed, manifests


def validate_matrix(rows):
    expected = {
        (variant, attack, rate, seed)
        for variant in VARIANTS
        for attack, rate in CONDITIONS
        for seed in TRAIN_SEEDS
    }
    actual = {
        (row["variant"], row["attack"], row["rate"], row["train_seed"])
        for row in rows
    }
    issues = []
    if missing := sorted(expected - actual):
        issues.append(f"missing matrix rows: {missing[:10]} count={len(missing)}")
    if unexpected := sorted(actual - expected):
        issues.append(
            f"unexpected matrix rows: {unexpected[:10]} count={len(unexpected)}"
        )
    if len(actual) != len(rows):
        issues.append(f"duplicate matrix rows: rows={len(rows)} unique={len(actual)}")
    return issues


def validate_full_equivalence(rows, baseline_protocol):
    baseline_rows = load_protocol_rows(
        ROOT / "outputs" / "runs", {baseline_protocol: None}
    )
    baseline = {
        (row["attack"], int(row["rate"]), int(row["train_seed"])):
        float(row["micro_f1"])
        for row in baseline_rows
        if row["dataset"] == "dblp"
        and row["model"] == "dvcl"
        and row["variant"] == "default"
    }
    comparisons = []
    issues = []
    for row in rows:
        if row["variant"] != "full":
            continue
        key = (row["attack"], row["rate"], row["train_seed"])
        if key not in baseline:
            issues.append(f"missing main-protocol baseline: {key}")
            continue
        difference = float(row["micro_f1"]) - baseline[key]
        comparisons.append({
            "attack": key[0],
            "rate": key[1],
            "train_seed": key[2],
            "ablation_micro_f1": float(row["micro_f1"]),
            "main_micro_f1": baseline[key],
            "difference": difference,
        })
        if abs(difference) > 1e-12:
            issues.append(f"full DVCL mismatch {key}: difference={difference}")
    if len(comparisons) != len(CONDITIONS) * len(TRAIN_SEEDS):
        issues.append(
            "full equivalence coverage mismatch: "
            f"{len(comparisons)}/{len(CONDITIONS) * len(TRAIN_SEEDS)}"
        )
    return comparisons, issues


def validate_feature_invariance(rows):
    selected = [row for row in rows if row["variant"] == "feature_only"]
    clean = {
        row["train_seed"]: float(row["micro_f1"])
        for row in selected if row["attack"] == "clean"
    }
    issues = []
    for row in selected:
        if row["train_seed"] not in clean:
            issues.append(f"missing feature-only clean seed {row['train_seed']}")
            continue
        difference = float(row["micro_f1"]) - clean[row["train_seed"]]
        if abs(difference) > 1e-12:
            issues.append(
                "feature-only structural invariance failed: "
                f"{row['attack']} rate={row['rate']} seed={row['train_seed']} "
                f"difference={difference}"
            )
    return issues


def family_summary(rows):
    by_variant_seed_family = defaultdict(list)
    for row in rows:
        families = [row["attack"] if row["attack"] != "clean" else "clean"]
        if row["attack"] != "clean":
            families.append("all")
        for family in families:
            by_variant_seed_family[
                (row["variant"], row["train_seed"], family)
            ].append(float(row["micro_f1"]))
    grouped = defaultdict(list)
    for (variant, _, family), values in by_variant_seed_family.items():
        grouped[(variant, family)].append(statistics.fmean(values))
    result = []
    for variant in VARIANTS:
        for family in ("clean", "prbcd", "heteprbcd", "all"):
            values = grouped.get((variant, family), [])
            result.append({
                "variant": variant,
                "family": family,
                "n": len(values),
                "micro_f1_mean": statistics.fmean(values) if values else None,
                "micro_f1_std": (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                ),
            })
    return result


def condition_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["attack"], row["rate"])].append(
            float(row["micro_f1"])
        )
    result = []
    for key, values in sorted(grouped.items()):
        result.append({
            "variant": key[0],
            "attack": key[1],
            "rate": key[2],
            "n": len(values),
            "micro_f1_mean": statistics.fmean(values),
            "micro_f1_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    return result


def paired_effects(rows):
    values = {
        (row["variant"], row["attack"], row["rate"], row["train_seed"]):
        float(row["micro_f1"])
        for row in rows
    }
    result = []
    for variant in VARIANTS[1:]:
        for family, attacks in (
            ("clean", {"clean"}),
            ("prbcd", {"prbcd"}),
            ("heteprbcd", {"heteprbcd"}),
            ("all", {"prbcd", "heteprbcd"}),
        ):
            differences = []
            for seed in TRAIN_SEEDS:
                per_condition = []
                for attack, rate in CONDITIONS:
                    if attack not in attacks:
                        continue
                    full = values[("full", attack, rate, seed)]
                    ablated = values[(variant, attack, rate, seed)]
                    per_condition.append(full - ablated)
                differences.append(statistics.fmean(per_condition))
            result.append({
                "variant": variant,
                "family": family,
                "n": len(differences),
                "full_gain_pp_mean": 100 * statistics.fmean(differences),
                "full_gain_pp_std": 100 * statistics.stdev(differences),
                "wins": sum(value > 1e-12 for value in differences),
                "ties": sum(abs(value) <= 1e-12 for value in differences),
                "losses": sum(value < -1e-12 for value in differences),
            })
    return result


def render_report(families, effects, audit):
    lookup = {
        (row["variant"], row["family"]): row for row in families
    }
    effect_lookup = {
        (row["variant"], row["family"]): row for row in effects
    }
    lines = [
        "# DBLP 组件消融结果",
        "",
        "## 实验设置",
        "",
        "- 数据集：DBLP；划分与攻击种子 $(s_s,s_a)=(1,1)$，训练种子 $s_t=1,\ldots,5$。",
        "- 攻击：PRBCD、HetePRBCD，$r\in\{5\%,15\%,25\%\}$；均为全局 poisoning。",
        "- 变体：Full DVCL、w/o Cross-view CL、w/o Feature View、w/o Topology View。",
        "- 训练：$E_{max}=200$，$P=100$；仅报告 Micro-F1。",
        "- 攻击平均先在每个训练种子内跨扰动率平均，再计算五种子的均值与标准差。",
        "",
        "## 实验结果",
        "",
        "| Variant | Clean | PRBCD Avg. | HetePRBCD Avg. | Attack Avg. |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        cells = [
            _score(lookup[(variant, family)])
            for family in ("clean", "prbcd", "heteprbcd", "all")
        ]
        lines.append(f"| {VARIANT_LABELS[variant]} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## 配对贡献",
        "",
        "正值表示 Full DVCL 优于对应消融；统计单元为五个训练种子。",
        "",
        "| Ablation | Clean | PRBCD Avg. | HetePRBCD Avg. | Attack Avg. |",
        "|---|---:|---:|---:|---:|",
    ])
    for variant in VARIANTS[1:]:
        cells = [
            _effect(effect_lookup[(variant, family)])
            for family in ("clean", "prbcd", "heteprbcd", "all")
        ]
        lines.append(f"| {VARIANT_LABELS[variant]} | " + " | ".join(cells) + " |")
    full_all = lookup[("full", "all")]
    no_cl = effect_lookup[("no_cl", "all")]
    no_feature = effect_lookup[("topology_only", "all")]
    no_topology = effect_lookup[("feature_only", "all")]
    lines.extend([
        "",
        "## 分析",
        "",
        f"- Full DVCL 的 Attack Average 为 {_percent(full_all['micro_f1_mean'])}。",
        f"- 移除跨视图对比学习后平均下降 {no_cl['full_gain_pp_mean']:.2f} pp；移除特征视图后下降 {no_feature['full_gain_pp_mean']:.2f} pp。",
        f"- 移除拓扑视图后的配对差异为 {no_topology['full_gain_pp_mean']:.2f} pp；该变体在纯结构攻击下逐种子保持不变，符合特征视图不读取攻击图的实现语义。",
        f"- Full DVCL 与 DBLP 主实验逐条件逐种子一致：{audit['full_equivalence_rows']}/{len(CONDITIONS) * len(TRAIN_SEEDS)}。",
        f"- 完整性审计：{audit['completed_runs']}/{audit['expected_runs']}，问题数 {len(audit['issues'])}；manifest commit 为 `{audit['manifest_git_commits'][0]}`。",
        "",
    ])
    return "\n".join(lines)


def _score(row):
    return (
        f"{_percent(row['micro_f1_mean'])} ± "
        f"{100 * float(row['micro_f1_std']):.2f}"
    )


def _effect(row):
    return (
        f"{float(row['full_gain_pp_mean']):.2f} ± "
        f"{float(row['full_gain_pp_std']):.2f}"
    )


def _percent(value):
    return f"{100 * float(value):.2f}"


def main():
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    output_root = (ROOT / args.output_root).resolve()
    rows, issues, expected, completed, manifests = collect_rows(config_path)
    issues.extend(validate_matrix(rows))
    equivalence, equivalence_issues = validate_full_equivalence(
        rows, args.baseline_protocol
    )
    issues.extend(equivalence_issues)
    issues.extend(validate_feature_invariance(rows))
    manifest_summary = _manifest_summary(manifests)
    if manifest_summary["dirty_manifests"] and not args.allow_dirty:
        issues.append(
            f"formal ablation contains {manifest_summary['dirty_manifests']} dirty manifests"
        )
    if len(manifest_summary["manifest_git_commits"]) != 1:
        issues.append(
            "formal ablation must use exactly one Git commit: "
            f"{manifest_summary['manifest_git_commits']}"
        )
    families = family_summary(rows) if len(rows) == expected else []
    conditions = condition_summary(rows)
    effects = paired_effects(rows) if len(rows) == expected else []
    complete = completed == expected and len(rows) == expected and not issues
    audit = {
        "config": str(config_path),
        "expected_runs": expected,
        "completed_runs": completed,
        "result_rows": len(rows),
        "full_equivalence_rows": len(equivalence),
        **manifest_summary,
        "issues": issues,
        "ok": complete,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "runs.csv", rows)
    write_csv(output_root / "condition_summary.csv", conditions)
    write_csv(output_root / "family_summary.csv", families)
    write_csv(output_root / "paired_effects.csv", effects)
    write_csv(output_root / "full_equivalence.csv", equivalence)
    (output_root / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if complete:
        report_path = (ROOT / args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            render_report(families, effects, audit), encoding="utf-8"
        )
    print(
        f"runs={completed}/{expected} rows={len(rows)} "
        f"equivalence={len(equivalence)} issues={len(issues)}"
    )
    return 0 if complete or args.allow_partial else 1


if __name__ == "__main__":
    raise SystemExit(main())
