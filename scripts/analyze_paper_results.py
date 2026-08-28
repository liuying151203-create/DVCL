import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.paper_analysis import (
    average_ranks,
    family_averages,
    load_protocol_rows,
    paired_significance,
    summarize,
    target_summary,
)
from dvcl_bench.artifacts import load_attack_artifact
from scripts.analyze_adaptive_pilot import (
    adaptive_average_ranks,
    aggregate_rows as aggregate_adaptive_rows,
    collect_rows as collect_adaptive_rows,
    paired_reference_comparisons,
)


MULTI_SEED_PROTOCOL = "acm_dblp_attack_seed_recheck_v1"
AMINER_PROTOCOLS = (
    "aminer_poisoning_main_v1",
    "aminer_rnd_poisoning_v1",
    "aminer_hg_baseline_target_evasion_v1",
)
POISONING_PROTOCOLS = (
    "acm_poisoning_main_v1",
    "dblp_poisoning_main_v1",
    "robust_baselines_poisoning_v1",
    "openhgnn_baselines_poisoning_v1",
    "aminer_poisoning_main_v1",
)
RND_PROTOCOLS = ("rnd_poisoning_v1", "aminer_rnd_poisoning_v1")
TARGET_PROTOCOLS = (
    "hg_baseline_target_evasion_v1",
    "aminer_hg_baseline_target_evasion_v1",
    "dvcl_adaptive_target_evasion_v1",
)
CLEAN_PROTOCOLS = ("acm_poisoning_main_v1", "dblp_poisoning_main_v1")
ABLATION_PROTOCOL = "acm_poisoning_ablation_v1"
ADAPTIVE_PROTOCOL = "adaptive_target_evasion_v1"
ADAPTIVE_CONFIG = ROOT / "configs" / "protocols" / f"{ADAPTIVE_PROTOCOL}.yaml"
EXPECTED_PROTOCOL_RUNS = {
    "acm_poisoning_main_v1": 220,
    "dblp_poisoning_main_v1": 220,
    "robust_baselines_poisoning_v1": 330,
    "openhgnn_baselines_poisoning_v1": 440,
    "rnd_poisoning_v1": 550,
    "aminer_poisoning_main_v1": 605,
    "aminer_rnd_poisoning_v1": 275,
    "aminer_hg_baseline_target_evasion_v1": 165,
    "hg_baseline_target_evasion_v1": 330,
    "acm_dblp_attack_seed_recheck_v1": 720,
    "dvcl_adaptive_target_evasion_v1": 30,
    "adaptive_target_evasion_v1": 99,
    "acm_poisoning_ablation_v1": 140,
}
MODEL_ORDER = (
    "han", "heterosage", "rohe", "heteroguard", "fastrohgcn", "hgt",
    "magnn", "heco", "simplehgn", "hseco", "dvcl",
)
MODEL_LABELS = {
    "han": "HAN",
    "heterosage": "HeteroSAGE",
    "rohe": "RoHe",
    "heteroguard": "HeteroGuard",
    "fastrohgcn": "FastRoHGCN",
    "hgt": "HGT",
    "magnn": "MAGNN",
    "heco": "HeCo",
    "simplehgn": "SimpleHGN",
    "hseco": "HSeCo",
    "dvcl": "DVCL",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Micro-F1 paper statistics, tables, and figures."
    )
    parser.add_argument("--run-root", default=str(ROOT / "outputs" / "runs"))
    parser.add_argument(
        "--output-dir", default=str(ROOT / "outputs" / "paper_analysis")
    )
    parser.add_argument("--docs-dir", default=str(ROOT / "docs"))
    parser.add_argument(
        "--figure-dir", default=str(ROOT / "docs" / "figures" / "paper")
    )
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    docs_dir = Path(args.docs_dir)
    protocols = tuple(EXPECTED_PROTOCOL_RUNS)
    rows = load_protocol_rows(run_root, dict.fromkeys(protocols))
    protocol_counts = _validate_protocol_counts(rows)
    multi_rows = [row for row in rows if row["protocol"] == MULTI_SEED_PROTOCOL]
    target_rows = [row for row in rows if row["protocol"] in TARGET_PROTOCOLS]
    benchmark_rows = [
        row for row in rows
        if row["protocol"] in (*POISONING_PROTOCOLS, *RND_PROTOCOLS)
    ]

    benchmark_summary = summarize(
        benchmark_rows, ("dataset", "model", "attack", "rate")
    )
    benchmark_family = _benchmark_family_rows(rows)
    _validate_benchmark_coverage(benchmark_summary, benchmark_family)
    multi_summary = summarize(
        multi_rows, ("dataset", "model", "attack", "rate")
    )
    multi_family = _multi_seed_family_rows(multi_rows, rows)
    significance = paired_significance(multi_rows)
    ranks = average_ranks(multi_rows)
    targets = target_summary(target_rows)
    aminer = _aminer_family_rows(rows, targets)
    ablation = _ablation_rows(rows)
    adaptive_budget = _adaptive_budget_rows(run_root)
    adaptive_rows, adaptive_issues, adaptive_expected, adaptive_completed = (
        collect_adaptive_rows(ADAPTIVE_CONFIG)
    )
    if adaptive_issues or adaptive_expected != 99 or adaptive_completed != 99:
        raise ValueError(
            "Incomplete formal adaptive matrix: "
            f"expected={adaptive_expected} completed={adaptive_completed} "
            f"issues={adaptive_issues}"
        )
    adaptive_summary = aggregate_adaptive_rows(adaptive_rows)
    adaptive_significance = paired_reference_comparisons(adaptive_rows)
    adaptive_ranks = adaptive_average_ranks(adaptive_rows)
    aminer_audit = _aminer_attack_audit(aminer)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "benchmark_summary.csv", benchmark_summary)
    _write_csv(output_dir / "benchmark_family_summary.csv", benchmark_family)
    _write_csv(output_dir / "multi_seed_summary.csv", multi_summary)
    _write_csv(output_dir / "multi_seed_family_summary.csv", multi_family)
    _write_csv(output_dir / "significance.csv", significance)
    _write_csv(output_dir / "average_ranks.csv", ranks)
    _write_csv(output_dir / "target_evasion_summary.csv", targets)
    _write_csv(output_dir / "aminer_family_summary.csv", aminer)
    _write_csv(output_dir / "ablation_summary.csv", ablation)
    _write_csv(output_dir / "adaptive_budget_summary.csv", adaptive_budget)
    _write_csv(output_dir / "adaptive_target_summary.csv", adaptive_summary)
    _write_csv(output_dir / "adaptive_target_significance.csv", adaptive_significance)
    _write_csv(output_dir / "adaptive_target_ranks.csv", adaptive_ranks)
    _write_csv(output_dir / "aminer_attack_audit.csv", aminer_audit)
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps({
            "protocols": list(dict.fromkeys(protocols)),
            "metrics": ["micro_f1"],
            "protocol_runs": {
                row["protocol"]: row["completed"] for row in protocol_counts
            },
            "benchmark_runs": len(benchmark_rows),
            "multi_seed_runs": len(multi_rows),
            "target_runs": len(target_rows),
            "adaptive_physical_runs": adaptive_completed,
            "adaptive_logical_results": len(adaptive_rows),
            "significance_test": "paired two-sided Wilcoxon signed-rank",
            "multiple_testing": "Holm family-wise correction",
            "adaptive_effect_ci": "paired mean Student-t 95% confidence interval",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "final-experiment-results.md").write_text(
        _final_document(
            benchmark_summary, benchmark_family, multi_family, significance,
            ranks, targets, ablation, adaptive_summary, adaptive_significance,
            adaptive_ranks, aminer_audit, protocol_counts,
        ),
        encoding="utf-8",
    )
    (docs_dir / "aminer-experiment-results.md").write_text(
        _aminer_document(aminer, aminer_audit, benchmark_summary),
        encoding="utf-8",
    )
    (docs_dir / "target-evasion-results.md").write_text(
        _target_document(
            targets, adaptive_summary, adaptive_significance, adaptive_ranks
        ),
        encoding="utf-8",
    )
    if not args.skip_figures:
        _write_figures(
            Path(args.figure_dir), multi_summary, ranks, adaptive_summary
        )
    print(
        f"Wrote paper analysis: multi_seed={len(multi_rows)} "
        f"targets={len(target_rows)} output={output_dir}"
    )
    return 0


def _validate_protocol_counts(rows):
    counts = defaultdict(int)
    for row in rows:
        if row["protocol"] == ADAPTIVE_PROTOCOL and (
            row["attack_seed"] != row["train_seed"]
            or row["attack_seed"] not in (1, 2, 3)
        ):
            continue
        counts[row["protocol"]] += 1
    issues = [
        f"{protocol}: expected={expected}, actual={counts[protocol]}"
        for protocol, expected in EXPECTED_PROTOCOL_RUNS.items()
        if counts[protocol] != expected
    ]
    if issues:
        raise ValueError("Incomplete paper protocols: " + "; ".join(issues))
    return [
        {"protocol": protocol, "expected": expected, "completed": counts[protocol]}
        for protocol, expected in EXPECTED_PROTOCOL_RUNS.items()
    ]


def _benchmark_family_rows(rows):
    result = []
    for protocol in (*POISONING_PROTOCOLS, *RND_PROTOCOLS):
        result.extend(family_averages(rows, protocol))
    return sorted(
        result,
        key=lambda row: (row["dataset"], row["model"], row["attack"]),
    )


def _validate_benchmark_coverage(summary, family):
    issues = []
    for dataset in ("acm", "dblp", "aminer"):
        for model in MODEL_ORDER:
            if _lookup(
                summary, dataset=dataset, model=model, attack="clean", rate=0.0
            ) is None:
                issues.append(f"{dataset}/{model}/clean")
            for attack in ("prbcd", "heteprbcd", "rnd"):
                for rate in (5.0, 10.0, 15.0, 20.0, 25.0):
                    if _lookup(
                        summary, dataset=dataset, model=model,
                        attack=attack, rate=rate,
                    ) is None:
                        issues.append(f"{dataset}/{model}/{attack}/rate_{rate:g}")
                if _lookup(
                    family, dataset=dataset, model=model, attack=attack
                ) is None:
                    issues.append(f"{dataset}/{model}/{attack}/average")
    if issues:
        raise ValueError("Incomplete 11-model benchmark: " + "; ".join(issues))


def _aminer_family_rows(rows, targets):
    poisoning = family_averages(rows, "aminer_poisoning_main_v1")
    rnd = family_averages(rows, "aminer_rnd_poisoning_v1")
    result = [*poisoning, *rnd]
    for row in targets:
        if (
            row["protocol"] == "aminer_hg_baseline_target_evasion_v1"
            and row["rate"] == 5
        ):
            result.append({
                "dataset": "aminer",
                "model": row["model"],
                "attack": "hg_baseline_delta_5",
                "n": row["n"],
                "micro_f1_mean": row["micro_f1_mean"],
                "micro_f1_std": row["micro_f1_std"],
            })
    return result


def _multi_seed_family_rows(multi_rows, all_rows):
    result = family_averages(multi_rows, MULTI_SEED_PROTOCOL)
    per_seed = defaultdict(list)
    for row in multi_rows:
        key = (
            row["dataset"], row["model"], row["attack_seed"], row["train_seed"]
        )
        per_seed[key].append(row["micro_f1"])
    all_groups = defaultdict(list)
    for key, values in per_seed.items():
        all_groups[key[:2]].append(sum(values) / len(values))
    for key, values in sorted(all_groups.items()):
        result.append({
            "dataset": key[0],
            "model": key[1],
            "attack": "all",
            "n": len(values),
            "micro_f1_mean": statistics.mean(values),
            "micro_f1_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    clean_rows = [
        row for row in all_rows
        if row["protocol"] in CLEAN_PROTOCOLS and row["attack"] == "clean"
    ]
    clean = {
        (row["dataset"], row["model"]): row
        for row in summarize(clean_rows, ("dataset", "model"))
    }
    for row in result:
        baseline = clean[(row["dataset"], row["model"])]
        row["clean_micro_f1_mean"] = baseline["micro_f1_mean"]
        row["clean_micro_f1_std"] = baseline["micro_f1_std"]
        row["drop_pp"] = 100 * (
            baseline["micro_f1_mean"] - row["micro_f1_mean"]
        )
    return result


def _ablation_rows(rows):
    selected = [row for row in rows if row["protocol"] == ABLATION_PROTOCOL]
    per_seed = defaultdict(list)
    for row in selected:
        condition = row["attack"] if row["attack"] != "clean" else "clean"
        key = (row["variant"], condition, row["train_seed"])
        per_seed[key].append(row["micro_f1"])
        if row["attack"] != "clean":
            per_seed[(row["variant"], "all", row["train_seed"])].append(
                row["micro_f1"]
            )
    grouped = defaultdict(list)
    for key, values in per_seed.items():
        grouped[key[:2]].append(statistics.mean(values))
    result = []
    for key, values in sorted(grouped.items()):
        result.append({
            "variant": key[0],
            "condition": key[1],
            "n": len(values),
            "micro_f1_mean": statistics.mean(values),
            "micro_f1_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    return result


def _adaptive_budget_rows(run_root):
    groups = defaultdict(list)
    protocol_root = run_root / "dvcl_adaptive_target_evasion_v1"
    for metrics_path in protocol_root.rglob("metrics.json"):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        artifact = load_attack_artifact(metrics_path.with_name("adaptive_attack.pt"))
        counts = [
            len(record.get("added", [])) + len(record.get("deleted", []))
            for record in artifact.target_changes
        ]
        rate = int(metrics["rate"])
        diagnostics = metrics["diagnostics"]["adaptive_attack"]
        groups[(metrics["dataset"], rate)].append({
            "targets": len(counts),
            "changed_targets": sum(value > 0 for value in counts),
            "total_changes": sum(counts),
            "full_budget_targets": sum(value == rate for value in counts),
            "queries": int(diagnostics["queries"]),
        })
    result = []
    for key, values in sorted(groups.items()):
        capacity = values[0]["targets"] * key[1]
        utilizations = [value["total_changes"] / capacity for value in values]
        result.append({
            "dataset": key[0],
            "rate": key[1],
            "n": len(values),
            "targets": values[0]["targets"],
            "changed_targets_mean": statistics.mean(
                value["changed_targets"] for value in values
            ),
            "total_changes_mean": statistics.mean(
                value["total_changes"] for value in values
            ),
            "full_budget_targets_mean": statistics.mean(
                value["full_budget_targets"] for value in values
            ),
            "budget_capacity": capacity,
            "budget_utilization_mean": statistics.mean(utilizations),
            "budget_utilization_std": (
                statistics.stdev(utilizations) if len(utilizations) > 1 else 0.0
            ),
            "queries_mean": statistics.mean(value["queries"] for value in values),
        })
    return result


def _aminer_attack_audit(aminer):
    result = []
    for attack in ("prbcd", "heteprbcd"):
        relations = set()
        train_shares = []
        surrogate_drops = []
        actual_rates = []
        for rate in (5, 10, 15, 20, 25):
            path = (
                ROOT / "data" / "attacks" / "aminer" / attack
                / f"rate_{rate}" / "seed_1" / "attack.pt"
            )
            artifact = load_attack_artifact(path)
            relations.update(
                name for name, stats in artifact.stats.items()
                if name != "_global"
                and stats.get("n_add", 0) + stats.get("n_del", 0) > 0
            )
            actual_rates.append(artifact.stats["_global"]["actual_rate"])
            report = json.loads(
                path.with_name("verification.json").read_text(encoding="utf-8")
            )
            train_shares.append(
                report["split_perturbation"]["_global"]["train"]["change_share"]
            )
            before = artifact.provenance.get("surrogate_before", {}).get("micro_f1")
            after = artifact.provenance.get("surrogate_after", {}).get("micro_f1")
            if before is not None and after is not None:
                surrogate_drops.append(100 * (before - after))
        formal_drops = []
        for model in MODEL_ORDER:
            clean = _lookup(aminer, dataset="aminer", model=model, attack="clean")
            attacked = _lookup(aminer, dataset="aminer", model=model, attack=attack)
            formal_drops.append(
                100 * (clean["micro_f1_mean"] - attacked["micro_f1_mean"])
            )
        result.append({
            "attack": attack,
            "relations": ",".join(sorted(relations)),
            "actual_rate_min": min(actual_rates),
            "actual_rate_max": max(actual_rates),
            "train_change_share_min": min(train_shares),
            "train_change_share_max": max(train_shares),
            "surrogate_drop_pp_min": min(surrogate_drops),
            "surrogate_drop_pp_max": max(surrogate_drops),
            "formal_model_drop_pp_min": min(formal_drops),
            "formal_model_drop_pp_median": statistics.median(formal_drops),
            "formal_model_drop_pp_max": max(formal_drops),
        })
    return result


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _lookup(rows, **conditions):
    for row in rows:
        if all(row.get(key) == value for key, value in conditions.items()):
            return row
    return None


def _score(row):
    return f"{100 * row['micro_f1_mean']:.2f} ± {100 * row['micro_f1_std']:.2f}"


def _final_document(
    benchmark, benchmark_family, multi_family, significance, ranks, targets,
    ablation, adaptive, adaptive_significance, adaptive_ranks, aminer_audit,
    protocol_counts,
):
    completed = sum(row["completed"] for row in protocol_counts)
    lines = [
        "# 最终实验结果与统计分析",
        "",
        "> **唯一论文结果主入口。** 主表统一使用三数据集、十一模型和 Micro-F1；覆盖范围不同的统计复验、自适应攻击与消融实验均独立成节，不与主表混合。",
        "",
        "## 1. 实验设置与覆盖范围",
        "",
        "| 实验族 | 数据集 | 模型范围 | 种子 | 用途 |",
        "|---|---|---|---|---|",
        "| 全局 poisoning 主实验 | ACM、DBLP、AMiner | 统一 11 模型 | $s_{atk}=1,s_{train}=1\\ldots5$ | PRBCD、HetePRBCD、RND 主比较 |",
        "| HG 迁移目标逃逸 | ACM、DBLP、AMiner | 统一 11 模型 | artifact seed 1，$s_{train}=1\\ldots5$ | 测试时固定攻击迁移性 |",
        "| 多攻击种子统计复验 | ACM、DBLP | HAN、HeteroSAGE、HSeCo、DVCL | $s_{atk}=1\\ldots3,s_{train}=1\\ldots5$ | 显著性与攻击种子稳定性 |",
        "| 模型自适应目标逃逸 | ACM、DBLP、AMiner | 统一 11 模型 | $(s_a,s_t)=(1,1),(2,2),(3,3)$ | 每模型独立优化攻击边 |",
        "| 组件消融 | ACM | DVCL 四个 variant | $s_{train}=1\\ldots5$ | 模块贡献 |",
        "",
        "统一训练设置为 $E_{max}=200$、$P=100$ 和完整模型 checkpoint。Poisoning 扰动率为 $r\\in\\{5,10,15,20,25\\}\\%$；目标逃逸预算为 $\\Delta\\in\\{1,3,5\\}$。表格报告均值 ± 样本标准差。",
        "",
        f"完整性检查覆盖 {len(protocol_counts)} 套协议、{completed}/{completed} 次运行，所有主表单元均通过三数据集×十一模型覆盖校验。",
        "",
        "## 2. 三数据集统一十一模型全局 Poisoning",
        "",
        "三张主表使用完全相同的 11 个模型和列顺序。PRBCD/HetePRBCD 展示 5%、15%、25% 代表点，RND Avg. 聚合 5%–25%；10%、20% 明细保留在数据集附录。",
        "",
    ]
    for dataset in ("acm", "dblp", "aminer"):
        lines.extend([f"### {dataset.upper()}", ""])
        lines.extend(_benchmark_table_lines(benchmark, benchmark_family, dataset))
        lines.append("")
    lines.extend([
        "**关系口径：** ACM/DBLP 的三类 poisoning artifact 均围绕 P–A 关系生成；AMiner 的 PRBCD/HetePRBCD 修改 P–A，RND 同时修改 P–A 与 P–R。三数据集结果可以比较模型表现，但不能把不同关系上的同一扰动率解释为完全相同的结构难度。",
        "",
        "## 3. ACM/DBLP 多攻击种子统计复验",
        "",
        "本节不是另一张主基线表，只复验四个核心模型。AMiner 当前只有一个正式 attack seed，因此不进入多攻击种子显著性检验。每个攻击平均单元聚合 3 个攻击种子×5 个训练种子。括号为相对 clean 的下降百分点。",
        "",
    ])
    lines.extend(_multi_seed_extension_lines(multi_family))
    lines.extend([
        "",
        "正效应表示 DVCL 更高；W/T/L 为 15 个配对的胜/平/负次数。",
        "",
        "| Dataset | Baseline | $n$ | Effect (pp) | W/T/L | $p_{Holm}$ |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in significance:
        if row["attack"] != "all":
            continue
        lines.append(
            f"| {row['dataset'].upper()} | {MODEL_LABELS[row['baseline']]} | "
            f"{row['n']} | {row['effect_pp']:+.2f} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | {row['p_holm']:.3g} |"
        )
    lines.extend([
        "",
        "| Scope | HAN | HeteroSAGE | HSeCo | DVCL |",
        "|---|---:|---:|---:|---:|",
    ])
    for dataset in ("acm", "dblp", "all"):
        cells = [
            f"{_lookup(ranks, dataset=dataset, model=model)['average_rank']:.2f}"
            for model in ("han", "heterosage", "hseco", "dvcl")
        ]
        lines.append(f"| {dataset.upper()} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## 4. 目标逃逸攻击",
        "",
        "### 4.1 HG Baseline：统一十一模型迁移攻击",
        "",
        "所有模型都在 clean 图训练，测试时应用同一批固定、非自适应目标 artifact。三数据集均使用相同模型集合和 $\\Delta=1,3,5$。",
        "",
    ])
    for dataset in ("acm", "dblp", "aminer"):
        lines.extend([f"#### {dataset.upper()}", ""])
        lines.extend(_target_table_lines(targets, dataset))
        lines.append("")
    lines.extend([
        "固定 HG artifact 不针对每个被评估模型优化，因此个别负下降表示少量错误预测被扰动纠正，不能解释为自适应鲁棒性。",
        "",
        "### 4.2 统一十一模型自适应攻击",
        "",
        "每个模型与 checkpoint 独立执行 score-based 贪心查询；相同数据集和 seed 共享候选池，但最终攻击边由 victim 模型决定。每目标候选池为 64 条增边和 64 条删边，$\\Delta\\in\\{1,3,5\\}$ 为最大预算。正式矩阵包含 99 次物理搜索和 297 个预算评估；权威审计为 `outputs/analysis/adaptive_target_evasion_v1/audit.json`。",
        "",
    ])
    for dataset in ("acm", "dblp", "aminer"):
        lines.extend([f"#### {dataset.upper()}", ""])
        lines.extend(_formal_adaptive_table(adaptive, dataset))
        lines.append("")
    lines.extend(["**攻击有效性诊断（跨 11 模型均值）**", ""])
    lines.extend(_formal_adaptive_diagnostic_table(adaptive))
    lines.extend(["", "**平均排名与最近竞争基线**", ""])
    lines.extend(_formal_adaptive_rank_table(adaptive_ranks))
    lines.append("")
    lines.extend(_formal_adaptive_key_significance_table(
        adaptive_significance, adaptive_ranks
    ))
    lines.extend([
        "",
        "ACM 和 AMiner 中 DVCL 在当前 64+64 有限候选攻击下未出现目标 Micro-F1 下降，但这只能说明该查询攻击未找到成功扰动，不能证明完整候选空间或白盒攻击下的鲁棒性。DBLP 中 DVCL 在 $\\Delta=5$ 下降 46.00 个百分点，且平均排名低于 HeteroGuard，确认 DVCL 存在数据集依赖的自适应目标逃逸脆弱性。每数据集只有 3 个独立配对重复，Holm 校正后均未达到显著，效应量和置信区间应与排名共同解释。",
        "",
        "## 5. ACM 组件消融",
        "",
        "消融只回答 DVCL 组件贡献，不作为三数据集基线比较。",
        "",
        "| Variant | Clean | PRBCD Avg. | HetePRBCD Avg. | Attack Avg. |",
        "|---|---:|---:|---:|---:|",
    ])
    lines.extend(_ablation_table_lines(ablation))
    lines.extend([
        "",
        "## 6. 异常结果审计与结论边界",
        "",
        "### 6.1 AMiner Poisoning 强度",
        "",
        "| Attack | Relations | Realized $r$ | Train change share | Surrogate drop (pp) | Formal median drop (pp) |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in aminer_audit:
        lines.append(
            f"| {row['attack'].upper()} | {row['relations']} | "
            f"{100 * row['actual_rate_min']:.2f}–{100 * row['actual_rate_max']:.2f}% | "
            f"{100 * row['train_change_share_min']:.1f}–{100 * row['train_change_share_max']:.1f}% | "
            f"{row['surrogate_drop_pp_min']:.2f}–{row['surrogate_drop_pp_max']:.2f} | "
            f"{row['formal_model_drop_pp_median']:+.2f} |"
        )
    lines.extend([
        "",
        "AMiner 的预算与 artifact 验证均正确，但替代模型和正式模型下降偏弱；这属于当前攻击生成器效果不足，不是漏施加预算。因此 AMiner poisoning 只能支持弱攻击条件下的横向比较。",
        "",
        "### 6.2 可支持与不可支持的结论",
        "",
        "1. 三数据集主表可以比较统一 11 模型在相同数据集、相同 artifact 下的 Micro-F1。",
        "2. 多攻击种子复验支持 DVCL 相对 HSeCo 的 ACM/DBLP 总体增益，但不支持 DVCL 在每个数据集、每种攻击上普遍最优。",
        "3. DVCL 在 DBLP PRBCD 平均下低于 HSeCo；ACM 相对 HAN/HeteroSAGE 的多种子差异未达到校正后显著。",
        "4. HG 固定迁移攻击、自适应查询攻击和 poisoning 具有不同语义，禁止合并计算总 Attack Average。",
        "5. 统一 11 模型自适应矩阵已完成；下一步按 topology/feature 视图漂移和预测分歧开展 DVCL 失效诊断，再决定是否改进门控融合。",
        "",
        "## 7. 论文图表",
        "",
        "![ACM HetePRBCD 多种子曲线](figures/paper/acm_heteprbcd_curve.png)",
        "",
        "![DBLP HetePRBCD 多种子曲线](figures/paper/dblp_heteprbcd_curve.png)",
        "",
        "![多种子平均排名](figures/paper/multi_seed_average_rank.png)",
        "",
        "![十一模型自适应目标逃逸](figures/paper/adaptive_target_evasion_11model.png)",
        "",
        "## 8. 专项附录",
        "",
        "- 完整扰动率与数据集明细：`docs/acm-experiment-results.md`、`docs/dblp-experiment-results.md`、`docs/aminer-experiment-results.md`",
        "- 目标逃逸逐模型结果：`docs/target-evasion-results.md`",
        "- 鲁棒/OpenHGNN/RND 基线：`docs/robust-baseline-results.md`、`docs/openhgnn-baseline-results.md`、`docs/rnd-attack-results.md`",
        "- 文档导航与口径优先级：`docs/README.md`",
        "- 当前有效后续实验计划：`docs/next-experiment-plan.md`",
        "",
    ])
    return "\n".join(lines)


def _benchmark_table_lines(summary, family, dataset):
    lines = [
        "| Model | Clean | PRBCD 5% | PRBCD 15% | PRBCD 25% | HetePRBCD 5% | HetePRBCD 15% | HetePRBCD 25% | RND Avg. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        cells = [
            _score(_lookup(
                summary, dataset=dataset, model=model, attack="clean", rate=0.0,
            ))
        ]
        for attack in ("prbcd", "heteprbcd"):
            cells.extend(
                _score(_lookup(
                    summary, dataset=dataset, model=model,
                    attack=attack, rate=rate,
                ))
                for rate in (5.0, 15.0, 25.0)
            )
        cells.append(
            _score(_lookup(
                family, dataset=dataset, model=model, attack="rnd",
            ))
        )
        lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(cells) + " |")
    return lines


def _multi_seed_extension_lines(rows):
    models = ("han", "heterosage", "hseco", "dvcl")
    lines = [
        "| Dataset | Condition | HAN | HeteroSAGE | HSeCo | DVCL |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in ("acm", "dblp"):
        clean_cells = []
        for model in models:
            row = _lookup(rows, dataset=dataset, model=model, attack="all")
            clean_cells.append(
                f"{100 * row['clean_micro_f1_mean']:.2f} ± "
                f"{100 * row['clean_micro_f1_std']:.2f}"
            )
        lines.append(
            f"| {dataset.upper()} | Clean | " + " | ".join(clean_cells) + " |"
        )
        for attack, label in (
            ("prbcd", "PRBCD Avg."),
            ("heteprbcd", "HetePRBCD Avg."),
            ("all", "Attack Avg."),
        ):
            cells = []
            for model in models:
                row = _lookup(rows, dataset=dataset, model=model, attack=attack)
                cells.append(f"{_score(row)} ({row['drop_pp']:+.2f})")
            lines.append(
                f"| {dataset.upper()} | {label} | " + " | ".join(cells) + " |"
            )
    return lines


def _target_table_lines(targets, dataset):
    protocol = (
        "aminer_hg_baseline_target_evasion_v1"
        if dataset == "aminer" else "hg_baseline_target_evasion_v1"
    )
    lines = [
        "| Model | Clean target | $\\Delta=1$ | $\\Delta=3$ | $\\Delta=5$ | Drop@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        rows = [
            _lookup(
                targets, protocol=protocol, dataset=dataset, model=model, rate=rate
            )
            for rate in (1.0, 3.0, 5.0)
        ]
        clean = rows[0]
        lines.append(
            f"| {MODEL_LABELS[model]} | "
            f"{100 * clean['clean_micro_f1_mean']:.2f} ± "
            f"{100 * clean['clean_micro_f1_std']:.2f} | "
            + " | ".join(_score(row) for row in rows)
            + f" | {rows[-1]['drop_pp_mean']:+.2f} |"
        )
    return lines


def _adaptive_table(targets):
    lines = [
        "| Dataset | Clean target | $\\Delta=1$ | $\\Delta=3$ | $\\Delta=5$ | Drop@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("acm", "dblp"):
        rows = [
            _lookup(
                targets,
                protocol="dvcl_adaptive_target_evasion_v1",
                dataset=dataset,
                model="dvcl",
                rate=rate,
            )
            for rate in (1.0, 3.0, 5.0)
        ]
        clean = rows[0]
        lines.append(
            f"| {dataset.upper()} | "
            f"{100 * clean['clean_micro_f1_mean']:.2f} ± {100 * clean['clean_micro_f1_std']:.2f} | "
            + " | ".join(_score(row) for row in rows)
            + f" | {rows[-1]['drop_pp_mean']:+.2f} |"
        )
    return lines


def _adaptive_budget_table(rows):
    lines = [
        "| Dataset | $\\Delta$ | Changed targets | Changes / capacity | Utilization | Full-budget targets |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset'].upper()} | {row['rate']} | "
            f"{row['changed_targets_mean']:.1f}/{row['targets']} | "
            f"{row['total_changes_mean']:.1f}/{row['budget_capacity']} | "
            f"{100 * row['budget_utilization_mean']:.1f}% | "
            f"{row['full_budget_targets_mean']:.1f}/{row['targets']} |"
        )
    return lines


def _adaptive_score(row):
    return (
        f"{100 * row['attacked_target_micro_f1_mean']:.2f} ± "
        f"{100 * row['attacked_target_micro_f1_std']:.2f}"
    )


def _formal_adaptive_table(rows, dataset):
    lines = [
        "| Model | Clean target | $\\Delta=1$ | $\\Delta=3$ | $\\Delta=5$ | Drop@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        selected = [
            _lookup(rows, dataset=dataset, model=model, rate=rate)
            for rate in (1, 3, 5)
        ]
        clean = selected[0]
        lines.append(
            f"| {MODEL_LABELS[model]} | "
            f"{100 * clean['clean_target_micro_f1_mean']:.2f} ± "
            f"{100 * clean['clean_target_micro_f1_std']:.2f} | "
            + " | ".join(_adaptive_score(row) for row in selected)
            + f" | {100 * selected[-1]['micro_f1_drop_mean']:+.2f} |"
        )
    return lines


def _formal_adaptive_diagnostic_table(rows):
    lines = [
        "| Dataset | $\\Delta$ | ASR | Budget utilization | Changes | Queries |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("acm", "dblp", "aminer"):
        for rate in (1, 3, 5):
            selected = [
                row for row in rows
                if row["dataset"] == dataset and row["rate"] == rate
            ]
            lines.append(
                f"| {dataset.upper()} | {rate} | "
                f"{100 * statistics.fmean(row['attack_success_rate_mean'] for row in selected):.2f}% | "
                f"{100 * statistics.fmean(row['budget_utilization_mean'] for row in selected):.2f}% | "
                f"{statistics.fmean(row['total_changes_mean'] for row in selected):.1f} | "
                f"{statistics.fmean(row['queries_mean'] for row in selected):.1f} |"
            )
    return lines


def _formal_adaptive_rank_table(rows):
    lines = [
        "| Dataset | Best model | Best rank | DVCL rank |",
        "|---|---|---:|---:|",
    ]
    for dataset in ("acm", "dblp", "aminer"):
        selected = sorted(
            (row for row in rows if row["dataset"] == dataset),
            key=lambda row: row["average_rank"],
        )
        best = selected[0]
        dvcl = _lookup(rows, dataset=dataset, model="dvcl")
        lines.append(
            f"| {dataset.upper()} | {MODEL_LABELS[best['model']]} | "
            f"{best['average_rank']:.2f} | {dvcl['average_rank']:.2f} |"
        )
    return lines


def _formal_adaptive_key_significance_table(significance, ranks):
    lines = [
        "正效应表示 DVCL 的三预算平均攻击后 Micro-F1 更高；95% CI 为三个配对重复的均值 t 区间。",
        "",
        "| Dataset | Baseline | $n$ | Effect (pp) [95% CI] | W/T/L | $p_{Holm}$ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in ("acm", "dblp", "aminer"):
        competitors = sorted(
            (
                row for row in ranks
                if row["dataset"] == dataset and row["model"] != "dvcl"
            ),
            key=lambda row: row["average_rank"],
        )
        baseline = competitors[0]["model"]
        row = _lookup(significance, dataset=dataset, baseline=baseline)
        lines.append(
            f"| {dataset.upper()} | {MODEL_LABELS[baseline]} | {row['n']} | "
            f"{row['effect_pp']:+.2f} "
            f"[{row['effect_ci_low_pp']:+.2f}, {row['effect_ci_high_pp']:+.2f}] | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | {row['p_holm']:.3g} |"
        )
    return lines


def _formal_adaptive_significance_table(rows):
    lines = [
        "| Dataset | Baseline | $n$ | Effect (pp) [95% CI] | W/T/L | $p_{Holm}$ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    dataset_order = {name: index for index, name in enumerate(("acm", "dblp", "aminer"))}
    model_order = {name: index for index, name in enumerate(MODEL_ORDER)}
    for row in sorted(
        rows,
        key=lambda value: (
            dataset_order[value["dataset"]], model_order[value["baseline"]]
        ),
    ):
        lines.append(
            f"| {row['dataset'].upper()} | {MODEL_LABELS[row['baseline']]} | "
            f"{row['n']} | {row['effect_pp']:+.2f} "
            f"[{row['effect_ci_low_pp']:+.2f}, {row['effect_ci_high_pp']:+.2f}] | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | {row['p_holm']:.3g} |"
        )
    return lines


def _aminer_table_lines(aminer):
    lines = [
        "| Model | Clean | PRBCD Avg. | HetePRBCD Avg. | RND Avg. | HG $\\Delta=5$ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        cells = []
        for attack in (
            "clean", "prbcd", "heteprbcd", "rnd", "hg_baseline_delta_5",
        ):
            row = _lookup(aminer, dataset="aminer", model=model, attack=attack)
            cells.append(_score(row))
        lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(cells) + " |")
    return lines


def _ablation_table_lines(ablation):
    variants = (
        ("full", "Full DVCL"),
        ("no_cl", "w/o Cross-view CL"),
        ("topology_only", "w/o Feature View"),
        ("feature_only", "w/o Topology View"),
    )
    lines = []
    for variant, label in variants:
        cells = []
        for condition in ("clean", "prbcd", "heteprbcd", "all"):
            row = _lookup(ablation, variant=variant, condition=condition)
            cells.append(_score(row))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def _aminer_document(aminer, aminer_audit, benchmark):
    lines = [
        "# AMiner 实验结果",
        "",
        "> **专项附录。** 主结论与统一口径以 `docs/final-experiment-results.md` 为准。本页记录 AMiner 完整逐模型结果，仅报告 Micro-F1。",
        "",
        "## 1. 实验设置",
        "",
        "| 设置 | 取值 |",
        "|---|---|",
        "| 数据划分 | `paper_seed_1` |",
        "| $s_{split},s_{atk}$ | 1, 1 |",
        "| $s_{train}$ | 1–5 |",
        "| Poisoning | PRBCD、HetePRBCD、RND，$r=5,10,15,20,25\\%$ |",
        "| 目标逃逸 | HG Baseline，$\\Delta=1,3,5$ |",
        "| 攻击关系 | PRBCD/HetePRBCD：P–A；RND：P–A + P–R；HG：P–R（均含反向关系） |",
        "| 训练 | $E_{max}=200$，$P=100$ |",
        "",
        "## 2. 超参数",
        "",
        "所有模型采用 `configs/models/` 中冻结配置；HSeCo 使用 $d_s=64,d_n=128,K_s=K_n=8,\\eta=0.005$，DVCL 使用 $d=128,K=4,k=20,\\tau_c=0.5,\\lambda_h=\\lambda_d=1$。",
        "",
        "## 3. 攻击平均结果",
        "",
        "Poisoning 平均值先在每个训练种子内跨 5 个扰动率平均；HG 列报告 $\\Delta=5$。",
        "",
    ]
    lines.extend(_aminer_table_lines(aminer))
    lines.extend([
        "",
        "## 4. 五档扰动率明细",
        "",
    ])
    for attack in ("prbcd", "heteprbcd", "rnd"):
        lines.extend([f"### {attack.upper()}", ""])
        lines.extend(_attack_rate_table_lines(benchmark, "aminer", attack))
        lines.append("")
    lines.extend([
        "## 5. 攻击有效性审计",
        "",
        "| Attack | Relations | Realized $r$ | Train change share | Surrogate drop (pp) | Formal drop min/median/max (pp) |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in aminer_audit:
        lines.append(
            f"| {row['attack'].upper()} | {row['relations']} | "
            f"{100 * row['actual_rate_min']:.2f}–{100 * row['actual_rate_max']:.2f}% | "
            f"{100 * row['train_change_share_min']:.1f}–{100 * row['train_change_share_max']:.1f}% | "
            f"{row['surrogate_drop_pp_min']:.2f}–{row['surrogate_drop_pp_max']:.2f} | "
            f"{row['formal_model_drop_pp_min']:+.2f}/"
            f"{row['formal_model_drop_pp_median']:+.2f}/"
            f"{row['formal_model_drop_pp_max']:+.2f} |"
        )
    lines.extend([
        "",
        "## 6. 结果分析",
        "",
        "1. Poisoning、RND 与目标逃逸分别报告，不将不同威胁模型混合平均。",
        "2. 11 模型使用相同 clean、split、攻击 artifact 和训练种子，横向比较具有一致输入基础。",
        "3. PRBCD/HetePRBCD 的预算和验证均正确，但替代模型下降偏小，正式模型下降弱且具有模型依赖性；这些数据不能单独支撑强鲁棒性结论。",
        "4. HG $\\Delta=5$ 应结合 clean-target 起点和下降幅度解释；完整目标结果见 `docs/target-evasion-results.md`。",
        "",
    ])
    return "\n".join(lines)


def _attack_rate_table_lines(summary, dataset, attack):
    lines = [
        "| Model | Clean | 5% | 10% | 15% | 20% | 25% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        cells = [
            _score(_lookup(
                summary, dataset=dataset, model=model, attack="clean", rate=0.0,
            ))
        ]
        cells.extend(
            _score(_lookup(
                summary, dataset=dataset, model=model, attack=attack, rate=rate,
            ))
            for rate in (5.0, 10.0, 15.0, 20.0, 25.0)
        )
        lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(cells) + " |")
    return lines


def _target_document(targets, adaptive, adaptive_significance, adaptive_ranks):
    lines = [
        "# 目标逃逸实验结果",
        "",
        "> **专项附录。** 主结论与统一口径以 `docs/final-experiment-results.md` 为准。本页记录目标节点 Micro-F1 明细。",
        "",
        "## 1. 实验设置",
        "",
        "- HG Baseline：模型在 clean 图训练，测试时替换目标节点攻击图，`adaptive=false`。",
        "- Adaptive Query：11 个模型分别绑定自身 checkpoint，独立执行 score-based 贪心查询，`adaptive=true`。",
        "- 候选池：每目标 64 条候选增边和 64 条候选删边；相同数据集和 seed 跨模型共享候选池。",
        "- 扰动预算：$\\Delta\\in\\{1,3,5\\}$ 是每目标上限；正式配对重复为 $(s_a,s_t)=(1,1),(2,2),(3,3)$。",
        "",
        "## 2. HG Baseline",
        "",
    ]
    for dataset in ("acm", "dblp", "aminer"):
        lines.extend([f"### {dataset.upper()}", ""])
        lines.append("| Model | Clean target | $\\Delta=1$ | $\\Delta=3$ | $\\Delta=5$ | Drop@5 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for model in MODEL_ORDER:
            rows = [
                _lookup(
                    targets,
                    protocol=(
                        "aminer_hg_baseline_target_evasion_v1"
                        if dataset == "aminer" else "hg_baseline_target_evasion_v1"
                    ),
                    dataset=dataset,
                    model=model,
                    rate=rate,
                )
                for rate in (1.0, 3.0, 5.0)
            ]
            if any(row is None for row in rows):
                continue
            clean = rows[0]
            lines.append(
                f"| {MODEL_LABELS[model]} | "
                f"{100 * clean['clean_micro_f1_mean']:.2f} ± {100 * clean['clean_micro_f1_std']:.2f} | "
                + " | ".join(_score(row) for row in rows)
                + f" | {rows[-1]['drop_pp_mean']:+.2f} |"
            )
        lines.append("")
    lines.extend(["## 3. 统一十一模型自适应攻击", ""])
    for dataset in ("acm", "dblp", "aminer"):
        lines.extend([f"### {dataset.upper()}", ""])
        lines.extend(_formal_adaptive_table(adaptive, dataset))
        lines.append("")
    lines.extend([
        "### 攻击有效性诊断",
        "",
    ])
    lines.extend(_formal_adaptive_diagnostic_table(adaptive))
    lines.extend(["", "### 平均排名", ""])
    lines.extend(_formal_adaptive_rank_table(adaptive_ranks))
    lines.extend([
        "",
        "### DVCL 与全部基线的配对比较",
        "",
        "每个数据集先在同一 seed 内对 $\\Delta=1,3,5$ 的攻击后 Micro-F1 求均值，再以 3 个配对 seed 做双侧 Wilcoxon 检验；Holm 校正在每个数据集的 10 个基线比较内执行。95% CI 为配对均值差的 t 区间。",
        "",
    ])
    lines.extend(_formal_adaptive_significance_table(adaptive_significance))
    lines.extend([
        "",
        "## 4. 分析与限制",
        "",
        "1. HG Baseline 衡量固定攻击的跨模型迁移效果，不能替代自适应鲁棒性结论。",
        "2. 正式自适应攻击与每个 victim checkpoint 绑定，但仍是 64+64 有限候选查询，不等同于完整候选空间或梯度白盒最强攻击。",
        "3. DVCL 在 ACM、AMiner 当前攻击下无观察到的成功扰动；该结果必须表述为有限候选攻击未成功，不能升级为普遍鲁棒性。",
        "4. DBLP 中 DVCL Drop@5 为 46.00 个百分点，平均排名 2.22，低于 HeteroGuard 的 1.06，说明其自适应脆弱性具有数据集依赖性。",
        "5. 每数据集只有 3 个独立配对重复，Holm 校正后无显著比较；正式结论应同时报告效应量、CI、排名和攻击诊断。",
        "",
    ])
    return "\n".join(lines)


def _write_figures(directory, multi, ranks, adaptive):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    models = ("han", "heterosage", "hseco", "dvcl")
    for dataset in ("acm", "dblp"):
        for attack in ("prbcd", "heteprbcd"):
            figure, axis = plt.subplots(figsize=(6.2, 4.2))
            for model in models:
                selected = sorted(
                    (
                        row for row in multi
                        if row["dataset"] == dataset
                        and row["attack"] == attack
                        and row["model"] == model
                    ),
                    key=lambda row: row["rate"],
                )
                rates = [row["rate"] for row in selected]
                means = [100 * row["micro_f1_mean"] for row in selected]
                errors = [100 * row["micro_f1_std"] for row in selected]
                axis.errorbar(
                    rates, means, yerr=errors, marker="o", capsize=3,
                    label=MODEL_LABELS[model],
                )
            axis.set_xlabel("Perturbation rate (%)")
            axis.set_ylabel("Micro-F1 (%)")
            axis.set_title(f"{dataset.upper()} — {attack.upper()}")
            axis.grid(alpha=0.25)
            axis.legend(frameon=False)
            figure.tight_layout()
            _save_figure(figure, directory / f"{dataset}_{attack}_curve")
            plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    selected = sorted(
        (row for row in ranks if row["dataset"] == "all"),
        key=lambda row: row["average_rank"], reverse=True,
    )
    axis.barh(
        [MODEL_LABELS[row["model"]] for row in selected],
        [row["average_rank"] for row in selected],
    )
    axis.set_xlabel("Average rank (lower is better)")
    axis.set_title("ACM + DBLP multi-seed poisoning")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, directory / "multi_seed_average_rank")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), sharey=True)
    for axis, dataset in zip(axes, ("acm", "dblp", "aminer")):
        for model in MODEL_ORDER:
            selected = [
                _lookup(adaptive, dataset=dataset, model=model, rate=rate)
                for rate in (1, 3, 5)
            ]
            style = (
                {"color": "black", "marker": "*", "markersize": 7,
                 "linewidth": 2.4, "zorder": 10}
                if model == "dvcl"
                else {"marker": "o", "markersize": 3, "linewidth": 1.2}
            )
            axis.plot(
                [1, 3, 5],
                [100 * row["attacked_target_micro_f1_mean"] for row in selected],
                label=MODEL_LABELS[model], **style,
            )
        axis.set_title(dataset.upper())
        axis.set_xlabel("Budget Δ")
        axis.set_ylim(0, 100)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Target Micro-F1 (%)")
    axes[-1].legend(
        frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8
    )
    figure.tight_layout(rect=(0, 0, 0.89, 1))
    _save_figure(figure, directory / "adaptive_target_evasion_11model")
    plt.close(figure)


def _save_figure(figure, base):
    figure.savefig(base.with_suffix(".png"), dpi=300)
    figure.savefig(base.with_suffix(".pdf"))


if __name__ == "__main__":
    raise SystemExit(main())
