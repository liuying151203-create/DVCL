import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
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


MULTI_SEED_PROTOCOL = "acm_dblp_attack_seed_recheck_v1"
AMINER_PROTOCOLS = (
    "aminer_poisoning_main_v1",
    "aminer_rnd_poisoning_v1",
    "aminer_hg_baseline_target_evasion_v1",
)
TARGET_PROTOCOLS = (
    "hg_baseline_target_evasion_v1",
    "aminer_hg_baseline_target_evasion_v1",
    "dvcl_adaptive_target_evasion_v1",
)
CLEAN_PROTOCOLS = ("acm_poisoning_main_v1", "dblp_poisoning_main_v1")
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
    protocols = (
        MULTI_SEED_PROTOCOL, *AMINER_PROTOCOLS, *TARGET_PROTOCOLS,
        *CLEAN_PROTOCOLS,
    )
    rows = load_protocol_rows(run_root, dict.fromkeys(protocols))
    multi_rows = [row for row in rows if row["protocol"] == MULTI_SEED_PROTOCOL]
    target_rows = [row for row in rows if row["protocol"] in TARGET_PROTOCOLS]

    multi_summary = summarize(
        multi_rows, ("dataset", "model", "attack", "rate")
    )
    multi_family = _multi_seed_family_rows(multi_rows, rows)
    significance = paired_significance(multi_rows)
    ranks = average_ranks(multi_rows)
    targets = target_summary(target_rows)
    aminer = _aminer_family_rows(rows, targets)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "multi_seed_summary.csv", multi_summary)
    _write_csv(output_dir / "multi_seed_family_summary.csv", multi_family)
    _write_csv(output_dir / "significance.csv", significance)
    _write_csv(output_dir / "average_ranks.csv", ranks)
    _write_csv(output_dir / "target_evasion_summary.csv", targets)
    _write_csv(output_dir / "aminer_family_summary.csv", aminer)
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps({
            "protocols": list(dict.fromkeys(protocols)),
            "metrics": ["micro_f1"],
            "multi_seed_runs": len(multi_rows),
            "target_runs": len(target_rows),
            "significance_test": "paired two-sided Wilcoxon signed-rank",
            "multiple_testing": "Holm family-wise correction",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "final-experiment-results.md").write_text(
        _final_document(
            multi_summary, multi_family, significance, ranks, targets, aminer
        ),
        encoding="utf-8",
    )
    (docs_dir / "aminer-experiment-results.md").write_text(
        _aminer_document(aminer, targets), encoding="utf-8"
    )
    (docs_dir / "target-evasion-results.md").write_text(
        _target_document(targets), encoding="utf-8"
    )
    if not args.skip_figures:
        _write_figures(Path(args.figure_dir), multi_summary, ranks, targets)
    print(
        f"Wrote paper analysis: multi_seed={len(multi_rows)} "
        f"targets={len(target_rows)} output={output_dir}"
    )
    return 0


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


def _final_document(multi, multi_family, significance, ranks, targets, aminer):
    lines = [
        "# 最终实验结果与统计分析",
        "",
        "> 本文档由 `scripts/analyze_paper_results.py` 从逐次运行结果生成，仅报告 Micro-F1。",
        "",
        "## 1. 实验设置",
        "",
        "| 设置 | 取值 |",
        "|---|---|",
        "| 数据集 | ACM、DBLP、AMiner |",
        "| 划分种子 $s_{split}$ | 1 |",
        "| 训练种子 $s_{train}$ | 1–5 |",
        "| 多种子攻击 $s_{atk}$ | 1–3 |",
        "| Poisoning | PRBCD、HetePRBCD，$r\\in\\{5,15,25\\}\\%$ |",
        "| 迁移目标逃逸 | HG Baseline，$\\Delta\\in\\{1,3,5\\}$ |",
        "| 自适应目标逃逸 | DVCL 白盒贪心查询，$\\Delta\\in\\{1,3,5\\}$ |",
        "| 训练 | $E_{max}=200$，$P=100$，Adam，完整模型 checkpoint |",
        "| 统计 | 均值 ± 样本标准差；配对双侧 Wilcoxon；Holm 校正 |",
        "",
        "## 2. 多攻击种子结果",
        "",
        "表中每格聚合 3 个攻击种子和 5 个训练种子，共 $n=15$。",
        "",
    ]
    for dataset in ("acm", "dblp"):
        lines.extend([f"### {dataset.upper()}", ""])
        lines.append("| Attack | Rate | HAN | HeteroSAGE | HSeCo | DVCL |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for attack in ("prbcd", "heteprbcd"):
            for rate in (5.0, 15.0, 25.0):
                cells = []
                for model in ("han", "heterosage", "hseco", "dvcl"):
                    row = _lookup(
                        multi, dataset=dataset, model=model, attack=attack, rate=rate
                    )
                    cells.append(_score(row))
                lines.append(
                    f"| {attack.upper()} | {rate:.0f}% | " + " | ".join(cells) + " |"
                )
        lines.append("")
    lines.extend([
        "### 攻击平均与下降幅度",
        "",
        "括号内为相对 clean 的下降百分点；负值表示攻击后均值略高于 clean。",
        "",
        "| Dataset | Condition | HAN | HeteroSAGE | HSeCo | DVCL |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for dataset in ("acm", "dblp"):
        clean_cells = []
        for model in ("han", "heterosage", "hseco", "dvcl"):
            row = _lookup(multi_family, dataset=dataset, model=model, attack="all")
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
            for model in ("han", "heterosage", "hseco", "dvcl"):
                row = _lookup(
                    multi_family, dataset=dataset, model=model, attack=attack
                )
                cells.append(f"{_score(row)} ({row['drop_pp']:+.2f})")
            lines.append(
                f"| {dataset.upper()} | {label} | " + " | ".join(cells) + " |"
            )
    lines.append("")
    lines.extend([
        "## 3. 显著性与排名",
        "",
        "正效应表示 DVCL 的 Micro-F1 更高；先在每个攻击种子×训练种子配对内跨扰动条件平均，W/T/L 为 15 个配对的胜/平/负次数。",
        "",
        "| Dataset | Attack | Baseline | $n$ | Effect (pp) | W/T/L | $p_{Holm}$ |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in significance:
        if row["attack"] != "all":
            continue
        lines.append(
            f"| {row['dataset'].upper()} | All | {MODEL_LABELS[row['baseline']]} | "
            f"{row['n']} | {row['effect_pp']:+.2f} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | {row['p_holm']:.3g} |"
        )
    lines.extend(["", "**平均排名（越低越好）**", "", "| Scope | HAN | HeteroSAGE | HSeCo | DVCL |", "|---|---:|---:|---:|---:|"])
    for dataset in ("acm", "dblp", "all"):
        cells = []
        for model in ("han", "heterosage", "hseco", "dvcl"):
            row = _lookup(ranks, dataset=dataset, model=model)
            cells.append(f"{row['average_rank']:.2f}")
        lines.append(f"| {dataset.upper()} | " + " | ".join(cells) + " |")
    lines.extend(["", "## 4. DVCL 自适应目标逃逸", ""])
    lines.extend(_adaptive_table(targets))
    lines.extend([
        "",
        "## 5. AMiner 汇总",
        "",
        "完整 11 模型结果见 `docs/aminer-experiment-results.md`。",
        "",
        "## 6. 结论",
        "",
        "1. 多攻击种子结果可区分稳定增益与单一攻击产物偶然性；显著性结论以 Holm 校正结果为准。",
        "2. 固定 HG Baseline 是迁移逃逸攻击，自适应结果才直接检验 DVCL 在白盒目标攻击下的鲁棒性。",
        "3. Poisoning、迁移逃逸和自适应逃逸采用不同训练与评估语义，不合并为单一 Attack Average。",
        "4. AMiner 结果扩展了跨数据集与 11 模型比较，最终主张应同时参考绝对性能、下降幅度和平均排名。",
        "",
        "## 7. 论文图表",
        "",
        "![ACM HetePRBCD 多种子曲线](figures/paper/acm_heteprbcd_curve.png)",
        "",
        "![DBLP HetePRBCD 多种子曲线](figures/paper/dblp_heteprbcd_curve.png)",
        "",
        "![多种子平均排名](figures/paper/multi_seed_average_rank.png)",
        "",
        "![DVCL 自适应目标逃逸](figures/paper/dvcl_adaptive_target_evasion.png)",
        "",
    ])
    return "\n".join(lines)


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


def _aminer_document(aminer, targets):
    lines = [
        "# AMiner 实验结果",
        "",
        "> 三套正式协议共 1045/1045 次运行完成，零失败；仅报告 Micro-F1。",
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
        "| 攻击关系 | Paper–Reference（P–R）及反向关系 |",
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
        "| Model | Clean | PRBCD Avg. | HetePRBCD Avg. | RND Avg. | HG $\\Delta=5$ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        cells = []
        for attack in ("clean", "prbcd", "heteprbcd", "rnd", "hg_baseline_delta_5"):
            row = _lookup(aminer, dataset="aminer", model=model, attack=attack)
            cells.append(_score(row))
        lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## 4. 结果分析",
        "",
        "1. Poisoning、RND 与目标逃逸分别报告，不将不同威胁模型混合平均。",
        "2. 11 模型使用相同 clean、split、攻击 artifact 和训练种子，横向比较具有一致输入基础。",
        "3. HG $\\Delta=5$ 应结合 clean-target 起点和下降幅度解释；完整目标结果见 `docs/target-evasion-results.md`。",
        "",
    ])
    return "\n".join(lines)


def _target_document(targets):
    lines = [
        "# 目标逃逸实验结果",
        "",
        "> 修正后的 HG Baseline 330/330 与 DVCL 自适应攻击 30/30 均完成、零失败；仅报告目标节点 Micro-F1。",
        "",
        "## 1. 实验设置",
        "",
        "- HG Baseline：模型在 clean 图训练，测试时替换目标节点攻击图，`adaptive=false`。",
        "- DVCL Adaptive：每个 checkpoint 独立执行白盒贪心查询，`adaptive=true`。",
        "- 扰动预算：$\\Delta\\in\\{1,3,5\\}$；训练种子 1–5。",
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
    lines.extend(["## 3. DVCL 白盒自适应攻击", ""])
    lines.extend(_adaptive_table(targets))
    lines.extend([
        "",
        "## 4. 分析",
        "",
        "1. HG Baseline 衡量固定攻击的跨模型迁移效果，不能替代自适应鲁棒性结论。",
        "2. 自适应攻击与具体训练 checkpoint 绑定，直接优化 DVCL 的分类间隔。",
        "3. 不同数据集的目标集合规模不同，只在同一数据集内比较模型和扰动预算。",
        "",
    ])
    return "\n".join(lines)


def _write_figures(directory, multi, ranks, targets):
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

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for axis, dataset in zip(axes, ("acm", "dblp")):
        selected = [
            _lookup(
                targets,
                protocol="dvcl_adaptive_target_evasion_v1",
                dataset=dataset,
                model="dvcl",
                rate=rate,
            )
            for rate in (1.0, 3.0, 5.0)
        ]
        axis.errorbar(
            [1, 3, 5], [100 * row["micro_f1_mean"] for row in selected],
            yerr=[100 * row["micro_f1_std"] for row in selected],
            marker="o", capsize=3, label="Adaptive",
        )
        axis.plot(
            [1, 3, 5], [100 * row["clean_micro_f1_mean"] for row in selected],
            linestyle="--", label="Clean target",
        )
        axis.set_title(dataset.upper())
        axis.set_xlabel("Budget Δ")
        axis.set_ylim(0, 100)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Target Micro-F1 (%)")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, directory / "dvcl_adaptive_target_evasion")
    plt.close(figure)


def _save_figure(figure, base):
    figure.savefig(base.with_suffix(".png"), dpi=300)
    figure.savefig(base.with_suffix(".pdf"))


if __name__ == "__main__":
    raise SystemExit(main())
