import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.analyze_dvcl_view_diagnosis import (
    _manifest_summary,
    aggregate_rows,
    collect_clean_rows,
    collect_evaluation_rows,
    write_csv,
)


DATASETS = ("acm", "dblp", "aminer")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit and aggregate the DVCL reliability-gate pilot."
    )
    parser.add_argument(
        "--clean-config",
        default="configs/protocols/dvcl_reliability_gate_clean_pilot_v1.yaml",
    )
    parser.add_argument(
        "--evaluation-config",
        default="configs/protocols/dvcl_reliability_gate_pilot_v1.yaml",
    )
    parser.add_argument(
        "--baseline-clean-config",
        default="configs/protocols/dvcl_view_diagnosis_clean_pilot_v1.yaml",
    )
    parser.add_argument(
        "--baseline-evaluation-config",
        default="configs/protocols/dvcl_view_diagnosis_pilot_v1.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/analysis/dvcl_reliability_gate_pilot_v1",
    )
    parser.add_argument(
        "--report", default="docs/dvcl-reliability-gate-results.md"
    )
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def stage_e_decision(
    clean_rows, summary, baseline_clean_rows, baseline_summary,
    candidate_variant="reliability_gate",
):
    clean = _clean_lookup(clean_rows)
    baseline_clean = _clean_lookup(baseline_clean_rows)
    values = _summary_lookup(summary)
    baseline = _summary_lookup(baseline_summary)
    candidate = values[("dblp", candidate_variant, "adaptive_query", 5)]
    reference = baseline[("dblp", "concat", "adaptive_query", 5)]
    dblp_gain = (
        float(candidate["attacked_target_micro_f1_mean"])
        - float(reference["attacked_target_micro_f1_mean"])
    )
    max_clean_loss = max(
        baseline_clean[(dataset, "concat")]
        - clean[(dataset, candidate_variant)]
        for dataset in DATASETS
    )
    max_other_attack_loss = max(
        float(baseline[(dataset, "concat", "adaptive_query", 5)][
            "attacked_target_micro_f1_mean"
        ])
        - float(values[(dataset, candidate_variant, "adaptive_query", 5)][
            "attacked_target_micro_f1_mean"
        ])
        for dataset in ("acm", "aminer")
    )
    clean_gate_not_collapsed = all(
        clean[(dataset, candidate_variant, "gate_std")] >= 0.02
        and clean[(dataset, candidate_variant, "gate_topology_fraction")] < 0.95
        and clean[(dataset, candidate_variant, "gate_feature_fraction")] < 0.95
        for dataset in DATASETS
    )
    result = {
        "candidate_variant": candidate_variant,
        "dblp_adaptive_gain": dblp_gain,
        "max_clean_loss": max_clean_loss,
        "max_acm_aminer_attacked_loss": max_other_attack_loss,
        "clean_gate_not_collapsed": clean_gate_not_collapsed,
    }
    result["passes"] = (
        dblp_gain >= 0.05
        and max_clean_loss <= 0.015 + 1e-12
        and max_other_attack_loss <= 0.02 + 1e-12
        and clean_gate_not_collapsed
    )
    if result["passes"]:
        result["next_action"] = (
            f"expand_{candidate_variant}_to_three_paired_seeds"
        )
    elif candidate_variant == "reliability_gate":
        result["next_action"] = "implement_reliability_gate_aug_pilot"
    else:
        result["next_action"] = "retain_concat_and_close_stage_e"
    return result


def render_report(
    clean_rows, summary, baseline_clean_rows, baseline_summary, decision, audit,
    candidate_variant="reliability_gate",
):
    clean = _clean_lookup(clean_rows)
    baseline_clean = _clean_lookup(baseline_clean_rows)
    values = _summary_lookup(summary)
    baseline = _summary_lookup(baseline_summary)
    lines = [
        f"# DVCL `{candidate_variant}` Pilot 结果",
        "",
        "## 1. 实验设置",
        "",
        "- 数据集：ACM、DBLP、AMiner；单配对种子 $(s_a,s_t)=(1,1)$。",
        "- 融合：$[\\alpha_i h_i^t\\,\\|\\,(1-\\alpha_i)h_i^f]$；门控只读取双视图熵、置信边界、JS 分歧、余弦一致性和范数比，不读取测试标签或 clean/attacked 成对信息。",
        "- 损失：$L=L_c+\\lambda_hL_h+\\lambda_dL_d+\\beta L_{aux}+\\lambda_rL_r$，其中 $\\lambda_h=\\lambda_d=\\lambda_r=1$，$\\beta=0.5$，$\\tau_r=1$，$d_g=16$。",
        "- 攻击：HG Baseline 迁移攻击和针对候选完整模型重新优化的 64+64 候选自适应查询攻击；$\\Delta=\\{1,3,5\\}$。",
        "- 指标：仅报告 Micro-F1；本阶段是单种子机制筛选，不作显著性结论。",
        "",
        "## 2. Clean Micro-F1",
        "",
        f"| Dataset | `feat` | `concat` | `{candidate_variant}` | 相对 `concat` | $\\alpha$ mean±std |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if candidate_variant.endswith("_aug"):
        lines.insert(8,
            "- 训练增强：仅在训练阶段随机重连 10% 拓扑图边，并加入 $L_{aug}$；推理结构与 `reliability_gate` 相同。"
        )
    for dataset in DATASETS:
        candidate = clean[(dataset, candidate_variant)]
        reference = baseline_clean[(dataset, "concat")]
        lines.append(
            f"| {dataset.upper()} | {_percent(baseline_clean[(dataset, 'feat')])} | "
            f"{_percent(reference)} | {_percent(candidate)} | "
            f"{_points(candidate - reference)} | "
            f"{clean[(dataset, candidate_variant, 'gate_mean')]:.4f}±"
            f"{clean[(dataset, candidate_variant, 'gate_std')]:.4f} |"
        )
    for attack, title in (
        ("hg_baseline", "HG Baseline 目标逃逸"),
        ("adaptive_query", "模型自适应目标逃逸"),
    ):
        lines.extend(["", f"## {3 if attack == 'hg_baseline' else 4}. {title}", ""])
        for dataset in DATASETS:
            lines.extend([
                f"### {dataset.upper()}",
                "",
                "| Model | Clean target | $\\Delta=1$ | $\\Delta=3$ | $\\Delta=5$ | Drop@5 | ASR@5 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ])
            for variant, source in (
                ("feat", baseline),
                ("concat", baseline),
                (candidate_variant, values),
            ):
                rows = [source[(dataset, variant, attack, rate)] for rate in (1, 3, 5)]
                rate5 = rows[-1]
                lines.append(
                    f"| `{variant}` | {_percent(float(rate5['clean_target_micro_f1_mean']))} | "
                    + " | ".join(
                        _percent(float(row["attacked_target_micro_f1_mean"]))
                        for row in rows
                    )
                    + f" | {_points(float(rate5['micro_f1_drop_mean']))} | "
                    + f"{_percent(float(rate5['attack_success_rate_mean']))} |"
                )
            lines.append("")
    lines.extend([
        "## 5. 门控行为与验收",
        "",
        "| Dataset | Attack | $\\alpha$ clean→attack | std clean→attack | View disagreement clean→attack |",
        "|---|---|---:|---:|---:|",
    ])
    for dataset in DATASETS:
        for attack in ("hg_baseline", "adaptive_query"):
            row = values[(dataset, candidate_variant, attack, 5)]
            lines.append(
                f"| {dataset.upper()} | `{attack}` | "
                f"{_transition(row, 'gate_clean_mean_mean', 'gate_attacked_mean_mean')} | "
                f"{_transition(row, 'gate_clean_std_mean', 'gate_attacked_std_mean')} | "
                f"{_transition(row, 'clean_view_disagreement_rate_mean', 'attacked_view_disagreement_rate_mean', percent=True)} |"
            )
    verdict = "通过" if decision["passes"] else "未通过"
    if decision["passes"]:
        next_step = "扩展到 3 个配对种子"
    elif candidate_variant == "reliability_gate":
        next_step = (
            "保持当前结果不变，进入带训练时结构扰动的 "
            "`reliability_gate_aug` Pilot"
        )
    else:
        next_step = (
            "停止继续追逐单种子测试结果，保留 `concat` 作为主模型并结束阶段 E"
        )
    lines.extend([
        "",
        f"- DBLP 自适应 $\\Delta=5$ 相对 `concat` 增益：{_points(decision['dblp_adaptive_gain'])}（门槛 $\\geq5$ pp）。",
        f"- 三数据集最大 clean 损失：{_points(decision['max_clean_loss'])}（门槛 $\\leq1.5$ pp）。",
        f"- ACM/AMiner 最大攻击后损失：{_points(decision['max_acm_aminer_attacked_loss'])}（门槛 $\\leq2$ pp）。",
        f"- 门控非塌缩：{'是' if decision['clean_gate_not_collapsed'] else '否'}（clean $\\alpha$ std $\\geq0.02$，极端路由比例均 $<95\\%$）。",
        f"- 阶段 E Pilot 判定：**{verdict}**；下一步{next_step}。",
        f"- 完整性审计：clean {audit['clean_physical_runs']}，攻击评估 {audit['evaluation_physical_runs']}，逻辑结果 {audit['logical_results']}，问题数 {len(audit['issues'])}。",
    ])
    if audit.get("dirty_manifests"):
        lines.append(
            f"- {audit['dirty_manifests']} 个候选运行 manifest 标记为 dirty worktree；"
            "本 Pilot 仅用于机制筛选，通过后仍须在冻结提交上重跑正式统计。"
        )
    lines.append("")
    return "\n".join(lines)


def _clean_lookup(rows):
    output = {}
    for row in rows:
        dataset = row["dataset"]
        variant = row["variant"]
        output[(dataset, variant)] = float(row["full_test_micro_f1"])
        for key, value in row.items():
            if key.startswith("gate_") and isinstance(value, (int, float)):
                output[(dataset, variant, key)] = float(value)
    return output


def _summary_lookup(rows):
    return {
        (row["dataset"], row["variant"], row["attack"], int(row["rate"])): row
        for row in rows
    }


def _percent(value):
    return f"{100 * float(value):.2f}"


def _points(value):
    return f"{100 * float(value):.2f} pp"


def _transition(row, left, right, percent=False):
    if left not in row or right not in row:
        return "—"
    if percent:
        return f"{_percent(row[left])}→{_percent(row[right])}"
    return f"{float(row[left]):.4f}→{float(row[right]):.4f}"


def _collect(config_path, clean):
    if clean:
        rows, issues, expected, completed, manifests = collect_clean_rows(
            config_path
        )
        return rows, issues, expected, completed, expected, manifests
    rows, per_target, issues, expected, completed, logical, manifests = (
        collect_evaluation_rows(config_path)
    )
    embedded = [row["_issue"] for row in rows if "_issue" in row]
    rows = [row for row in rows if "_issue" not in row]
    return rows, issues + embedded, expected, completed, logical, manifests, per_target


def main():
    args = parse_args()
    clean_path = (ROOT / args.clean_config).resolve()
    evaluation_path = (ROOT / args.evaluation_config).resolve()
    baseline_clean_path = (ROOT / args.baseline_clean_config).resolve()
    baseline_evaluation_path = (ROOT / args.baseline_evaluation_config).resolve()
    output_root = (ROOT / args.output_root).resolve()
    clean_rows, clean_issues, clean_expected, clean_completed, _, clean_manifests = (
        _collect(clean_path, True)
    )
    (
        rows, evaluation_issues, expected, completed, expected_logical,
        evaluation_manifests, per_target,
    ) = _collect(evaluation_path, False)
    (
        baseline_clean_rows, baseline_clean_issues, baseline_clean_expected,
        baseline_clean_completed, _, _,
    ) = _collect(baseline_clean_path, True)
    (
        baseline_rows, baseline_issues, baseline_expected, baseline_completed,
        baseline_logical, _, _,
    ) = _collect(baseline_evaluation_path, False)
    summary = aggregate_rows(rows)
    baseline_summary = aggregate_rows(baseline_rows)
    candidate_variants = sorted({row["variant"] for row in clean_rows})
    if len(candidate_variants) != 1:
        clean_issues.append(
            f"expected one candidate variant, found {candidate_variants}"
        )
    candidate_variant = (
        candidate_variants[0] if len(candidate_variants) == 1 else "unknown"
    )
    issues = clean_issues + evaluation_issues
    baseline_audit_ok = (
        baseline_clean_completed == baseline_clean_expected
        and baseline_completed == baseline_expected
        and len(baseline_rows) == baseline_logical
        and not baseline_clean_issues
        and not baseline_issues
    )
    definitions_ok = all(
        row.get("view_definition") == "independent auxiliary view classifiers"
        for row in rows
    )
    if rows and not definitions_ok:
        issues.append("reliability-gate view diagnostics use an invalid definition")
    complete = (
        clean_completed == clean_expected
        and completed == expected
        and len(rows) == expected_logical
        and not issues
        and baseline_audit_ok
    )
    decision = None
    if complete:
        decision = stage_e_decision(
            clean_rows, summary, baseline_clean_rows, baseline_summary,
            candidate_variant,
        )
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "clean.csv", clean_rows)
    write_csv(output_root / "runs.csv", rows)
    write_csv(output_root / "summary.csv", summary)
    write_csv(output_root / "per_target.csv", per_target)
    manifest_summary = _manifest_summary(clean_manifests + evaluation_manifests)
    audit = {
        "clean_config": str(clean_path),
        "evaluation_config": str(evaluation_path),
        "clean_physical_runs": f"{clean_completed}/{clean_expected}",
        "evaluation_physical_runs": f"{completed}/{expected}",
        "logical_results": f"{len(rows)}/{expected_logical}",
        "baseline_audit_ok": baseline_audit_ok,
        "view_definitions_ok": definitions_ok,
        "issues": issues,
        **manifest_summary,
        "ok": complete,
    }
    (output_root / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if decision is not None:
        (output_root / "decision.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report_path = (ROOT / args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            render_report(
                clean_rows, summary, baseline_clean_rows, baseline_summary,
                decision, audit, candidate_variant,
            ),
            encoding="utf-8",
        )
    print(
        f"clean={clean_completed}/{clean_expected} "
        f"evaluation={completed}/{expected} "
        f"logical={len(rows)}/{expected_logical} issues={len(issues)}"
    )
    return 0 if complete or args.allow_partial else 1


if __name__ == "__main__":
    raise SystemExit(main())
