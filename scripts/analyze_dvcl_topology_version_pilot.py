import argparse
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
    aggregate_rows,
    collect_evaluation_rows,
    write_csv,
)
from dvcl_bench.paths import ExperimentLayout


VARIANTS = ("graph_hard", "graph_no_filter")
VARIANT_LABELS = {
    "graph_hard": "Graph + hard filter + $L_{HAN}$",
    "graph_no_filter": "Graph + no second filter + $L_{HAN}$",
}
STRATEGIES = {
    "graph_hard": ("graph", "hard"),
    "graph_no_filter": ("graph", "none"),
}
CONDITIONS = (("clean", 0), ("heteprbcd", 25))
SEED_PAIRS = ((1, 1), (2, 2), (3, 3))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit and aggregate the DVCL topology-version pilot."
    )
    parser.add_argument(
        "--train-config",
        default="configs/protocols/dvcl_topology_version_train_pilot_v1.yaml",
    )
    parser.add_argument(
        "--adaptive-config",
        default="configs/protocols/dvcl_topology_version_adaptive_pilot_v1.yaml",
    )
    parser.add_argument(
        "--train-input-audit",
        default="outputs/audits/dvcl_topology_version_train_pilot_v1-inputs.json",
    )
    parser.add_argument(
        "--adaptive-input-audit",
        default="outputs/audits/dvcl_topology_version_adaptive_pilot_v1-inputs.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/analysis/dvcl_topology_version_pilot_v1",
    )
    parser.add_argument(
        "--report", default="docs/dvcl-topology-version-pilot.md"
    )
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def collect_training_rows(config_path):
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
        variant = spec.model.config["variant"]
        diagnostics = payload.get("diagnostics", {})
        expected_source, expected_filter = STRATEGIES[variant]
        if diagnostics.get("topology_source") != expected_source:
            issues.append(f"topology source mismatch: {run_dir}")
        if diagnostics.get("semantic_topology_filter") != expected_filter:
            issues.append(f"semantic filter mismatch: {run_dir}")
        micro_f1 = float(payload["metrics"]["micro_f1"])
        accuracy = float(payload["metrics"]["accuracy"])
        if abs(micro_f1 - accuracy) > 1e-12:
            issues.append(f"accuracy and Micro-F1 differ: {run_dir}")
        rows.append({
            "variant": variant,
            "attack": spec.attack.name,
            "rate": int(spec.attack.rate),
            "attack_seed": spec.seeds.attack,
            "train_seed": spec.seeds.train,
            "micro_f1": micro_f1,
            "best_epoch": int(payload["best_epoch"]),
            "run_dir": str(run_dir.resolve()),
        })
    return rows, issues, len(commands), completed, manifests


def validate_training_matrix(rows):
    expected = {
        (variant, attack, rate, attack_seed, train_seed)
        for variant in VARIANTS
        for attack, rate in CONDITIONS
        for attack_seed, train_seed in SEED_PAIRS
    }
    actual = {
        (
            row["variant"], row["attack"], row["rate"],
            row["attack_seed"], row["train_seed"],
        )
        for row in rows
    }
    issues = []
    if missing := sorted(expected - actual):
        issues.append(f"missing training rows: {missing[:10]} count={len(missing)}")
    if unexpected := sorted(actual - expected):
        issues.append(
            f"unexpected training rows: {unexpected[:10]} count={len(unexpected)}"
        )
    if len(actual) != len(rows):
        issues.append(
            f"duplicate training rows: rows={len(rows)} unique={len(actual)}"
        )
    return issues


def summarize_training(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["attack"], row["rate"])].append(
            float(row["micro_f1"])
        )
    output = []
    for key, values in sorted(grouped.items()):
        output.append({
            "variant": key[0],
            "attack": key[1],
            "rate": key[2],
            "n": len(values),
            "micro_f1_mean": statistics.fmean(values),
            "micro_f1_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    return output


def topology_version_decision(training_rows, adaptive_summary):
    training = {
        (
            row["variant"], row["attack"], row["rate"],
            row["attack_seed"], row["train_seed"],
        ): float(row["micro_f1"])
        for row in training_rows
    }
    adaptive = {
        (row["variant"], row["attack"], int(row["rate"])): row
        for row in adaptive_summary
    }
    candidates = []
    for variant in VARIANTS[1:]:
        clean_losses = []
        poisoning_losses = []
        for attack_seed, train_seed in SEED_PAIRS:
            clean_losses.append(
                training[("graph_hard", "clean", 0, attack_seed, train_seed)]
                - training[(variant, "clean", 0, attack_seed, train_seed)]
            )
            poisoning_losses.append(
                training[(
                    "graph_hard", "heteprbcd", 25, attack_seed, train_seed,
                )]
                - training[(
                    variant, "heteprbcd", 25, attack_seed, train_seed,
                )]
            )
        reference = adaptive[("graph_hard", "adaptive_query", 5)]
        attacked = adaptive[(variant, "adaptive_query", 5)]
        adaptive_gain = (
            float(attacked["attacked_target_micro_f1_mean"])
            - float(reference["attacked_target_micro_f1_mean"])
        )
        result = {
            "variant": variant,
            "max_clean_loss": max(clean_losses),
            "max_heteprbcd_loss": max(poisoning_losses),
            "adaptive_attacked_gain": adaptive_gain,
        }
        result["passes"] = (
            result["max_clean_loss"] <= 0.015 + 1e-12
            and result["max_heteprbcd_loss"] <= 0.02 + 1e-12
            and adaptive_gain >= 0.05 - 1e-12
        )
        candidates.append(result)
    passed = [row for row in candidates if row["passes"]]
    selected = (
        max(passed, key=lambda row: row["adaptive_attacked_gain"])["variant"]
        if passed else "graph_hard"
    )
    return {
        "reference_variant": "graph_hard",
        "selected_variant": selected,
        "method_change": selected != "graph_hard",
        "candidates": candidates,
        "next_action": (
            "rerun_main_protocols_with_selected_variant"
            if selected != "graph_hard"
            else "retain_graph_hard_and_start_f3"
        ),
    }


def render_report(training_summary, adaptive_summary, decision, audit):
    training = {
        (row["variant"], row["attack"], int(row["rate"])): row
        for row in training_summary
    }
    adaptive = {
        (row["variant"], int(row["rate"])): row
        for row in adaptive_summary
        if row["attack"] == "adaptive_query"
    }
    lines = [
        "# DVCL 拓扑实现版本 Pilot",
        "",
        "## 实验设置",
        "",
        "- 数据集：DBLP；配对重复 $(s_a,s_t)=(1,1),(2,2),(3,3)$。",
        "- 变体：当前 `graph_hard` 与取消第二级硬阈值的 `graph_no_filter`。",
        r"- 条件：clean、HetePRBCD $r=25\%$ poisoning、每模型独立优化的 64+64 候选自适应目标逃逸。",
        r"- 预算：$\Delta=\{1,3,5\}$；$E_{max}=200$，$P=100$；仅报告 Micro-F1。",
        r"- 两个变体均令 $\lambda_h=1$，其余模型、训练和攻击超参数保持一致。",
        "",
        "## Poisoning 与 Clean",
        "",
        "| Variant | Clean | HetePRBCD 25% | Drop |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        clean = training[(variant, "clean", 0)]
        attacked = training[(variant, "heteprbcd", 25)]
        lines.append(
            f"| {VARIANT_LABELS[variant]} | {_score(clean)} | "
            f"{_score(attacked)} | "
            f"{_points(clean['micro_f1_mean'] - attacked['micro_f1_mean'])} |"
        )
    lines.extend([
        "",
        "## 自适应目标逃逸",
        "",
        r"| Variant | Clean target | $\Delta=1$ | $\Delta=3$ | $\Delta=5$ | Drop@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for variant in VARIANTS:
        rows = [adaptive[(variant, rate)] for rate in (1, 3, 5)]
        rate5 = rows[-1]
        lines.append(
            f"| {VARIANT_LABELS[variant]} | "
            f"{_percent(rate5['clean_target_micro_f1_mean'])} | "
            + " | ".join(
                _score_value(
                    row["attacked_target_micro_f1_mean"],
                    row["attacked_target_micro_f1_std"],
                )
                for row in rows
            )
            + f" | {_points(rate5['micro_f1_drop_mean'])} |"
        )
    no_filter_gains = {
        rate: (
            adaptive[("graph_no_filter", rate)][
                "attacked_target_micro_f1_mean"
            ]
            - adaptive[("graph_hard", rate)][
                "attacked_target_micro_f1_mean"
            ]
        )
        for rate in (1, 3, 5)
    }
    lines.extend([
        "",
        "## 结果分析",
        "",
        "- `graph_no_filter` 的低预算优势未随预算保持：相对 "
        f"`graph_hard`，$\\Delta=1,3,5$ 的攻击后 Micro-F1 差异依次为 "
        f"{_points(no_filter_gains[1])}、{_points(no_filter_gains[3])}、"
        f"{_points(no_filter_gains[5])}。",
        "- 高预算下取消第二级硬过滤反而扩大下降，且 HetePRBCD 的最大配对损失超过预注册门槛；证据不支持用 `graph_no_filter` 替换当前实现。",
        "- 后续敏感性实验固定使用 `graph_hard`；`han_semantic` 继续只作为研究开关，不混入同架构超参数比较。",
    ])
    lines.extend([
        "",
        "## 版本判定",
        "",
        "| Candidate | 最大 Clean 损失 | 最大 HetePRBCD 损失 | Adaptive@5 增益 | 通过 |",
        "|---|---:|---:|---:|:---:|",
    ])
    for candidate in decision["candidates"]:
        lines.append(
            f"| `{candidate['variant']}` | "
            f"{_points(candidate['max_clean_loss'])} | "
            f"{_points(candidate['max_heteprbcd_loss'])} | "
            f"{_points(candidate['adaptive_attacked_gain'])} | "
            f"{'是' if candidate['passes'] else '否'} |"
        )
    selected = decision["selected_variant"]
    lines.extend([
        "",
        f"- 冻结结论：`{selected}`。",
        "- 门槛：最大 clean 损失不超过 1.5 pp、最大 HetePRBCD 损失不超过 2 pp、Adaptive@5 攻击后 Micro-F1 至少提升 5 pp。",
        f"- 完整性：训练 {audit['training_physical_runs']}，自适应搜索 {audit['adaptive_physical_runs']}，逻辑预算结果 {audit['adaptive_logical_results']}，问题数 {len(audit['issues'])}。",
        "",
    ])
    return "\n".join(lines)


def _input_audit_ok(path, issues):
    if not path.is_file():
        issues.append(f"missing input audit: {path}")
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    ok = (
        summary.get("failed") == 0
        and summary.get("passed") == summary.get("total")
    )
    if not ok:
        issues.append(f"input audit failed: {path}")
    return ok


def _score(row):
    return _score_value(row["micro_f1_mean"], row["micro_f1_std"])


def _score_value(mean, std):
    return f"{100 * float(mean):.2f} ± {100 * float(std):.2f}"


def _percent(value):
    return f"{100 * float(value):.2f}"


def _points(value):
    return f"{100 * float(value):.2f} pp"


def _resolve(path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def main():
    args = parse_args()
    train_config = _resolve(args.train_config)
    adaptive_config = _resolve(args.adaptive_config)
    output_root = _resolve(args.output_root)
    (
        training_rows, training_issues, training_expected, training_completed,
        training_manifests,
    ) = collect_training_rows(train_config)
    training_issues.extend(validate_training_matrix(training_rows))
    (
        adaptive_rows, per_target, adaptive_issues, adaptive_expected,
        adaptive_completed, adaptive_logical_expected, adaptive_manifests,
    ) = collect_evaluation_rows(adaptive_config)
    embedded = [row["_issue"] for row in adaptive_rows if "_issue" in row]
    adaptive_rows = [row for row in adaptive_rows if "_issue" not in row]
    adaptive_issues.extend(embedded)
    issues = training_issues + adaptive_issues
    _input_audit_ok(_resolve(args.train_input_audit), issues)
    _input_audit_ok(_resolve(args.adaptive_input_audit), issues)
    manifests = training_manifests + adaptive_manifests
    manifest_summary = _manifest_summary(manifests)
    if manifest_summary["dirty_manifests"]:
        issues.append(
            f"dirty manifests: {manifest_summary['dirty_manifests']}"
        )
    if len(manifest_summary["manifest_git_commits"]) != 1:
        issues.append(
            "expected one manifest commit, found "
            f"{manifest_summary['manifest_git_commits']}"
        )
    complete = (
        training_completed == training_expected
        and len(training_rows) == training_expected
        and adaptive_completed == adaptive_expected
        and len(adaptive_rows) == adaptive_logical_expected
        and not issues
    )
    training_summary = summarize_training(training_rows)
    adaptive_summary = aggregate_rows(adaptive_rows)
    decision = (
        topology_version_decision(training_rows, adaptive_summary)
        if complete else None
    )
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "training_runs.csv", training_rows)
    write_csv(output_root / "training_summary.csv", training_summary)
    write_csv(output_root / "adaptive_runs.csv", adaptive_rows)
    write_csv(output_root / "adaptive_summary.csv", adaptive_summary)
    write_csv(output_root / "adaptive_per_target.csv", per_target)
    audit = {
        "train_config": str(train_config),
        "adaptive_config": str(adaptive_config),
        "training_physical_runs": f"{training_completed}/{training_expected}",
        "adaptive_physical_runs": f"{adaptive_completed}/{adaptive_expected}",
        "adaptive_logical_results": (
            f"{len(adaptive_rows)}/{adaptive_logical_expected}"
        ),
        "issues": issues,
        **manifest_summary,
        "ok": complete,
    }
    (output_root / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if decision is not None:
        (output_root / "decision.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report = _resolve(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            render_report(training_summary, adaptive_summary, decision, audit),
            encoding="utf-8",
        )
    print(
        f"training={training_completed}/{training_expected} "
        f"adaptive={adaptive_completed}/{adaptive_expected} "
        f"logical={len(adaptive_rows)}/{adaptive_logical_expected} "
        f"issues={len(issues)}"
    )
    return 0 if complete or args.allow_partial else 1


if __name__ == "__main__":
    raise SystemExit(main())
