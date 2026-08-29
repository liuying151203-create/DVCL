import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict
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
from dvcl_bench.manifest import file_sha256


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
    parser.add_argument(
        "--report",
        default="docs/dvcl-view-diagnosis-results.md",
    )
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def collect_clean_rows(config_path):
    config, commands = _commands(config_path)
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
        row = {
            "dataset": spec.dataset,
            "variant": spec.model.config["variant"],
            "train_seed": spec.seeds.train,
            "full_test_micro_f1": float(payload["metrics"]["micro_f1"]),
            "best_epoch": int(payload["best_epoch"]),
            "run_dir": str(run_dir.resolve()),
        }
        for name, value in payload.get("diagnostics", {}).items():
            if name.startswith("gate_") and isinstance(value, (int, float)):
                row[name] = value
        rows.append(row)
    return rows, issues, len(commands), completed, manifests


def collect_evaluation_rows(config_path):
    config, commands = _commands(config_path)
    evaluation_budgets = [
        int(value) for value in config.get("evaluation_budgets", [])
    ]
    rows = []
    per_target = []
    issues = []
    manifests = []
    hash_cache = {}
    candidate_pool_hashes = defaultdict(set)
    completed = 0
    expected_logical = 0
    for command in commands:
        spec = spec_from_command(command)
        run_dir = ExperimentLayout(ROOT).run_dir(spec)
        budgets = evaluation_budgets if spec.attack.adaptive else [int(spec.attack.rate)]
        expected_logical += len(budgets)
        payload = _completed_payload(
            run_dir, spec, command, issues, manifests, hash_cache
        )
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
        checkpoint_path = _resolve_path(diagnostics.get("checkpoint_source", ""))
        checkpoint_input = manifests[-1]["inputs"].get("checkpoint", {})
        if checkpoint_path != _resolve_path(checkpoint_input.get("path", "")):
            issues.append(f"checkpoint source mismatch: {run_dir}")
        if spec.attack.adaptive:
            adaptive = diagnostics.get("adaptive_attack", {})
            checkpoint_sha256 = checkpoint_input.get("sha256")
            if adaptive.get("victim_checkpoint_sha256") != checkpoint_sha256:
                issues.append(f"adaptive checkpoint hash mismatch: {run_dir}")
            candidate_hash = adaptive.get("candidate_pool_sha256")
            if not candidate_hash:
                issues.append(f"missing adaptive candidate hash: {run_dir}")
            else:
                candidate_pool_hashes[(spec.dataset, spec.seeds.attack)].add(
                    candidate_hash
                )
            if spec.attack.variant == "cand_64" and (
                adaptive.get("candidate_additions") != 64
                or adaptive.get("candidate_deletions") != 64
            ):
                issues.append(f"adaptive candidate size mismatch: {run_dir}")
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
    for key, hashes in sorted(candidate_pool_hashes.items()):
        if len(hashes) != 1:
            issues.append(f"candidate pool hash mismatch for {key}: {sorted(hashes)}")
    return (
        rows, per_target, issues, len(commands), completed,
        expected_logical, manifests,
    )


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
        "view_definition": view.get("definition", ""),
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


def stage_e_decision(clean_rows, summary):
    clean = {
        (row["dataset"], row["variant"]): float(row["full_test_micro_f1"])
        for row in clean_rows
    }
    values = {
        (row["dataset"], row["variant"], row["attack"], int(row["rate"])): row
        for row in summary
    }
    reference = values[("dblp", "concat", "adaptive_query", 5)]
    candidates = []
    for variant in ("gate", "gated_concat"):
        attacked = values[("dblp", variant, "adaptive_query", 5)]
        gain = (
            float(attacked["attacked_target_micro_f1_mean"])
            - float(reference["attacked_target_micro_f1_mean"])
        )
        clean_loss = max(
            clean[(dataset, "concat")] - clean[(dataset, variant)]
            for dataset in ("acm", "dblp", "aminer")
        )
        other_attack_loss = max(
            float(values[(dataset, "concat", "adaptive_query", 5)][
                "attacked_target_micro_f1_mean"
            ])
            - float(values[(dataset, variant, "adaptive_query", 5)][
                "attacked_target_micro_f1_mean"
            ])
            for dataset in ("acm", "aminer")
        )
        candidates.append({
            "variant": variant,
            "dblp_gain": gain,
            "max_clean_loss": clean_loss,
            "max_other_attack_loss": other_attack_loss,
        })
    best = max(candidates, key=lambda row: row["dblp_gain"])
    best["passes"] = (
        best["dblp_gain"] >= 0.05
        and best["max_clean_loss"] <= 0.015
        and best["max_other_attack_loss"] <= 0.02
    )
    return best


def render_report(clean_rows, summary, audit=None):
    variants = ("topo", "feat", "concat", "gate", "gated_concat")
    datasets = ("acm", "dblp", "aminer")
    clean = {
        (row["dataset"], row["variant"]): float(row["full_test_micro_f1"])
        for row in clean_rows
    }
    values = {
        (row["dataset"], row["variant"], row["attack"], int(row["rate"])): row
        for row in summary
    }
    lines = [
        "# DVCL 视图失效诊断结果",
        "",
        "## 1. 实验设置",
        "",
        "- 数据集：ACM、DBLP、AMiner；单种子机制 Pilot，$(s_a,s_t)=(1,1)$。",
        "- 模式：`topo`、`feat`、`concat`、`gate`、`gated_concat`。",
        "- 攻击：HG Baseline 迁移攻击与每模型独立优化的 64+64 候选自适应查询攻击。",
        "- 预算：$\\Delta=\\{1,3,5\\}$；结果只报告 Micro-F1。",
        "- 自适应攻击和 HG 使用各自冻结的目标集，绝对 Micro-F1 不跨攻击类型直接比较。",
        "- 最终审计核对实验规格、输入路径、输入 SHA-256 和 victim checkpoint 身份。",
        "",
        "## 2. Clean Micro-F1",
        "",
        "| Variant | ACM | DBLP | AMiner |",
        "|---|---:|---:|---:|",
    ]
    for variant in variants:
        lines.append(
            f"| `{variant}` | "
            + " | ".join(_percent(clean[(dataset, variant)]) for dataset in datasets)
            + " |"
        )
    for attack, title in (
        ("hg_baseline", "HG Baseline 目标逃逸"),
        ("adaptive_query", "模型自适应目标逃逸"),
    ):
        lines.extend(["", f"## {3 if attack == 'hg_baseline' else 4}. {title}", ""])
        for dataset in datasets:
            lines.extend([
                f"### {dataset.upper()}",
                "",
                "| Variant | Clean target | $\\Delta=1$ | $\\Delta=3$ | $\\Delta=5$ | Drop@5 | ASR@5 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ])
            for variant in variants:
                rate_rows = [values[(dataset, variant, attack, rate)] for rate in (1, 3, 5)]
                rate5 = rate_rows[-1]
                lines.append(
                    f"| `{variant}` | {_percent(float(rate5['clean_target_micro_f1_mean']))} | "
                    + " | ".join(
                        _percent(float(row["attacked_target_micro_f1_mean"]))
                        for row in rate_rows
                    )
                    + f" | {_points(float(rate5['micro_f1_drop_mean']))} | "
                    + f"{_percent(float(rate5['attack_success_rate_mean']))} |"
                )
            lines.append("")
    lines.extend([
        "## 5. 视图诊断（$\\Delta=5$）",
        "",
        "| Dataset | Variant | Attack | Topology L2 | Feature L2 | Disagreement clean→attack | Gate clean→attack |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for dataset in datasets:
        for variant in ("concat", "gate", "gated_concat"):
            for attack in ("hg_baseline", "adaptive_query"):
                row = values[(dataset, variant, attack, 5)]
                lines.append(
                    f"| {dataset.upper()} | `{variant}` | `{attack}` | "
                    f"{_number(row.get('drift_topology_l2_mean_mean'))} | "
                    f"{_number(row.get('drift_feature_l2_mean_mean'))} | "
                    f"{_transition(row, 'clean_view_disagreement_rate_mean', 'attacked_view_disagreement_rate_mean', percent=True)} | "
                    f"{_transition(row, 'gate_clean_mean_mean', 'gate_attacked_mean_mean')} |"
                )
    decision = stage_e_decision(clean_rows, summary)
    topo = values[("dblp", "topo", "adaptive_query", 5)]
    feat = values[("dblp", "feat", "adaptive_query", 5)]
    concat = values[("dblp", "concat", "adaptive_query", 5)]
    gate = values[("dblp", "gate", "adaptive_query", 5)]
    gated_concat = values[("dblp", "gated_concat", "adaptive_query", 5)]
    action = (
        f"将 `{decision['variant']}` 扩展到 3 个配对种子"
        if decision["passes"]
        else "进入 `reliability_gate` 单种子机制 Pilot"
    )
    lines.extend([
        "",
        "## 6. 分析与阶段 E 决策",
        "",
        f"- DBLP 自适应攻击下，`topo` 的 Drop@5 为 {_points(float(topo['micro_f1_drop_mean']))}，`feat` 为 {_points(float(feat['micro_f1_drop_mean']))}。",
        "- Feature embedding 漂移为 0 是威胁模型的预期结果：攻击只修改异构结构边，特征 KNN 图和节点特征保持冻结。",
        f"- DBLP 的双视图预测分歧在 `concat` 中由 {_transition(concat, 'clean_view_disagreement_rate_mean', 'attacked_view_disagreement_rate_mean', percent=True)}；但 `gate` 的拓扑权重由 {_transition(gate, 'gate_clean_mean_mean', 'gate_attacked_mean_mean')}，`gated_concat` 由 {_transition(gated_concat, 'gate_clean_mean_mean', 'gate_attacked_mean_mean')}，没有在拓扑失真时显著降低。",
        f"- 最佳已有门控候选为 `{decision['variant']}`：相对 `concat` 的 DBLP 攻击后增益为 {_points(decision['dblp_gain'])}，三数据集最大 clean 损失为 {_points(decision['max_clean_loss'])}，ACM/AMiner 最大攻击后损失为 {_points(decision['max_other_attack_loss'])}。",
        f"- 预注册门槛判定：{'通过' if decision['passes'] else '未通过'}；下一步应{action}。",
        "- 本结果为单种子机制 Pilot，只用于选择阶段 E 路线，不作为论文显著性结论。",
    ])
    if audit and audit.get("dirty_manifests"):
        lines.append(
            f"- {audit['dirty_manifests']} 个运行 manifest 标记为 dirty worktree；"
            "本阶段仅作机制筛选，正式论文证据必须在冻结提交上重跑。"
        )
    lines.append("")
    return "\n".join(lines)


def _percent(value):
    return f"{100 * value:.2f}"


def _points(value):
    return f"{100 * value:.2f} pp"


def _number(value):
    return "—" if value in (None, "") else f"{float(value):.4f}"


def _transition(row, left, right, percent=False):
    if left not in row or right not in row:
        return "—"
    if percent:
        return f"{_percent(float(row[left]))}→{_percent(float(row[right]))}"
    return f"{float(row[left]):.4f}→{float(row[right]):.4f}"


def _commands(config_path):
    run_suite = load_run_suite()
    config = run_suite.load_config(config_path)
    return config, list(run_suite.commands(config, sys.executable, ROOT))


def _completed_payload(run_dir, spec, command, issues, manifests, hash_cache):
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "metrics.json"
    if not status_path.is_file() or not metrics_path.is_file():
        issues.append(f"missing: {run_dir}")
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        issues.append(f"not completed: {run_dir}")
        return None
    manifest = _audit_manifest(run_dir, spec, command, issues, hash_cache)
    if manifest is None:
        return None
    manifests.append(manifest)
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _audit_manifest(run_dir, spec, command, issues, hash_cache):
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        issues.append(f"missing manifest: {run_dir}")
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        issues.append(f"manifest schema mismatch: {run_dir}")
    expected_experiment = json.loads(json.dumps(asdict(spec)))
    actual_experiment = dict(manifest.get("experiment", {}))
    expected_experiment.pop("device", None)
    actual_experiment.pop("device", None)
    if actual_experiment != expected_experiment:
        issues.append(f"manifest experiment mismatch: {run_dir}")
    expected_inputs = _expected_inputs(spec, command)
    actual_inputs = manifest.get("inputs", {})
    if set(actual_inputs) != set(expected_inputs):
        issues.append(f"manifest input set mismatch: {run_dir}")
    for name, expected_path in expected_inputs.items():
        fingerprint = actual_inputs.get(name, {})
        recorded_path = _resolve_path(fingerprint.get("path", ""))
        if recorded_path != expected_path:
            issues.append(f"manifest {name} path mismatch: {run_dir}")
            continue
        if not expected_path.is_file():
            issues.append(f"manifest {name} input missing: {run_dir}")
            continue
        cache_key = str(expected_path)
        if cache_key not in hash_cache:
            hash_cache[cache_key] = file_sha256(expected_path)
        if fingerprint.get("sha256") != hash_cache[cache_key]:
            issues.append(f"manifest {name} hash mismatch: {run_dir}")
    return manifest


def _expected_inputs(spec, command):
    options = {
        command[index]: command[index + 1]
        for index in range(len(command) - 1)
        if command[index].startswith("--")
    }
    layout = ExperimentLayout(ROOT)
    values = {
        "clean": layout.clean_path(spec.dataset),
        "split": layout.split_path(spec.dataset, spec.split_name),
    }
    if spec.attack.name != "clean":
        values["attack"] = options.get(
            "--attack-path",
            layout.attack_path(
                spec.dataset, spec.attack.name, spec.attack.rate,
                spec.seeds.attack,
            ),
        )
    if "--checkpoint-source" in options:
        values["checkpoint"] = options["--checkpoint-source"]
    return {name: _resolve_path(path) for name, path in values.items()}


def _resolve_path(path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _manifest_summary(manifests):
    return {
        "manifest_count": len(manifests),
        "dirty_manifests": sum(bool(value.get("git_dirty")) for value in manifests),
        "manifest_git_commits": sorted({
            str(value.get("git_commit")) for value in manifests
        }),
    }


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
    (
        clean_rows, clean_issues, clean_expected, clean_completed,
        clean_manifests,
    ) = collect_clean_rows(clean_config)
    (
        rows, per_target, issues, expected, completed, expected_logical,
        evaluation_manifests,
    ) = (
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
        **_manifest_summary(clean_manifests + evaluation_manifests),
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
    if audit["ok"]:
        report = (ROOT / args.report).resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            render_report(clean_rows, summary, audit), encoding="utf-8"
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
