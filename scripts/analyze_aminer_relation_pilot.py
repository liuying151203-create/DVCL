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

from scripts.analyze_adaptive_pilot import load_run_suite, options, spec_from_command
from dvcl_bench.artifacts import file_sha256, load_attack_artifact
from dvcl_bench.paths import ExperimentLayout


MODELS = ("heterosage", "dvcl")
ATTACKS = ("prbcd", "heteprbcd")
RELATION_SCOPES = ("pa", "pr", "joint")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit and select the AMiner poisoning relation scope."
    )
    parser.add_argument(
        "--config",
        default="configs/protocols/aminer_poisoning_relation_pilot_v1.yaml",
    )
    parser.add_argument(
        "--clean-protocol", default="aminer_poisoning_main_v1"
    )
    parser.add_argument(
        "--output-root",
        default="outputs/analysis/aminer_poisoning_relation_pilot_v1",
    )
    parser.add_argument(
        "--report", default="docs/aminer-poisoning-relation-pilot.md"
    )
    parser.add_argument("--minimum-drop", type=float, default=0.02)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def collect_clean_rows(protocol):
    rows = []
    issues = []
    for model in MODELS:
        run_dir = (
            ROOT / "outputs" / "runs" / protocol / "aminer" / model
            / "default" / "clean" / "rate_0" / "split_seed_1"
            / "attack_seed_1" / "train_seed_1"
        )
        payload = _read_completed(run_dir, issues)
        if payload is None:
            continue
        metrics, manifest = payload
        rows.append({
            "model": model,
            "micro_f1": float(metrics["metrics"]["micro_f1"]),
            "run_dir": str(run_dir.resolve()),
            "git_commit": manifest.get("git_commit"),
            "git_dirty": bool(manifest.get("git_dirty")),
        })
    return rows, issues


def collect_pilot_rows(config_path, clean_rows):
    run_suite = load_run_suite()
    config = run_suite.load_config(config_path)
    commands = list(run_suite.commands(config, sys.executable, ROOT))
    clean = {row["model"]: float(row["micro_f1"]) for row in clean_rows}
    rows = []
    issues = []
    manifests = []
    artifact_cache = {}
    completed = 0
    for command in commands:
        spec = spec_from_command(command)
        run_dir = ExperimentLayout(ROOT).run_dir(spec)
        payload = _read_completed(run_dir, issues)
        if payload is None:
            continue
        completed += 1
        metrics, manifest = payload
        manifests.append(manifest)
        command_options = options(command[2:])
        configured_path = (ROOT / command_options["--attack-path"]).resolve()
        attack_input = manifest.get("inputs", {}).get("attack", {})
        manifest_path = Path(attack_input.get("path", "")).resolve()
        if configured_path != manifest_path:
            issues.append(f"attack input path mismatch: {run_dir}")
            continue
        if not configured_path.is_file():
            issues.append(f"missing attack artifact: {configured_path}")
            continue
        actual_hash = file_sha256(configured_path)
        if attack_input.get("sha256") != actual_hash:
            issues.append(f"attack input hash mismatch: {run_dir}")
            continue
        artifact_info = artifact_cache.get(configured_path)
        if artifact_info is None:
            artifact_info = _audit_artifact(
                configured_path, spec.attack.name, spec.attack.variant,
                int(spec.attack.rate), spec.seeds.attack, issues,
            )
            artifact_cache[configured_path] = artifact_info
        if artifact_info is None:
            continue
        attacked = float(metrics["metrics"]["micro_f1"])
        rows.append({
            "model": spec.model.name,
            "attack": spec.attack.name,
            "relation_scope": spec.attack.variant,
            "rate": int(spec.attack.rate),
            "attack_seed": spec.seeds.attack,
            "train_seed": spec.seeds.train,
            "clean_micro_f1": clean[spec.model.name],
            "attacked_micro_f1": attacked,
            "micro_f1_drop": clean[spec.model.name] - attacked,
            **artifact_info,
            "run_dir": str(run_dir.resolve()),
        })
    return rows, issues, len(commands), completed, manifests


def _read_completed(run_dir, issues):
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "metrics.json"
    manifest_path = run_dir / "manifest.json"
    if not all(path.is_file() for path in (status_path, metrics_path, manifest_path)):
        issues.append(f"missing run output: {run_dir}")
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        issues.append(f"run not completed: {run_dir}")
        return None
    return (
        json.loads(metrics_path.read_text(encoding="utf-8")),
        json.loads(manifest_path.read_text(encoding="utf-8")),
    )


def _audit_artifact(path, attack, relation_scope, rate, seed, issues):
    artifact = load_attack_artifact(path)
    provenance = artifact.provenance
    expected = {
        "dataset": "aminer",
        "attack": "PRBCD" if attack == "prbcd" else "HetePRBCD",
        "relation_scope": relation_scope,
        "rate": rate,
        "seed": seed,
    }
    mismatches = {
        key: {"actual": provenance.get(key), "expected": value}
        for key, value in expected.items() if provenance.get(key) != value
    }
    budget_relations = {
        edge_type[1] for edge_type in provenance.get("budget", [])
        if len(edge_type) == 3
    }
    expected_relations = {
        "pa": {"pa"}, "pr": {"pr"}, "joint": {"pa", "pr"}
    }[relation_scope]
    if budget_relations != expected_relations:
        mismatches["budget"] = {
            "actual": sorted(budget_relations),
            "expected": sorted(expected_relations),
        }
    verification_path = path.with_name("verification.json")
    if not verification_path.is_file():
        issues.append(f"missing verification report: {path}")
        return None
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if not verification.get("ok") or verification.get("issues"):
        mismatches["verification"] = verification.get("issues", ["not ok"])
    budget_report = verification.get("budget", {})
    if not budget_report.get("ok"):
        mismatches["verification_budget"] = budget_report
    if mismatches:
        issues.append(f"invalid attack artifact {path}: {mismatches}")
        return None
    before = float(provenance["surrogate_before"]["test_micro_f1"])
    after = float(provenance["surrogate_after"]["test_micro_f1"])
    return {
        "surrogate_before_micro_f1": before,
        "surrogate_after_micro_f1": after,
        "surrogate_micro_f1_drop": before - after,
        "expected_perturbations": int(provenance["expected_perturbations"]),
        "actual_perturbations": int(provenance["actual_perturbations"]),
        "artifact_sha256": file_sha256(path),
    }


def summarize_scopes(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["relation_scope"]].append(row)
    summary = []
    for relation_scope in RELATION_SCOPES:
        values = grouped.get(relation_scope, [])
        drops = [float(row["micro_f1_drop"]) for row in values]
        surrogate = {
            row["attack"]: float(row["surrogate_micro_f1_drop"])
            for row in values
        }
        summary.append({
            "relation_scope": relation_scope,
            "n": len(values),
            "mean_downstream_micro_f1_drop": (
                statistics.fmean(drops) if drops else None
            ),
            "min_downstream_micro_f1_drop": min(drops) if drops else None,
            "prbcd_surrogate_micro_f1_drop": surrogate.get("prbcd"),
            "heteprbcd_surrogate_micro_f1_drop": surrogate.get("heteprbcd"),
        })
    return summary


def choose_scope(summary, minimum_drop=0.02):
    complete = [
        row for row in summary
        if row["n"] == len(MODELS) * len(ATTACKS)
        and row["mean_downstream_micro_f1_drop"] is not None
    ]
    if len(complete) != len(RELATION_SCOPES):
        return None
    selected = max(
        complete, key=lambda row: row["mean_downstream_micro_f1_drop"]
    )
    surrogate_nonnegative = all(
        float(selected[f"{attack}_surrogate_micro_f1_drop"]) >= -1e-12
        for attack in ATTACKS
    )
    return {
        "selected_relation_scope": selected["relation_scope"],
        "mean_downstream_micro_f1_drop": selected[
            "mean_downstream_micro_f1_drop"
        ],
        "minimum_required_drop": minimum_drop,
        "surrogate_nonnegative": surrogate_nonnegative,
        "passes": (
            selected["mean_downstream_micro_f1_drop"] >= minimum_drop
            and surrogate_nonnegative
        ),
        "next_action": (
            "expand_selected_scope_to_three_attack_seeds"
            if selected["mean_downstream_micro_f1_drop"] >= minimum_drop
            and surrogate_nonnegative
            else "stop_aminer_poisoning_expansion_and_report_weak_attack"
        ),
    }


def render_report(rows, summary, selection, audit):
    by_key = {
        (row["relation_scope"], row["attack"], row["model"]): row
        for row in rows
    }
    clean = {
        model: next(
            float(row["clean_micro_f1"]) for row in rows
            if row["model"] == model
        )
        for model in MODELS
    }
    lines = [
        "# AMiner 中毒攻击关系范围 Pilot",
        "",
        "## 实验设置",
        "",
        "- 攻击：PRBCD、HetePRBCD；全局扰动率 $r=15\\%$，攻击种子 $s_a=1$。",
        "- 关系范围：P–A、P–R、P–A+P–R；三种范围使用相同全局扰动预算。",
        "- 下游模型：HeteroSAGE、DVCL；训练种子 $s_t=1$；指标仅报告 Micro-F1。",
        f"- Clean Micro-F1：HeteroSAGE {_percent(clean['heterosage'])}，DVCL {_percent(clean['dvcl'])}。",
        "- 扩展门槛：最佳公共关系范围的四条件平均下降至少 2 pp，且两种代理模型的下降均非负。",
        "",
        "## 实验结果",
        "",
        "| 范围 | 攻击 | HeteroSAGE attacked / drop | DVCL attacked / drop | Surrogate drop |",
        "|---|---|---:|---:|---:|",
    ]
    labels = {"pa": "P–A", "pr": "P–R", "joint": "P–A+P–R"}
    for scope in RELATION_SCOPES:
        for attack in ATTACKS:
            heterosage = by_key[(scope, attack, "heterosage")]
            dvcl = by_key[(scope, attack, "dvcl")]
            lines.append(
                f"| {labels[scope]} | `{attack}` | "
                f"{_percent(heterosage['attacked_micro_f1'])} / "
                f"{_points(heterosage['micro_f1_drop'])} | "
                f"{_percent(dvcl['attacked_micro_f1'])} / "
                f"{_points(dvcl['micro_f1_drop'])} | "
                f"{_points(heterosage['surrogate_micro_f1_drop'])} |"
            )
    lines.extend([
        "",
        "| 范围 | 四条件平均下降 | 最小下降 | PRBCD surrogate | HetePRBCD surrogate |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in summary:
        lines.append(
            f"| {labels[row['relation_scope']]} | "
            f"{_points(row['mean_downstream_micro_f1_drop'])} | "
            f"{_points(row['min_downstream_micro_f1_drop'])} | "
            f"{_points(row['prbcd_surrogate_micro_f1_drop'])} | "
            f"{_points(row['heteprbcd_surrogate_micro_f1_drop'])} |"
        )
    verdict = "通过" if selection["passes"] else "未通过"
    pa_hete_hs = by_key[("pa", "heteprbcd", "heterosage")]
    pa_hete_dvcl = by_key[("pa", "heteprbcd", "dvcl")]
    lines.extend([
        "",
        "## 结论",
        "",
        f"- 最佳公共关系范围：{labels[selection['selected_relation_scope']]}，平均下降 {_points(selection['mean_downstream_micro_f1_drop'])}。",
        f"- F1 扩展门槛：**{verdict}**。",
        f"- P–A HetePRBCD 在代理模型上下降 {_points(pa_hete_hs['surrogate_micro_f1_drop'])}，"
        f"但在 HeteroSAGE/DVCL 上分别为 {_points(pa_hete_hs['micro_f1_drop'])}/"
        f"{_points(pa_hete_dvcl['micro_f1_drop'])}，说明主要问题是跨模型迁移弱，而不是扰动预算不足。",
        "- 不生成 attack seed 2–3 的正式全矩阵；既有 AMiner poisoning 结果只作描述性结果，不作为强鲁棒性证据。",
        f"- 完整性审计：{audit['completed_physical_runs']}/{audit['expected_physical_runs']}，问题数 {len(audit['issues'])}。",
    ])
    if audit["dirty_manifests"]:
        lines.append(
            f"- {audit['dirty_manifests']} 个 Pilot manifest 来自 dirty worktree；"
            "若门槛通过，正式矩阵须在冻结提交后运行。"
        )
    lines.append("")
    return "\n".join(lines)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _percent(value):
    return f"{100 * float(value):.2f}"


def _points(value):
    return f"{100 * float(value):.2f} pp"


def main():
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    output_root = (ROOT / args.output_root).resolve()
    clean_rows, clean_issues = collect_clean_rows(args.clean_protocol)
    rows, issues, expected, completed, manifests = collect_pilot_rows(
        config_path, clean_rows
    )
    issues = clean_issues + issues
    summary = summarize_scopes(rows)
    selection = choose_scope(summary, args.minimum_drop)
    complete = (
        len(clean_rows) == len(MODELS)
        and completed == expected
        and len(rows) == expected
        and selection is not None
        and not issues
    )
    audit = {
        "config": str(config_path),
        "expected_physical_runs": expected,
        "completed_physical_runs": completed,
        "result_rows": len(rows),
        "dirty_manifests": sum(
            bool(manifest.get("git_dirty")) for manifest in manifests
        ),
        "git_commits": sorted({
            manifest.get("git_commit") for manifest in manifests
            if manifest.get("git_commit")
        }),
        "issues": issues,
        "ok": complete,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "clean.csv", clean_rows)
    write_csv(output_root / "runs.csv", rows)
    write_csv(output_root / "scope_summary.csv", summary)
    (output_root / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if complete:
        (output_root / "selection.json").write_text(
            json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report_path = (ROOT / args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            render_report(rows, summary, selection, audit), encoding="utf-8"
        )
    print(
        f"runs={completed}/{expected} rows={len(rows)} "
        f"issues={len(issues)} complete={complete}"
    )
    return 0 if complete or args.allow_partial else 1


if __name__ == "__main__":
    raise SystemExit(main())
