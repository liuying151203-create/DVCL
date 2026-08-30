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
    write_csv,
)
from dvcl_bench.paths import ExperimentLayout


FACTOR_ORDER = ("lambda_h", "lambda_d", "tau", "k", "heads")
FACTOR_SPECS = {
    "lambda_h": {
        "config_key": "lambda_han",
        "default": 1.0,
        "values": (0.0, 0.1, 0.5, 1.0, 2.0),
        "label": r"$\lambda_h$",
    },
    "lambda_d": {
        "config_key": "lambda_dvcl",
        "default": 1.0,
        "values": (0.0, 0.1, 0.5, 1.0, 2.0),
        "label": r"$\lambda_d$",
    },
    "tau": {
        "config_key": "temperature",
        "default": 0.5,
        "values": (0.1, 0.2, 0.5, 1.0, 2.0),
        "label": r"$\tau$",
    },
    "k": {
        "config_key": "knn_k",
        "default": 20.0,
        "values": (5.0, 10.0, 20.0, 40.0, 80.0),
        "label": r"$k$",
    },
    "heads": {
        "config_key": "heads",
        "default": 4.0,
        "values": (1.0, 2.0, 4.0, 8.0),
        "label": r"$K$",
    },
}
CONDITIONS = (("clean", 0), ("heteprbcd", 15))
SEED_PAIRS = ((1, 1), (2, 2), (3, 3))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit and aggregate the DVCL hyperparameter sensitivity study."
    )
    parser.add_argument(
        "--config",
        default="configs/protocols/dvcl_hyperparameter_sensitivity_v1.yaml",
    )
    parser.add_argument(
        "--input-audit",
        default="outputs/audits/dvcl_hyperparameter_sensitivity_v1-inputs.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/analysis/dvcl_hyperparameter_sensitivity_v1",
    )
    parser.add_argument(
        "--report", default="docs/dvcl-hyperparameter-sensitivity.md"
    )
    parser.add_argument(
        "--figure",
        default="docs/figures/paper/dvcl_hyperparameter_sensitivity.png",
    )
    parser.add_argument("--skip-figure", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def variant_definitions(config, base_config):
    variants = config.get("variants", [])
    issues = []
    definitions = {}
    references = [row for row in variants if row.get("factor") == "reference"]
    if len(references) != 1:
        issues.append(f"expected one reference variant, found {len(references)}")
    for variant in variants:
        name = variant.get("name")
        if not name or name in definitions:
            issues.append(f"missing or duplicate variant name: {name}")
            continue
        factor = variant.get("factor")
        overrides = variant.get("model_config", {})
        if factor == "reference":
            if overrides:
                issues.append("reference variant must not override model config")
            definitions[name] = {"factor": factor, "value": None}
            continue
        if factor not in FACTOR_SPECS:
            issues.append(f"unsupported sensitivity factor: {factor}")
            continue
        spec = FACTOR_SPECS[factor]
        value = float(variant.get("value"))
        if value == float(spec["default"]):
            issues.append(f"default value must use reference variant: {name}")
        expected_nondefault = {
            float(item) for item in spec["values"]
            if float(item) != float(spec["default"])
        }
        if value not in expected_nondefault:
            issues.append(f"unexpected {factor} value {value}: {name}")
        if set(overrides) != {spec["config_key"]}:
            issues.append(f"variant must override only {spec['config_key']}: {name}")
        elif float(overrides[spec["config_key"]]) != value:
            issues.append(f"variant value mismatch: {name}")
        definitions[name] = {"factor": factor, "value": value}
    expected_count = 1 + sum(len(spec["values"]) - 1 for spec in FACTOR_SPECS.values())
    if len(definitions) != expected_count:
        issues.append(
            f"variant count mismatch: expected={expected_count} actual={len(definitions)}"
        )
    for factor, spec in FACTOR_SPECS.items():
        if float(base_config[spec["config_key"]]) != float(spec["default"]):
            issues.append(f"base default mismatch for {factor}")
        actual = {
            row["value"] for row in definitions.values()
            if row["factor"] == factor
        }
        expected = {
            float(item) for item in spec["values"]
            if float(item) != float(spec["default"])
        }
        if actual != expected:
            issues.append(
                f"factor values mismatch for {factor}: expected={sorted(expected)} "
                f"actual={sorted(actual)}"
            )
    return definitions, issues


def collect_rows(config_path):
    run_suite = load_run_suite()
    config = run_suite.load_config(config_path)
    base_config = run_suite.resolve_model_config({
        "config_path": config.get("model_config_path"),
        "config": config.get("model_config", {}),
    }, ROOT)
    definitions, issues = variant_definitions(config, base_config)
    commands = list(run_suite.commands(config, sys.executable, ROOT))
    rows = []
    manifests = []
    hash_cache = {}
    completed = 0
    layout = ExperimentLayout(ROOT)
    for command in commands:
        experiment = spec_from_command(command)
        run_dir = layout.run_dir(experiment)
        payload = _completed_payload(
            run_dir, experiment, command, issues, manifests, hash_cache
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
        variant = experiment.model.config["variant"]
        definition = definitions.get(variant)
        if definition is None:
            issues.append(f"unknown completed variant: {variant}")
            continue
        expected_config = dict(base_config)
        if definition["factor"] != "reference":
            factor_spec = FACTOR_SPECS[definition["factor"]]
            expected_config[factor_spec["config_key"]] = definition["value"]
        actual_config = {
            key: value for key, value in experiment.model.config.items()
            if key != "variant"
        }
        if actual_config != expected_config:
            issues.append(f"model config is not one-factor-only: {run_dir}")
        diagnostics = payload.get("diagnostics", {})
        if diagnostics.get("topology_source") != "graph":
            issues.append(f"topology source mismatch: {run_dir}")
        if diagnostics.get("semantic_topology_filter") != "hard":
            issues.append(f"semantic filter mismatch: {run_dir}")
        micro_f1 = float(payload["metrics"]["micro_f1"])
        if abs(micro_f1 - float(payload["metrics"]["accuracy"])) > 1e-12:
            issues.append(f"accuracy and Micro-F1 differ: {run_dir}")
        rows.append({
            "variant": variant,
            "factor": definition["factor"],
            "value": definition["value"],
            "attack": experiment.attack.name,
            "rate": int(experiment.attack.rate),
            "attack_seed": experiment.seeds.attack,
            "train_seed": experiment.seeds.train,
            "micro_f1": micro_f1,
            "best_epoch": int(payload["best_epoch"]),
            "run_dir": str(run_dir.resolve()),
        })
    return rows, issues, len(commands), completed, manifests


def validate_matrix(rows):
    expected_variants = {"reference"}
    for factor, spec in FACTOR_SPECS.items():
        expected_variants.update(
            (factor, float(value))
            for value in spec["values"]
            if float(value) != float(spec["default"])
        )
    expected = set()
    for variant in expected_variants:
        for attack, rate in CONDITIONS:
            for attack_seed, train_seed in SEED_PAIRS:
                expected.add((variant, attack, rate, attack_seed, train_seed))
    actual = set()
    for row in rows:
        variant = (
            "reference" if row["factor"] == "reference"
            else (row["factor"], float(row["value"]))
        )
        actual.add((
            variant, row["attack"], row["rate"],
            row["attack_seed"], row["train_seed"],
        ))
    issues = []
    if missing := sorted(expected - actual, key=str):
        issues.append(f"missing sensitivity rows: {missing[:10]} count={len(missing)}")
    if unexpected := sorted(actual - expected, key=str):
        issues.append(
            f"unexpected sensitivity rows: {unexpected[:10]} "
            f"count={len(unexpected)}"
        )
    if len(actual) != len(rows):
        issues.append(f"duplicate sensitivity rows: rows={len(rows)} unique={len(actual)}")
    return issues


def expand_reference_rows(rows):
    expanded = []
    for row in rows:
        if row["factor"] != "reference":
            expanded.append(dict(row))
            continue
        for factor in FACTOR_ORDER:
            value = dict(row)
            value["factor"] = factor
            value["value"] = float(FACTOR_SPECS[factor]["default"])
            expanded.append(value)
    return expanded


def summarize_rows(rows):
    grouped = defaultdict(list)
    for row in expand_reference_rows(rows):
        grouped[(
            row["factor"], float(row["value"]), row["attack"], row["rate"],
        )].append(float(row["micro_f1"]))
    summary = []
    for key, values in sorted(
        grouped.items(), key=lambda item: (
            FACTOR_ORDER.index(item[0][0]), item[0][1], item[0][2],
        )
    ):
        summary.append({
            "factor": key[0],
            "value": key[1],
            "attack": key[2],
            "rate": key[3],
            "n": len(values),
            "micro_f1_mean": statistics.fmean(values),
            "micro_f1_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    return summary


def paired_effects(rows):
    reference = {
        (row["attack"], row["rate"], row["attack_seed"], row["train_seed"]): row
        for row in rows if row["factor"] == "reference"
    }
    grouped = defaultdict(list)
    for row in rows:
        if row["factor"] == "reference":
            continue
        key = (row["attack"], row["rate"], row["attack_seed"], row["train_seed"])
        if key not in reference:
            continue
        grouped[(
            row["factor"], float(row["value"]), row["attack"], row["rate"],
        )].append(float(row["micro_f1"]) - float(reference[key]["micro_f1"]))
    output = []
    for key, values in sorted(grouped.items()):
        output.append({
            "factor": key[0],
            "value": key[1],
            "attack": key[2],
            "rate": key[3],
            "n": len(values),
            "paired_effect_mean": statistics.fmean(values),
            "paired_effect_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    return output


def stability_assessment(summary):
    lookup = {
        (row["factor"], float(row["value"]), row["attack"], row["rate"]): row
        for row in summary
    }
    output = []
    for factor in FACTOR_ORDER:
        spec = FACTOR_SPECS[factor]
        values = [float(value) for value in spec["values"]]
        default = float(spec["default"])
        default_index = values.index(default)
        neighbors = [values[default_index - 1], values[default_index + 1]]
        local_losses = []
        best_gaps = []
        for attack, rate in CONDITIONS:
            reference = lookup[(factor, default, attack, rate)]["micro_f1_mean"]
            local_losses.extend(
                reference - lookup[(factor, value, attack, rate)]["micro_f1_mean"]
                for value in neighbors
            )
            best = max(
                lookup[(factor, value, attack, rate)]["micro_f1_mean"]
                for value in values
            )
            best_gaps.append(best - reference)
        max_local_loss = max(0.0, max(local_losses))
        output.append({
            "factor": factor,
            "default": default,
            "neighbors": neighbors,
            "max_local_loss": max_local_loss,
            "max_gap_to_best": max(best_gaps),
            "locally_stable": max_local_loss <= 0.02 + 1e-12,
        })
    return output


def render_report(summary, stability, audit):
    lookup = {
        (row["factor"], float(row["value"]), row["attack"], row["rate"]): row
        for row in summary
    }
    lines = [
        "# DVCL 超参数敏感性实验",
        "",
        "## 实验设置",
        "",
        "- 数据集：DBLP；最终模型固定为 `concat + graph_hard`。",
        r"- 条件：clean 与 HetePRBCD $r=15\%$；配对重复 $(s_a,s_t)=(1,1),(2,2),(3,3)$。",
        r"- 单因素取值：$\lambda_h,\lambda_d\in\{0,0.1,0.5,1,2\}$，$\tau\in\{0.1,0.2,0.5,1,2\}$，$k\in\{5,10,20,40,80\}$，$K\in\{1,2,4,8\}$。",
        r"- 参照值：$(\lambda_h,\lambda_d,\tau,k,K)=(1,1,0.5,20,4)$；$E_{max}=200$，$P=100$；仅报告 Micro-F1。",
        "- `配对 Δ` 为相同 $(s_a,s_t)$ 下当前取值相对公共参照的 Micro-F1 均值差。",
        "- 本实验只描述敏感性，不根据结果修改已冻结的论文主模型。",
        "",
        "## 实验结果",
        "",
        "| Parameter | Value | Clean | 配对 Δ | HetePRBCD 15% | 配对 Δ | Drop |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for factor in FACTOR_ORDER:
        default = float(FACTOR_SPECS[factor]["default"])
        clean_reference = lookup[(factor, default, "clean", 0)]
        attacked_reference = lookup[(factor, default, "heteprbcd", 15)]
        for value in FACTOR_SPECS[factor]["values"]:
            clean = lookup[(factor, float(value), "clean", 0)]
            attacked = lookup[(factor, float(value), "heteprbcd", 15)]
            lines.append(
                f"| {FACTOR_SPECS[factor]['label']} | {_value(value)} | "
                f"{_score(clean)} | "
                f"{_points(clean['micro_f1_mean'] - clean_reference['micro_f1_mean'])} | "
                f"{_score(attacked)} | "
                f"{_points(attacked['micro_f1_mean'] - attacked_reference['micro_f1_mean'])} | "
                f"{_points(clean['micro_f1_mean'] - attacked['micro_f1_mean'])} |"
            )
    lines.extend([
        "",
        "## 稳定性分析",
        "",
        "| Parameter | Reference | Local neighbors | Max local loss | Gap to best | Stable |",
        "|---|---:|---:|---:|---:|:---:|",
    ])
    for row in stability:
        lines.append(
            f"| {FACTOR_SPECS[row['factor']]['label']} | "
            f"{_value(row['default'])} | "
            f"{', '.join(_value(value) for value in row['neighbors'])} | "
            f"{_points(row['max_local_loss'])} | "
            f"{_points(row['max_gap_to_best'])} | "
            f"{'是' if row['locally_stable'] else '否'} |"
        )
    unstable = [
        FACTOR_SPECS[row["factor"]]["label"]
        for row in stability if not row["locally_stable"]
    ]
    analysis = (
        "五个参数在参照值邻域内均未出现超过 2 pp 的最坏损失。"
        if not unstable else
        f"局部最坏损失超过 2 pp 的参数为：{', '.join(unstable)}。"
    )
    lines.extend([
        "",
        f"- {analysis}",
        "- `Gap to best` 仅用于描述参照值与本轮最佳观测值的距离，不作为重新选择超参数的依据。",
        f"- 完整性：物理运行 {audit['physical_runs']}，manifest {audit['manifest_count']}，dirty manifest {audit['dirty_manifests']}，问题数 {len(audit['issues'])}。",
        "",
    ])
    return "\n".join(lines)


def write_figure(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {
        (row["factor"], float(row["value"]), row["attack"], row["rate"]): row
        for row in summary
    }
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for axis, factor in zip(axes.flat, FACTOR_ORDER):
        values = [float(value) for value in FACTOR_SPECS[factor]["values"]]
        for attack, rate, label, color in (
            ("clean", 0, "Clean", "#3568a8"),
            ("heteprbcd", 15, "HetePRBCD 15%", "#c94f45"),
        ):
            selected = [lookup[(factor, value, attack, rate)] for value in values]
            axis.errorbar(
                values,
                [100 * row["micro_f1_mean"] for row in selected],
                yerr=[100 * row["micro_f1_std"] for row in selected],
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=label,
                color=color,
            )
        axis.axvline(
            float(FACTOR_SPECS[factor]["default"]),
            color="#777777", linestyle="--", linewidth=1,
        )
        axis.set_title(FACTOR_SPECS[factor]["label"])
        axis.set_xticks(values)
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    axes[0, 0].set_ylabel("Micro-F1 (%)")
    axes[1, 0].set_ylabel("Micro-F1 (%)")
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _input_audit_ok(path, issues):
    if not path.is_file():
        issues.append(f"missing input audit: {path}")
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    ok = summary.get("failed") == 0 and summary.get("passed") == summary.get("total")
    if not ok:
        issues.append(f"input audit failed: {path}")
    return ok


def _score(row):
    return (
        f"{100 * float(row['micro_f1_mean']):.2f} ± "
        f"{100 * float(row['micro_f1_std']):.2f}"
    )


def _points(value):
    return f"{100 * float(value):.2f} pp"


def _value(value):
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _resolve(path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def main():
    args = parse_args()
    config_path = _resolve(args.config)
    rows, issues, expected, completed, manifests = collect_rows(config_path)
    issues.extend(validate_matrix(rows))
    _input_audit_ok(_resolve(args.input_audit), issues)
    manifest_summary = _manifest_summary(manifests)
    if manifest_summary["dirty_manifests"]:
        issues.append(f"dirty manifests: {manifest_summary['dirty_manifests']}")
    if len(manifest_summary["manifest_git_commits"]) != 1:
        issues.append(
            "expected one manifest commit, found "
            f"{manifest_summary['manifest_git_commits']}"
        )
    complete = completed == expected and len(rows) == expected and not issues
    summary = summarize_rows(rows) if rows else []
    effects = paired_effects(rows) if rows else []
    stability = stability_assessment(summary) if complete else []
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "physical_runs.csv", rows)
    write_csv(output_root / "sensitivity_summary.csv", summary)
    write_csv(output_root / "paired_effects.csv", effects)
    audit = {
        "config": str(config_path),
        "physical_runs": f"{completed}/{expected}",
        "issues": issues,
        **manifest_summary,
        "ok": complete,
    }
    (output_root / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if complete:
        (output_root / "stability.json").write_text(
            json.dumps(stability, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report = _resolve(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(render_report(summary, stability, audit), encoding="utf-8")
        if not args.skip_figure:
            write_figure(summary, _resolve(args.figure))
    print(f"physical={completed}/{expected} issues={len(issues)}")
    return 0 if complete or args.allow_partial else 1


if __name__ == "__main__":
    raise SystemExit(main())
