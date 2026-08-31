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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.artifacts import file_sha256
from dvcl_bench.paths import ExperimentLayout
from dvcl_bench.specs import (
    AttackSpec, ExperimentSpec, ModelSpec, ProfilingSpec, SeedSpec,
)
from scripts import run_suite
from scripts.analyze_adaptive_pilot import (
    aggregate_rows as aggregate_adaptive_rows,
    collect_rows as collect_adaptive_rows,
)


CONFIG = ROOT / "configs/protocols/model_efficiency_v1.yaml"
ADAPTIVE_CONFIG = ROOT / "configs/protocols/adaptive_target_evasion_v1.yaml"
F3_RUNS = (
    ROOT / "outputs/analysis/dvcl_hyperparameter_sensitivity_v1/physical_runs.csv"
)
OUTPUT_ROOT = ROOT / "outputs/analysis/model_efficiency_v1"
REPORT = ROOT / "docs/model-efficiency-results.md"
HARDWARE_AUDIT = ROOT / "outputs/audits/model_efficiency_v1-hardware.json"
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
DATASET_ORDER = ("acm", "dblp", "aminer")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit and summarize the formal model-efficiency protocol."
    )
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--adaptive-config", default=str(ADAPTIVE_CONFIG))
    parser.add_argument("--f3-runs", default=str(F3_RUNS))
    parser.add_argument("--hardware-audit", default=str(HARDWARE_AUDIT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _options(values):
    output = {}
    index = 0
    while index < len(values):
        value = values[index]
        if (
            value.startswith("--")
            and index + 1 < len(values)
            and not values[index + 1].startswith("--")
        ):
            output[value] = values[index + 1]
            index += 2
        else:
            output[value] = True
            index += 1
    return output


def _spec_from_command(command):
    options = _options(command[2:])
    return ExperimentSpec(
        protocol=options["--protocol"],
        dataset=options["--dataset"],
        split_name=options["--split-name"],
        seeds=SeedSpec(
            int(options["--split-seed"]),
            int(options["--attack-seed"]),
            int(options["--train-seed"]),
        ),
        attack=AttackSpec(
            options["--attack"], float(options["--rate"]),
            options["--threat-model"], options["--scope"],
            bool(options.get("--adaptive", False)),
            options["--attack-variant"],
        ),
        model=ModelSpec(
            options["--model"], options["--backend"],
            json.loads(options["--model-config-json"]),
        ),
        device=options["--device"],
        epochs=int(options["--epochs"]),
        patience=int(options["--patience"]),
        profiling=ProfilingSpec(
            enabled=bool(options.get("--profile-efficiency", False)),
            inference_warmup=int(options["--profile-inference-warmup"]),
            inference_repetitions=int(
                options["--profile-inference-repetitions"]
            ),
        ),
    )


def collect_efficiency_rows(config_path):
    config = run_suite.load_config(config_path)
    commands = list(run_suite.commands(config, sys.executable, ROOT))
    rows = []
    issues = []
    hash_cache = {}
    for command in commands:
        spec = _spec_from_command(command)
        run_dir = ExperimentLayout(ROOT).run_dir(spec)
        required = {
            name: run_dir / name for name in (
                "manifest.json", "metrics.json", "history.csv", "checkpoint.pt",
                "status.json",
            )
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            issues.append({"run_dir": str(run_dir), "missing": missing})
            continue
        try:
            status = json.loads(required["status.json"].read_text(encoding="utf-8"))
            if status.get("state") != "completed":
                raise ValueError(f"status is {status.get('state')!r}")
            manifest = json.loads(
                required["manifest.json"].read_text(encoding="utf-8")
            )
            metrics = json.loads(
                required["metrics.json"].read_text(encoding="utf-8")
            )
            if manifest.get("experiment") != asdict(spec):
                raise ValueError("manifest experiment spec mismatch")
            for fingerprint in manifest.get("inputs", {}).values():
                path = Path(fingerprint["path"])
                current = hash_cache.get(path)
                if current is None:
                    current = file_sha256(path)
                    hash_cache[path] = current
                if current != fingerprint["sha256"]:
                    raise ValueError(f"input SHA-256 mismatch: {path}")
            efficiency = metrics.get("diagnostics", {}).get("efficiency")
            if not isinstance(efficiency, dict):
                raise ValueError("missing efficiency diagnostics")
            row = _efficiency_row(spec, manifest, metrics, efficiency, run_dir)
            rows.append(row)
        except Exception as exc:
            issues.append({"run_dir": str(run_dir), "error": str(exc)})
    return config, commands, rows, issues


def _efficiency_row(spec, manifest, metrics, efficiency, run_dir):
    expected = spec.profiling
    if efficiency.get("scope") != (
        "trainer_pipeline_excluding_profile_repetitions"
    ):
        raise ValueError("unexpected training timing scope")
    if efficiency.get("inference_warmup") != expected.inference_warmup:
        raise ValueError("inference warmup mismatch")
    if (
        efficiency.get("inference_repetitions")
        != expected.inference_repetitions
    ):
        raise ValueError("inference repetition mismatch")
    positive = (
        "trainable_parameters", "total_parameters", "parameter_bytes",
        "training_seconds", "training_iterations", "seconds_per_iteration",
        "inference_latency_ms_mean", "inference_latency_ms_median",
        "peak_allocated_bytes", "peak_reserved_bytes",
    )
    for key in positive:
        if not isinstance(efficiency.get(key), (int, float)) or efficiency[key] <= 0:
            raise ValueError(f"invalid positive efficiency field: {key}")
    device = str(efficiency["device"])
    if device != spec.device:
        raise ValueError(f"profile device mismatch: {device} != {spec.device}")
    device_index = int(device.split(":", 1)[1])
    devices = manifest["environment"]["accelerator"]["devices"]
    gpu = next(item for item in devices if item["index"] == device_index)
    return {
        "dataset": spec.dataset,
        "model": spec.model.name,
        "train_seed": spec.seeds.train,
        "micro_f1": float(metrics["metrics"]["micro_f1"]),
        "trainable_parameters": int(efficiency["trainable_parameters"]),
        "total_parameters": int(efficiency["total_parameters"]),
        "parameter_bytes": int(efficiency["parameter_bytes"]),
        "training_seconds": float(efficiency["training_seconds"]),
        "training_iterations": int(efficiency["training_iterations"]),
        "seconds_per_iteration": float(efficiency["seconds_per_iteration"]),
        "inference_latency_ms_mean": float(
            efficiency["inference_latency_ms_mean"]
        ),
        "inference_latency_ms_std_within": float(
            efficiency["inference_latency_ms_std"]
        ),
        "inference_latency_ms_median": float(
            efficiency["inference_latency_ms_median"]
        ),
        "peak_allocated_bytes": int(efficiency["peak_allocated_bytes"]),
        "peak_reserved_bytes": int(efficiency["peak_reserved_bytes"]),
        "device": device,
        "device_name": gpu["name"],
        "git_commit": manifest.get("git_commit"),
        "git_dirty": manifest.get("git_dirty"),
        "run_dir": str(run_dir.resolve()),
    }


def validate_rows(config, commands, rows, issues, hardware_audit):
    output = list(issues)
    if len(commands) != 99:
        output.append({"error": f"expected 99 commands, got {len(commands)}"})
    if len(rows) != len(commands):
        output.append({"error": f"completed {len(rows)}/{len(commands)} rows"})
    measurement = config.get("measurement", {})
    if measurement.get("concurrency") != 1 or not measurement.get("exclusive_gpu"):
        output.append({"error": "protocol does not require one exclusive GPU"})
    if not hardware_audit.get("ok"):
        output.append({"error": "hardware launch audit did not pass"})
    commits = {row["git_commit"] for row in rows}
    if len(commits) != 1 or None in commits:
        output.append({"error": f"manifest commits are not singular: {commits}"})
    if any(row["git_dirty"] is not False for row in rows):
        output.append({"error": "one or more manifests are dirty"})
    if commits and hardware_audit.get("git_commit") not in commits:
        output.append({"error": "hardware audit commit differs from manifests"})
    devices = {row["device"] for row in rows}
    names = {row["device_name"] for row in rows}
    if len(devices) != 1 or hardware_audit.get("device") not in devices:
        output.append({"error": f"profile devices are inconsistent: {devices}"})
    if len(names) != 1 or hardware_audit.get("gpu", {}).get("name") not in names:
        output.append({"error": f"GPU names are inconsistent: {names}"})
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"])].append(row)
    expected_groups = {
        (dataset, model) for dataset in DATASET_ORDER for model in MODEL_ORDER
    }
    if set(grouped) != expected_groups:
        output.append({"error": "dataset/model coverage mismatch"})
    for key, values in grouped.items():
        if len(values) != 3 or {row["train_seed"] for row in values} != {1, 2, 3}:
            output.append({"condition": list(key), "error": "seed coverage mismatch"})
        parameters = {row["trainable_parameters"] for row in values}
        if len(parameters) != 1:
            output.append({
                "condition": list(key),
                "error": f"parameter count changes across seeds: {parameters}",
            })
    return output


def summarize_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"])].append(row)
    output = []
    metrics = (
        "micro_f1", "training_seconds", "training_iterations",
        "seconds_per_iteration", "inference_latency_ms_mean",
        "peak_allocated_bytes", "peak_reserved_bytes",
    )
    for (dataset, model), values in sorted(grouped.items()):
        row = {
            "dataset": dataset,
            "model": model,
            "n": len(values),
            "trainable_parameters": values[0]["trainable_parameters"],
            "parameter_millions": values[0]["trainable_parameters"] / 1_000_000,
        }
        for metric in metrics:
            samples = [float(value[metric]) for value in values]
            row[f"{metric}_mean"] = statistics.fmean(samples)
            row[f"{metric}_std"] = (
                statistics.stdev(samples) if len(samples) > 1 else 0.0
            )
        row["peak_allocated_mib_mean"] = (
            row["peak_allocated_bytes_mean"] / 1024 ** 2
        )
        row["peak_allocated_mib_std"] = (
            row["peak_allocated_bytes_std"] / 1024 ** 2
        )
        output.append(row)
    return output


def query_cost_rows(config_path):
    rows, issues, expected, completed = collect_adaptive_rows(config_path)
    if issues or expected != 99 or completed != 99 or len(rows) != 297:
        raise ValueError(
            "formal adaptive query source is incomplete: "
            f"physical={completed}/{expected} logical={len(rows)}/297 "
            f"issues={issues}"
        )
    summary = aggregate_adaptive_rows(rows)
    for row in summary:
        row["queries_per_target_mean"] = row["queries_mean"] / 50.0
        row["queries_per_target_std"] = row["queries_std"] / 50.0
    return summary


def head_capacity_rows(f3_runs_path, expected_k4_parameters):
    import torch

    with f3_runs_path.open(newline="", encoding="utf-8-sig") as stream:
        physical = list(csv.DictReader(stream))
    variants = {1: "heads_1", 2: "heads_2", 4: "reference", 8: "heads_8"}
    output = []
    for heads, variant in variants.items():
        selected = [
            row for row in physical
            if row["variant"] == variant and row["attack"] == "clean"
        ]
        if len(selected) != 3:
            raise ValueError(f"F3 K={heads} checkpoint coverage is {len(selected)}/3")
        counts = set()
        byte_counts = set()
        for row in selected:
            checkpoint = Path(row["run_dir"]) / "checkpoint.pt"
            payload = torch.load(checkpoint, map_location="cpu")
            state = payload["state_dict"]
            counts.add(sum(value.numel() for value in state.values()))
            byte_counts.add(sum(
                value.numel() * value.element_size() for value in state.values()
            ))
        if len(counts) != 1 or len(byte_counts) != 1:
            raise ValueError(f"F3 K={heads} state size changes across seeds")
        output.append({
            "heads": heads,
            "state_elements": counts.pop(),
            "state_bytes": byte_counts.pop(),
        })
    k4 = next(row for row in output if row["heads"] == 4)
    if k4["state_elements"] != expected_k4_parameters:
        raise ValueError(
            "F3 K=4 state elements do not match F4 trainable parameters: "
            f"{k4['state_elements']} != {expected_k4_parameters}"
        )
    reference = k4["state_elements"]
    for row in output:
        row["relative_to_k4"] = row["state_elements"] / reference
    return output


def render_report(summary, queries, capacity, audit):
    lookup = {(row["dataset"], row["model"]): row for row in summary}
    query_lookup = {
        (row["dataset"], row["model"], int(row["rate"])): row
        for row in queries
    }
    lines = [
        "# 模型效率与资源实验",
        "",
        "## 实验设置",
        "",
        "- 数据集：ACM、DBLP、AMiner；统一 11 模型；clean 条件；$s_t=1,2,3$。",
        "- 硬件：单张独占 Tesla V100 串行运行；$E_{max}=200$，$P=100$。",
        "- 推理：静态图预处理完成后执行完整图前向，预热 10 次并同步测量 50 次。",
        "- 训练时间包含训练器内部预处理、模型构建、优化、早停、最佳模型恢复与一次测试；不含 artifact 磁盘读取和额外 profiling 前向。",
        "- 峰值显存为 CUDA allocated memory；查询成本复用正式自适应攻击，仅报告查询次数，不报告不可公平比较的旧墙钟时间。",
        "- 准确率列仅报告 Micro-F1。",
        "",
        "## 模型效率",
        "",
    ]
    for dataset in DATASET_ORDER:
        lines.extend([
            f"### {dataset.upper()}",
            "",
            "| Model | Params (M) | Train (s) | s/iter | Inference (ms) | Peak GPU (MiB) | Micro-F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for model in MODEL_ORDER:
            row = lookup[(dataset, model)]
            lines.append(
                f"| {MODEL_LABELS[model]} | {row['parameter_millions']:.3f} | "
                f"{row['training_seconds_mean']:.2f} ± {row['training_seconds_std']:.2f} | "
                f"{row['seconds_per_iteration_mean']:.4f} ± {row['seconds_per_iteration_std']:.4f} | "
                f"{row['inference_latency_ms_mean_mean']:.3f} ± {row['inference_latency_ms_mean_std']:.3f} | "
                f"{row['peak_allocated_mib_mean']:.1f} ± {row['peak_allocated_mib_std']:.1f} | "
                f"{100 * row['micro_f1_mean']:.2f} ± {100 * row['micro_f1_std']:.2f} |"
            )
        lines.append("")
    lines.extend([
        "## 自适应攻击查询成本",
        "",
        "下表报告 $\\Delta=5$ 时每个目标节点的平均 victim-model 查询次数；每个单元聚合三个配对种子。",
        "",
        "| Model | ACM | DBLP | AMiner |",
        "|---|---:|---:|---:|",
    ])
    for model in MODEL_ORDER:
        cells = []
        for dataset in DATASET_ORDER:
            row = query_lookup[(dataset, model, 5)]
            cells.append(
                f"{row['queries_per_target_mean']:.1f} ± "
                f"{row['queries_per_target_std']:.1f}"
            )
        lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## DVCL 多头容量",
        "",
        "| $K$ | State parameters (M) | Relative to $K=4$ |",
        "|---:|---:|---:|",
    ])
    for row in capacity:
        lines.append(
            f"| {row['heads']} | {row['state_elements'] / 1_000_000:.3f} | "
            f"{row['relative_to_k4']:.2f}× |"
        )
    lines.extend([
        "",
        "## 审计结论",
        "",
        f"- 完整性：{audit['completed']}/{audit['expected']} 次物理运行，问题数 {len(audit['issues'])}。",
        f"- Git 提交：`{audit['manifest_git_commits'][0]}`；dirty manifest 为 {audit['dirty_manifests']}。",
        f"- 计时设备：`{audit['devices'][0]}`，{audit['device_names'][0]}。",
        "- 参数量、时间、延迟和显存只在同一数据集内横向解释；不同模型的预处理与早停步数均属于其真实算法开销。",
        "",
    ])
    return "\n".join(lines)


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config_path = _resolve(args.config)
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    hardware_path = _resolve(args.hardware_audit)
    hardware = (
        json.loads(hardware_path.read_text(encoding="utf-8"))
        if hardware_path.is_file() else {"ok": False}
    )
    config, commands, rows, collection_issues = collect_efficiency_rows(
        config_path
    )
    issues = validate_rows(config, commands, rows, collection_issues, hardware)
    audit = {
        "config": str(config_path.resolve()),
        "hardware_audit": str(hardware_path.resolve()),
        "expected": len(commands),
        "completed": len(rows),
        "issues": issues,
        "manifest_count": len(rows),
        "dirty_manifests": sum(row["git_dirty"] is not False for row in rows),
        "manifest_git_commits": sorted({
            row["git_commit"] for row in rows if row["git_commit"]
        }),
        "devices": sorted({row["device"] for row in rows}),
        "device_names": sorted({row["device_name"] for row in rows}),
        "ok": not issues and len(rows) == len(commands) == 99,
    }
    (output_root / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not audit["ok"]:
        print(
            f"completed={len(rows)}/{len(commands)} issues={len(issues)}"
        )
        return 0 if args.allow_partial else 1
    summary = summarize_rows(rows)
    queries = query_cost_rows(_resolve(args.adaptive_config))
    dvcl_parameters = next(
        row["trainable_parameters"] for row in summary
        if row["dataset"] == "dblp" and row["model"] == "dvcl"
    )
    capacity = head_capacity_rows(_resolve(args.f3_runs), dvcl_parameters)
    _write_csv(output_root / "physical_runs.csv", rows)
    _write_csv(output_root / "model_efficiency_summary.csv", summary)
    _write_csv(output_root / "adaptive_query_cost.csv", queries)
    _write_csv(output_root / "dvcl_head_capacity.csv", capacity)
    report = _resolve(args.report)
    report.write_text(
        render_report(summary, queries, capacity, audit), encoding="utf-8"
    )
    print(f"completed={len(rows)}/{len(commands)} issues=0")
    print(f"Wrote {output_root} and {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
