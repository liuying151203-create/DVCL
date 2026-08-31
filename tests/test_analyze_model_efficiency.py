import json
from dataclasses import asdict

from scripts import analyze_model_efficiency as ANALYZER
from scripts import audit_efficiency_hardware as HARDWARE
from scripts import run_suite


def _rows():
    rows = []
    for dataset in ANALYZER.DATASET_ORDER:
        for model_index, model in enumerate(ANALYZER.MODEL_ORDER, start=1):
            for train_seed in (1, 2, 3):
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "train_seed": train_seed,
                    "micro_f1": 0.8 + model_index / 1000,
                    "trainable_parameters": model_index * 1000,
                    "total_parameters": model_index * 1000,
                    "parameter_bytes": model_index * 4000,
                    "training_seconds": 10.0 + train_seed,
                    "training_iterations": 20,
                    "seconds_per_iteration": (10.0 + train_seed) / 20,
                    "inference_latency_ms_mean": 1.0 + train_seed / 10,
                    "inference_latency_ms_std_within": 0.01,
                    "inference_latency_ms_median": 1.0,
                    "peak_allocated_bytes": model_index * 1024 ** 2,
                    "peak_reserved_bytes": model_index * 2 * 1024 ** 2,
                    "device": "cuda:4",
                    "device_name": "Tesla V100-PCIE-32GB",
                    "git_commit": "abc123",
                    "git_dirty": False,
                    "run_dir": "unused",
                })
    return rows


def test_formal_matrix_summary_and_invariants_cover_all_models():
    rows = _rows()
    config = {"measurement": {"concurrency": 1, "exclusive_gpu": True}}
    hardware = {
        "ok": True,
        "device": "cuda:4",
        "gpu": {"name": "Tesla V100-PCIE-32GB"},
        "git_commit": "abc123",
    }
    assert ANALYZER.validate_rows(
        config, [None] * 99, rows, [], hardware
    ) == []
    summary = ANALYZER.summarize_rows(rows)
    assert len(summary) == 33
    assert all(row["n"] == 3 for row in summary)
    assert all(row["parameter_millions"] > 0 for row in summary)


def test_report_uses_micro_f1_and_query_counts_only():
    summary = ANALYZER.summarize_rows(_rows())
    queries = []
    for dataset in ANALYZER.DATASET_ORDER:
        for model in ANALYZER.MODEL_ORDER:
            queries.append({
                "dataset": dataset,
                "model": model,
                "rate": 5,
                "queries_per_target_mean": 100.0,
                "queries_per_target_std": 2.0,
            })
    capacity = [
        {"heads": heads, "state_elements": heads * 1000,
         "relative_to_k4": heads / 4}
        for heads in (1, 2, 4, 8)
    ]
    audit = {
        "completed": 99,
        "expected": 99,
        "issues": [],
        "manifest_git_commits": ["abc123"],
        "dirty_manifests": 0,
        "devices": ["cuda:4"],
        "device_names": ["Tesla V100-PCIE-32GB"],
    }
    report = ANALYZER.render_report(summary, queries, capacity, audit)
    assert "Micro-F1" in report
    assert "Macro" not in report
    assert "查询次数" in report
    assert "## 结果分析" in report
    assert "不能表述为全面更高效" in report


def test_suite_commands_embed_manifested_profile_spec():
    config = run_suite.load_config(ANALYZER.CONFIG)
    command = next(run_suite.commands(config, "python", ANALYZER.ROOT))
    spec = ANALYZER._spec_from_command(command)
    assert spec.profiling.enabled
    assert spec.profiling.inference_warmup == 10
    assert spec.profiling.inference_repetitions == 50


def test_hardware_device_overrides_protocol_default_for_manifest_audit():
    config = ANALYZER.load_efficiency_config(ANALYZER.CONFIG, "cuda:4")
    command = next(run_suite.commands(config, "python", ANALYZER.ROOT))
    spec = ANALYZER._spec_from_command(command)
    assert spec.device == "cuda:4"


def test_manifest_spec_is_compared_after_json_normalization():
    config = ANALYZER.load_efficiency_config(ANALYZER.CONFIG, "cuda:4")
    command = next(run_suite.commands(config, "python", ANALYZER.ROOT))
    spec = ANALYZER._spec_from_command(command)
    persisted = json.loads(json.dumps(asdict(spec)))
    assert persisted["extra_args"] == []
    assert persisted != asdict(spec)


def test_hardware_device_parser_rejects_non_cuda_values():
    assert HARDWARE.parse_device_index("cuda:6") == 6
    try:
        HARDWARE.parse_device_index("cpu")
    except ValueError:
        pass
    else:
        raise AssertionError("CPU was accepted for the GPU hardware audit")
