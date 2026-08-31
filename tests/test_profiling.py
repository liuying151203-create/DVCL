import pytest
import torch

from dvcl_bench.profiling import profile_inference, profile_run
from dvcl_bench.specs import ProfilingSpec


def test_cpu_profile_records_parameters_latency_and_training_scope():
    model = torch.nn.Linear(3, 2)
    inputs = torch.ones(4, 3)
    spec = ProfilingSpec(
        enabled=True, inference_warmup=1, inference_repetitions=3
    )

    with profile_run(spec, "cpu") as profiler:
        profile_inference(model, lambda: model(inputs))

    summary = profiler.summary([{"epoch": 0}, {"epoch": 1}])
    assert summary["trainable_parameters"] == 8
    assert summary["total_parameters"] == 8
    assert summary["training_iterations"] == 2
    assert summary["training_seconds"] > 0
    assert summary["seconds_per_iteration"] > 0
    assert summary["inference_repetitions"] == 3
    assert summary["inference_latency_ms_mean"] > 0
    assert summary["peak_allocated_bytes"] is None


def test_profile_rejects_multiple_model_reports():
    model = torch.nn.Linear(2, 2)
    inputs = torch.ones(1, 2)
    spec = ProfilingSpec(
        enabled=True, inference_warmup=0, inference_repetitions=1
    )

    with pytest.raises(RuntimeError, match="more than one model"):
        with profile_run(spec, "cpu"):
            profile_inference(model, lambda: model(inputs))
            profile_inference(model, lambda: model(inputs))


def test_disabled_profile_requires_zero_measurement_counts():
    with pytest.raises(ValueError, match="disabled profiling"):
        ProfilingSpec(
            enabled=False, inference_warmup=1, inference_repetitions=1
        )
