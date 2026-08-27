import json
import os
import socket
from pathlib import Path

from scripts.run_experiment import existing_run_skip_reason, is_cuda_oom


def test_running_run_is_not_started_twice(tmp_path: Path):
    status = tmp_path / "status.json"
    metrics = tmp_path / "metrics.json"
    status.write_text(json.dumps({"state": "running"}), encoding="utf-8")
    assert existing_run_skip_reason(status, metrics) == "running"
    assert existing_run_skip_reason(status, metrics, force=True) is None


def test_completed_run_requires_metrics_to_skip(tmp_path: Path):
    status = tmp_path / "status.json"
    metrics = tmp_path / "metrics.json"
    status.write_text(json.dumps({"state": "completed"}), encoding="utf-8")
    assert existing_run_skip_reason(status, metrics) is None
    metrics.write_text("{}", encoding="utf-8")
    assert existing_run_skip_reason(status, metrics) == "completed"


def test_dead_local_running_process_is_recoverable(tmp_path: Path):
    status = tmp_path / "status.json"
    metrics = tmp_path / "metrics.json"
    status.write_text(json.dumps({
        "state": "running",
        "pid": 2**31 - 1,
        "hostname": socket.gethostname(),
    }), encoding="utf-8")
    assert existing_run_skip_reason(status, metrics) is None


def test_live_local_running_process_is_skipped(tmp_path: Path):
    status = tmp_path / "status.json"
    metrics = tmp_path / "metrics.json"
    status.write_text(json.dumps({
        "state": "running",
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
    }), encoding="utf-8")
    assert existing_run_skip_reason(status, metrics) == "running"


def test_cuda_oom_detection_is_specific():
    assert is_cuda_oom(RuntimeError("CUDA out of memory"))
    assert not is_cuda_oom(RuntimeError("CPU out of memory"))
    assert not is_cuda_oom(RuntimeError("CUDA invalid device"))
