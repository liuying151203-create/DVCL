import pytest

torch = pytest.importorskip("torch")

from dvcl_bench import environment


def test_cpu_device_is_explicit():
    assert environment.resolve_device("cpu") == torch.device("cpu")


def test_cuda_request_does_not_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA device requested"):
        environment.resolve_device("cuda:0")


def test_runtime_environment_contains_audit_fields():
    report = environment.runtime_environment()
    assert report["python"]["version"]
    assert report["platform"]["system"]
    assert "torch" in report["packages"]
    assert "cuda_available" in report["accelerator"]
    assert "dgl_backend" in report["accelerator"]


def test_cpu_environment_profile_reports_checks():
    report = environment.validate_environment("cpu")
    names = {item["name"] for item in report["checks"]}
    assert "python_version" in names
    assert "package:torch" in names
    assert "torch_import" in names
