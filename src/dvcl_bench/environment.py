"""Runtime environment inspection and device validation."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import sys
from typing import Any, Dict, List, Optional


EXPECTED_PACKAGES = {
    "torch": "2.1.2",
    "torchvision": "0.16.2",
    "torchaudio": "2.1.2",
    "dgl": "1.1.3",
    "torch-geometric": "2.5.3",
    "numpy": "1.26.4",
    "scipy": "1.11.4",
    "pandas": "2.2.2",
    "PyYAML": "6.0.1",
    "scikit-learn": "1.4.2",
}

AUDIT_ONLY_PACKAGES = (
    "openhgnn",
    "ogb",
    "optuna",
    "tensorboard",
    "lmdb",
    "ordered-set",
    "igraph",
)


def package_versions() -> Dict[str, Optional[str]]:
    result = {}
    for name in (*EXPECTED_PACKAGES, *AUDIT_ONLY_PACKAGES):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def runtime_environment() -> Dict[str, Any]:
    versions = package_versions()
    report = {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "description": platform.platform(),
        },
        "packages": versions,
        "accelerator": _accelerator_report(),
        "environment_variables": {
            name: os.environ.get(name)
            for name in ("CUDA_VISIBLE_DEVICES", "DGLBACKEND")
        },
    }
    return report


def validate_environment(profile: str, smoke: bool = False) -> Dict[str, Any]:
    if profile not in {"cpu", "gpu"}:
        raise ValueError("Environment profile must be cpu or gpu")
    runtime = runtime_environment()
    checks: List[Dict[str, Any]] = []
    python_version = tuple(sys.version_info[:3])
    checks.append(_check(
        "python_version",
        (3, 9) <= python_version < (3, 12),
        ">=3.9,<3.12",
        platform.python_version(),
    ))
    for name, expected in EXPECTED_PACKAGES.items():
        actual = runtime["packages"][name]
        checks.append(_check(
            f"package:{name}",
            actual is not None and _public_version(actual) == expected,
            expected,
            actual,
        ))
    accelerator = runtime["accelerator"]
    checks.append(_check(
        "torch_import",
        accelerator["torch_import_ok"],
        True,
        accelerator["torch_import_error"],
    ))
    checks.append(_check(
        "dgl_import",
        accelerator["dgl_import_ok"],
        True,
        accelerator["dgl_import_error"],
    ))
    checks.append(_check(
        "pyg_import",
        accelerator["pyg_import_ok"],
        True,
        accelerator["pyg_import_error"],
    ))
    if profile == "gpu":
        checks.append(_check(
            "cuda_available",
            accelerator["cuda_available"],
            True,
            accelerator["cuda_available"],
        ))
        checks.append(_check(
            "cuda_device_count",
            accelerator["cuda_device_count"] > 0,
            ">=1",
            accelerator["cuda_device_count"],
        ))
        checks.append(_check(
            "torch_cuda_build",
            accelerator["torch_cuda_version"] == "12.1",
            "12.1",
            accelerator["torch_cuda_version"],
        ))
        dgl_version = runtime["packages"]["dgl"]
        checks.append(_check(
            "dgl_cuda_build",
            dgl_version is not None and "+cu121" in dgl_version,
            "1.1.3+cu121",
            dgl_version,
        ))
    if smoke and all(item["ok"] for item in checks if item["name"].endswith("_import")):
        checks.append(_smoke_check(profile))
    return {
        "ok": all(item["ok"] for item in checks),
        "profile": profile,
        "checks": checks,
        "runtime": runtime,
    }


def resolve_device(requested: str):
    torch = importlib.import_module("torch")
    device = torch.device(requested)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested ({requested}) but torch.cuda.is_available() is False. "
            "Use --device cpu only for explicit CPU validation runs."
        )
    index = torch.cuda.current_device() if device.index is None else device.index
    count = torch.cuda.device_count()
    if index < 0 or index >= count:
        raise RuntimeError(
            f"CUDA device index {index} is out of range for {count} visible device(s)"
        )
    return torch.device("cuda", index)


def _accelerator_report() -> Dict[str, Any]:
    result = {
        "torch_import_ok": False,
        "torch_import_error": None,
        "torch_cuda_version": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cudnn_version": None,
        "devices": [],
        "dgl_import_ok": False,
        "dgl_import_error": None,
        "dgl_backend": None,
        "pyg_import_ok": False,
        "pyg_import_error": None,
    }
    try:
        torch = importlib.import_module("torch")
        result["torch_import_ok"] = True
        result["torch_cuda_version"] = torch.version.cuda
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_device_count"] = int(torch.cuda.device_count())
        result["cudnn_version"] = torch.backends.cudnn.version()
        if result["cuda_available"]:
            for index in range(result["cuda_device_count"]):
                properties = torch.cuda.get_device_properties(index)
                result["devices"].append({
                    "index": index,
                    "name": properties.name,
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory": int(properties.total_memory),
                })
    except Exception as exc:
        result["torch_import_error"] = f"{type(exc).__name__}: {exc}"
    try:
        dgl = importlib.import_module("dgl")
        result["dgl_import_ok"] = True
        result["dgl_backend"] = dgl.backend.backend_name
    except Exception as exc:
        result["dgl_import_error"] = f"{type(exc).__name__}: {exc}"
    try:
        importlib.import_module("torch_geometric")
        result["pyg_import_ok"] = True
    except Exception as exc:
        result["pyg_import_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _smoke_check(profile: str) -> Dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
        dgl = importlib.import_module("dgl")
        graph_conv = importlib.import_module("dgl.nn.pytorch").GraphConv
        pyg_conv = importlib.import_module("torch_geometric.nn").GCNConv
        device = resolve_device("cuda:0" if profile == "gpu" else "cpu")
        source = torch.tensor([0, 1, 2, 3], device=device)
        target = torch.tensor([1, 2, 3, 0], device=device)
        features = torch.randn(4, 3, device=device, requires_grad=True)
        graph = dgl.graph((source, target), num_nodes=4, device=device)
        dgl_model = graph_conv(3, 2, allow_zero_in_degree=True).to(device)
        edge_index = torch.stack((source, target))
        pyg_model = pyg_conv(3, 2).to(device)
        loss = dgl_model(graph, features).sum() + pyg_model(features, edge_index).sum()
        loss.backward()
        if features.grad is None or not torch.isfinite(features.grad).all():
            raise RuntimeError("Non-finite or missing gradient")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return _check("tensor_dgl_pyg_smoke", True, True, True)
    except Exception as exc:
        return _check(
            "tensor_dgl_pyg_smoke",
            False,
            True,
            f"{type(exc).__name__}: {exc}",
        )


def _public_version(value: str) -> str:
    return value.split("+", 1)[0]


def _check(name: str, ok: bool, expected: Any, actual: Any) -> Dict[str, Any]:
    return {"name": name, "ok": bool(ok), "expected": expected, "actual": actual}
