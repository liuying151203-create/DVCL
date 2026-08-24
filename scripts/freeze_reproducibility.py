import sys
if sys.version_info < (3, 9):
    raise SystemExit(
        "DVCL reproducibility freeze requires Python 3.9–3.11. "
        "Run: source scripts/activate_gpu_env.sh"
    )

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Freeze the final DVCL environment, artifacts, results, and Git state."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "reproducibility" / "final_freeze.yaml"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "reproducibility" / "final_manifest.json"),
    )
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    git = git_state()
    if git["dirty"] and not args.allow_dirty:
        raise RuntimeError(
            "Refusing to freeze a dirty worktree; commit the final implementation first "
            "or use --allow-dirty for a non-publication preflight."
        )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_dir = output.parent / "audits"
    audits = [
        audit_protocol(ROOT / path, audit_dir)
        for path in config["protocol_configs"]
    ]
    artifact_files = files_under(config["artifact_roots"])
    analysis_files = files_under(config["analysis_paths"])
    implementation_files = implementation_paths()
    protocol_results = {
        Path(report["config"]).stem: result_digest(Path(report["config"]).stem)
        for report in audits
    }
    manifest = {
        "schema_version": 1,
        "publication_ready": not git["dirty"] and all(report["ok"] for report in audits),
        "git": git,
        "environment": environment_snapshot(),
        "freeze_config": {
            "path": relative(config_path),
            "sha256": sha256_file(config_path),
        },
        "implementation": {
            "files": len(implementation_files),
            "tree_sha256": tree_sha256(implementation_files),
        },
        "protocol_audits": [
            {
                "config": relative(Path(report["config"])),
                "expected": report["expected"],
                "completed": report["completed"],
                "ok": report["ok"],
            }
            for report in audits
        ],
        "protocol_results": protocol_results,
        "artifacts": {
            "count": len(artifact_files),
            "tree_sha256": tree_sha256(artifact_files),
            "files": file_hashes(artifact_files),
        },
        "analysis": {
            "count": len(analysis_files),
            "tree_sha256": tree_sha256(analysis_files),
            "files": file_hashes(analysis_files),
        },
    }
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"publication_ready={manifest['publication_ready']} "
        f"protocols={len(audits)} artifacts={len(artifact_files)} output={output}"
    )
    return 0 if all(report["ok"] for report in audits) else 1


def audit_protocol(config_path: Path, audit_dir: Path):
    audit_dir.mkdir(parents=True, exist_ok=True)
    output = audit_dir / f"{config_path.stem}.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_suite_results.py"),
            "--config", str(config_path),
            "--output", str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def git_state():
    commit = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def git_output(*args):
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def environment_snapshot():
    packages = sorted(
        {
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        },
        key=str.lower,
    )
    snapshot = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "variables": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "DGLBACKEND": os.environ.get("DGLBACKEND"),
        },
    }
    try:
        import dgl
        import torch
        import torch_geometric

        snapshot.update({
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn": torch.backends.cudnn.version(),
            "dgl": dgl.__version__,
            "torch_geometric": torch_geometric.__version__,
            "gpus": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
        })
    except Exception as error:
        snapshot["accelerator_error"] = repr(error)
    try:
        snapshot["nvidia_smi"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            encoding="utf-8",
        ).strip().splitlines()
    except Exception as error:
        snapshot["nvidia_smi_error"] = repr(error)
    return snapshot


def implementation_paths():
    roots = [
        ROOT / "src", ROOT / "scripts", ROOT / "configs", ROOT / "tests",
        ROOT / "docs", ROOT / "pyproject.toml", ROOT / "requirements.txt",
        ROOT / "requirements-cu121.txt", ROOT / "requirements-cpu.txt",
    ]
    return files_under(roots)


def files_under(paths):
    result = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            result.add(path.resolve())
        elif path.is_dir():
            result.update(
                candidate.resolve() for candidate in path.rglob("*")
                if candidate.is_file() and "__pycache__" not in candidate.parts
            )
    return sorted(result, key=lambda path: relative(path))


def file_hashes(paths):
    return [
        {"path": relative(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]


def tree_sha256(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(relative(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def result_digest(protocol):
    root = ROOT / "outputs" / "runs" / protocol
    metrics = sorted(root.rglob("metrics.json")) if root.is_dir() else []
    manifests = sorted(root.rglob("manifest.json")) if root.is_dir() else []
    return {
        "metrics_count": len(metrics),
        "metrics_tree_sha256": tree_sha256(metrics),
        "manifest_count": len(manifests),
        "manifest_tree_sha256": tree_sha256(manifests),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
