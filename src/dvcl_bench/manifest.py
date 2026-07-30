import hashlib
import json
import subprocess
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .environment import package_versions, runtime_environment


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> Optional[str]:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_dirty(root: Path) -> Optional[bool]:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def build_manifest(spec, project_root: Path, inputs: Dict[str, Path]) -> Dict[str, Any]:
    fingerprints = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in inputs.items()
        if path.exists()
    }
    environment = runtime_environment()
    return {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": asdict(spec),
        "inputs": fingerprints,
        "git_commit": git_commit(project_root),
        "git_dirty": git_dirty(project_root),
        "python": environment["python"]["version"],
        "platform": environment["platform"]["description"],
        "packages": package_versions(),
        "environment": environment,
    }


def save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
