import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional


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


def build_manifest(spec, project_root: Path, inputs: Dict[str, Path]) -> Dict[str, Any]:
    fingerprints = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in inputs.items()
        if path.exists()
    }
    return {
        "schema_version": 1,
        "experiment": asdict(spec),
        "inputs": fingerprints,
        "git_commit": git_commit(project_root),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
