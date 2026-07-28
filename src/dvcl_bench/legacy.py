import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from .paths import ExperimentLayout
from .specs import ExperimentSpec


def build_legacy_command(
    spec: ExperimentSpec,
    layout: ExperimentLayout,
    python_bin: Optional[str] = None,
) -> List[str]:
    if spec.model.backend != "legacy":
        raise ValueError("build_legacy_command requires model.backend=legacy")
    runner = layout.legacy_hseco / "scripts" / "run_artifact_experiment.py"
    command = [
        python_bin or sys.executable,
        str(runner),
        "--model",
        spec.model.name,
        "--dataset",
        spec.dataset,
        "--attack",
        spec.attack.name,
        "--rate",
        str(spec.attack.rate),
        "--seed",
        str(spec.seeds.train),
        "--attack-seed",
        str(spec.seeds.attack),
        "--split-name",
        spec.split_name,
        "--data-root",
        str(layout.data),
        "--output-root",
        str(layout.outputs / "legacy"),
        "--device",
        spec.device,
        "--epochs",
        str(spec.epochs),
        "--patience",
        str(spec.patience),
    ]
    for key, value in sorted(spec.model.config.items()):
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(option)
        else:
            command.extend([option, str(value)])
    command.extend(spec.extra_args)
    return command


def run_legacy(command: Iterable[str], legacy_root: Path) -> int:
    if not legacy_root.is_dir():
        raise FileNotFoundError(
            "Private compatibility backend not found. The public repository does not include "
            "the original HSeCo source. Configure DVCL_PRIVATE_HSECO_ROOT only for an authorized "
            "local reference copy, or use a native independently implemented backend."
        )
    result = subprocess.run(list(command), cwd=str(legacy_root), check=False)
    return int(result.returncode)
