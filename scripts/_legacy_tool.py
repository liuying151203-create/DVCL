import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = (
    Path(os.environ["DVCL_PRIVATE_HSECO_ROOT"]).expanduser().resolve()
    if os.environ.get("DVCL_PRIVATE_HSECO_ROOT")
    else ROOT.parent / "private" / "hseco_reference"
)


def main(tool: str) -> int:
    script = LEGACY_ROOT / "scripts" / f"{tool}.py"
    if not script.exists():
        raise FileNotFoundError(
            "This compatibility tool requires an authorized local reference backend, which is "
            "not distributed in the public repository. Set DVCL_PRIVATE_HSECO_ROOT for internal "
            "auditing, or migrate the tool to the native public implementation."
        )
    args = list(sys.argv[1:])
    if "--data-root" not in args:
        args.extend(["--data-root", str(ROOT / "data")])
    command = [sys.executable, str(script), *args]
    return subprocess.run(command, cwd=str(LEGACY_ROOT), check=False).returncode
