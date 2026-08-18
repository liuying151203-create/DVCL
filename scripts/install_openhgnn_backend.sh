#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${DVCL_GPU_ENV:-$ROOT/.conda/dvcl-cu121-py39}"
PYTHON="$ENV_ROOT/bin/python"
REVISION="27a483eeb25e5cdfb3be81ab66ba8ef8b3cf73a3"

"$PYTHON" -m pip install -r "$ROOT/requirements-openhgnn.txt"
"$PYTHON" -m pip install --no-deps --no-build-isolation \
  "git+https://github.com/BUPT-GAMMA/OpenHGNN.git@$REVISION"
"$PYTHON" "$ROOT/scripts/check_openhgnn_backend.py"
