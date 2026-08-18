import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.integrations.openhgnn import (
    OPENHGNN_MODEL_HASHES,
    OPENHGNN_REVISION,
    OPENHGNN_VERSION,
    require_openhgnn_model,
)


def main() -> int:
    for name in OPENHGNN_MODEL_HASHES:
        require_openhgnn_model(name)
    print(
        f"OpenHGNN backend verified: version={OPENHGNN_VERSION} "
        f"revision={OPENHGNN_REVISION} models={','.join(OPENHGNN_MODEL_HASHES)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
