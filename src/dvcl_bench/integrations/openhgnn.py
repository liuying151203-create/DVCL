import hashlib
import importlib.metadata
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


OPENHGNN_BASELINES: Dict[str, str] = {
    "hgt": "HGT",
    "magnn": "MAGNN",
    "heco": "HeCo",
    "simplehgn": "SimpleHGN",
}

OPENHGNN_REVISION = "27a483eeb25e5cdfb3be81ab66ba8ef8b3cf73a3"
OPENHGNN_VERSION = "0.4.1"
OPENHGNN_MODEL_HASHES = {
    "hgt": "e55478a502bb6f7c5e87c103f2b380b36d2634fa0215b6bb4b4ff2c08f4eb5c8",
    "magnn": "dd468333989036ae5453cb0810249301db9098436d12e956dc1c37b687ee3854",
    "heco": "ea39bf7673664dc8451c61e0f1031c270d8056e89d8fd74ab5135c554f4ea45a",
    "simplehgn": "33a772fff0e5e0c89f1314a94c3d7d3813de2c735b932a98edf415ebaa49997c",
}


@dataclass(frozen=True)
class OpenHGNNBaseline:
    name: str

    @property
    def upstream_name(self) -> str:
        try:
            return OPENHGNN_BASELINES[self.name.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported OpenHGNN baseline: {self.name}") from exc


def require_openhgnn():
    try:
        import openhgnn  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenHGNN is optional. Install the versions pinned for the experiment environment "
            "before enabling the OpenHGNN adapter."
        ) from exc
    version = importlib.metadata.version("openhgnn")
    if version != OPENHGNN_VERSION:
        raise RuntimeError(
            f"OpenHGNN {OPENHGNN_VERSION} is required, found {version}. "
            "Use scripts/install_openhgnn_backend.sh."
        )
    return openhgnn


def require_openhgnn_model(name: str):
    baseline = OpenHGNNBaseline(name)
    require_openhgnn()
    from openhgnn.models import build_model

    model_type = build_model(baseline.upstream_name)
    expected = OPENHGNN_MODEL_HASHES.get(name.lower())
    if expected is not None:
        source_path = Path(inspect.getfile(model_type)).resolve()
        actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"OpenHGNN source hash mismatch for {name}: expected={expected}, actual={actual}"
            )
    return model_type
