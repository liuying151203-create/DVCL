from dataclasses import dataclass
from typing import Dict


OPENHGNN_BASELINES: Dict[str, str] = {
    "han": "HAN",
    "hgt": "HGT",
    "magnn": "MAGNN",
    "heco": "HeCo",
    "simplehgn": "SimpleHGN",
    "hgsl": "HGSL",
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
    return openhgnn
