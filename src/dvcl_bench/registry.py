"""Native model adapter registry."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict

from .adapters import train_dvcl, train_hseco
from .training import DVCLTrainConfig, HSeCoTrainConfig


NATIVE_MODELS = {
    "hseco": (HSeCoTrainConfig, train_hseco),
    "dvcl": (DVCLTrainConfig, train_dvcl),
}


def build_model_config(name: str, values: Dict[str, Any]):
    try:
        config_type, _ = NATIVE_MODELS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported native model: {name}") from exc
    allowed = {item.name for item in fields(config_type)}
    unknown = set(values) - allowed - {"variant"}
    if unknown:
        raise ValueError(f"Unknown {name} configuration fields: {sorted(unknown)}")
    return config_type(**{key: value for key, value in values.items() if key in allowed})


def get_native_trainer(name: str):
    try:
        return NATIVE_MODELS[name][1]
    except KeyError as exc:
        raise ValueError(f"Unsupported native model: {name}") from exc
