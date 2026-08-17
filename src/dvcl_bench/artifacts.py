"""Versioned artifacts independent of legacy HSeCo classes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import scipy.sparse as sp
import torch

SCHEMA_VERSION = 1
CanonicalEType = Tuple[str, str, str]


@dataclass
class CleanGraphArtifact:
    dataset: str
    version: str
    predict_ntype: str
    node_counts: Dict[str, int]
    hete_adjs: Dict[str, sp.csr_matrix]
    features: torch.Tensor
    labels: torch.Tensor
    num_classes: int
    meta_paths: List[List[str]]
    canonical_etypes: List[CanonicalEType]
    stats: Dict[str, Any]


@dataclass
class SplitArtifact:
    dataset: str
    split_name: str
    seed: int
    protocol: str
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    stats: Dict[str, Any]


@dataclass
class AttackArtifact:
    dataset: str
    attack_name: str
    attack_rate: float
    seed: int
    clean_version: str
    split_name: str
    split_seed: int
    perturbed_hete_adjs: Dict[str, sp.csr_matrix]
    added_edges: Dict[str, sp.csr_matrix]
    deleted_edges: Dict[str, sp.csr_matrix]
    target_nodes: Optional[torch.Tensor]
    stats: Dict[str, Any]
    source: str
    source_sha256: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)


def save_clean_artifact(value: CleanGraphArtifact, path: Path) -> None:
    _save("clean", value, Path(path))


def save_split_artifact(value: SplitArtifact, path: Path) -> None:
    _save("split", value, Path(path))


def save_attack_artifact(value: AttackArtifact, path: Path) -> None:
    _save("attack", value, Path(path))


def load_clean_artifact(path: Path) -> CleanGraphArtifact:
    value = _load(Path(path), "clean")
    return value if isinstance(value, CleanGraphArtifact) else clean_from_object(value)


def load_split_artifact(path: Path) -> SplitArtifact:
    value = _load(Path(path), "split")
    return value if isinstance(value, SplitArtifact) else split_from_object(value)


def load_attack_artifact(path: Path) -> AttackArtifact:
    value = _load(Path(path), "attack")
    return value if isinstance(value, AttackArtifact) else attack_from_object(value)


def import_legacy_artifact(source: Path, kind: str, output: Path) -> None:
    converters = {
        "clean": (clean_from_object, save_clean_artifact),
        "split": (split_from_object, save_split_artifact),
        "attack": (attack_from_object, save_attack_artifact),
    }
    try:
        converter, saver = converters[kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported artifact kind: {kind}") from exc
    saver(converter(_torch_load(Path(source))), Path(output))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(kind: str, value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "kind": kind, "payload": asdict(value)}, path)
    meta = _metadata(kind, value)
    meta.update({"schema_version": SCHEMA_VERSION, "artifact_sha256": file_sha256(path)})
    path.with_name("meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load(path: Path, expected_kind: str) -> Any:
    raw = _torch_load(path)
    if not isinstance(raw, Mapping) or "schema_version" not in raw:
        return raw
    if raw.get("kind") != expected_kind:
        raise ValueError(f"Artifact kind mismatch: {raw.get('kind')} != {expected_kind}")
    if int(raw["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported artifact schema: {raw['schema_version']}")
    payload = dict(raw["payload"])
    if expected_kind == "clean":
        payload["canonical_etypes"] = [tuple(item) for item in payload["canonical_etypes"]]
        payload["hete_adjs"] = _csr_dict(payload["hete_adjs"])
    elif expected_kind == "attack":
        for name in ("perturbed_hete_adjs", "added_edges", "deleted_edges"):
            payload[name] = _csr_dict(payload[name])
    constructors = {"clean": CleanGraphArtifact, "split": SplitArtifact, "attack": AttackArtifact}
    return constructors[expected_kind](**payload)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _get(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def clean_from_object(value: Any) -> CleanGraphArtifact:
    hg = _get(value, "hg")
    node_counts = _get(value, "node_counts")
    if node_counts is None and hg is not None:
        node_counts = {ntype: int(hg.num_nodes(ntype)) for ntype in hg.ntypes}
    return CleanGraphArtifact(
        dataset=str(_get(value, "dataset")),
        version=str(_get(value, "version")),
        predict_ntype=str(_get(value, "predict_ntype")),
        node_counts={str(k): int(v) for k, v in dict(node_counts).items()},
        hete_adjs=_csr_dict(_get(value, "hete_adjs")),
        features=_get(value, "features").detach().cpu().float(),
        labels=_get(value, "labels").detach().cpu().long(),
        num_classes=int(_get(value, "num_classes")),
        meta_paths=[list(path) for path in _get(value, "meta_paths")],
        canonical_etypes=[tuple(item) for item in _get(value, "canonical_etypes")],
        stats=dict(_get(value, "stats", {})),
    )


def split_from_object(value: Any) -> SplitArtifact:
    masks = {
        name: _get(value, name).detach().cpu().bool()
        for name in ("train_mask", "val_mask", "test_mask")
    }
    return SplitArtifact(
        dataset=str(_get(value, "dataset")),
        split_name=str(_get(value, "split_name")),
        seed=int(_get(value, "seed", 0)),
        protocol=str(_get(value, "protocol", "imported")),
        train_mask=masks["train_mask"],
        val_mask=masks["val_mask"],
        test_mask=masks["test_mask"],
        train_idx=_index_or_mask(value, "train_idx", masks["train_mask"]),
        val_idx=_index_or_mask(value, "val_idx", masks["val_mask"]),
        test_idx=_index_or_mask(value, "test_idx", masks["test_mask"]),
        stats=dict(_get(value, "stats", {})),
    )


def attack_from_object(value: Any) -> AttackArtifact:
    return AttackArtifact(
        dataset=str(_get(value, "dataset")),
        attack_name=str(_get(value, "attack_name")).lower(),
        attack_rate=float(_get(value, "attack_rate")),
        seed=int(_get(value, "seed")),
        clean_version=str(_get(value, "clean_version")),
        split_name=str(_get(value, "split_name")),
        split_seed=int(_get(value, "split_seed")),
        perturbed_hete_adjs=_csr_dict(_get(value, "perturbed_hete_adjs")),
        added_edges=_csr_dict(_get(value, "added_edges", {})),
        deleted_edges=_csr_dict(_get(value, "deleted_edges", {})),
        target_nodes=_get(value, "target_nodes"),
        stats=dict(_get(value, "stats", {})),
        source=str(_get(value, "source", "legacy_artifact")),
        source_sha256=_get(value, "source_sha256"),
        provenance=dict(_get(value, "provenance", {})),
    )


def _index_or_mask(value: Any, name: str, mask: torch.Tensor) -> torch.Tensor:
    index = _get(value, name)
    return torch.nonzero(mask, as_tuple=False).view(-1) if index is None else index.detach().cpu().long()


def _csr_dict(values: Mapping[str, Any]) -> Dict[str, sp.csr_matrix]:
    return {str(name): matrix.tocsr() for name, matrix in dict(values).items()}


def _metadata(kind: str, value: Any) -> Dict[str, Any]:
    base = {"kind": kind, "dataset": value.dataset}
    if kind == "clean":
        base.update({
            "version": value.version,
            "predict_ntype": value.predict_ntype,
            "node_counts": value.node_counts,
            "num_classes": value.num_classes,
            "meta_paths": value.meta_paths,
            "canonical_etypes": [list(item) for item in value.canonical_etypes],
            "stats": value.stats,
        })
    elif kind == "split":
        base.update({
            "split_name": value.split_name,
            "seed": value.seed,
            "protocol": value.protocol,
            "stats": value.stats,
        })
    else:
        base.update({
            "attack_name": value.attack_name,
            "attack_rate": value.attack_rate,
            "seed": value.seed,
            "clean_version": value.clean_version,
            "split_name": value.split_name,
            "split_seed": value.split_seed,
            "target_nodes": None if value.target_nodes is None else int(value.target_nodes.numel()),
            "source": value.source,
            "source_sha256": value.source_sha256,
            "provenance": value.provenance,
            "stats": value.stats,
        })
    return base
