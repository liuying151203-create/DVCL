"""Frozen split generation and import."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np
import torch

from .artifacts import CleanGraphArtifact, SplitArtifact, split_from_object


def build_split_artifact(
    clean: CleanGraphArtifact,
    seed: int,
    protocol: str = "paper",
    train_ratio: float = 0.1,
    val_ratio: float = 0.1,
    test_ratio: float = 0.8,
) -> SplitArtifact:
    if protocol == "paper" and clean.dataset == "acm":
        masks = acm_paper_split(clean.labels, clean.num_classes, seed)
    elif protocol in {"paper", "random"}:
        masks = random_split(len(clean.labels), seed, train_ratio, val_ratio, test_ratio)
    else:
        raise ValueError(f"Unsupported split protocol: {protocol}")
    name = f"paper_seed_{seed}" if protocol == "paper" else f"seed_{seed}"
    return _make_split(clean.dataset, name, seed, protocol, *masks)


def import_split_artifact(
    clean: CleanGraphArtifact,
    source: Path,
    split_name: str,
    seed: int = 0,
) -> SplitArtifact:
    raw = _torch_load(Path(source))
    if isinstance(raw, (list, tuple)) and len(raw) >= 6:
        return _validated(
            clean,
            _make_split(
                clean.dataset,
                split_name,
                seed,
                "imported",
                _to_mask(raw[3], len(clean.labels)),
                _to_mask(raw[4], len(clean.labels)),
                _to_mask(raw[5], len(clean.labels)),
            ),
        )
    if not isinstance(raw, Mapping) or any(
        name not in raw for name in ("train_mask", "val_mask", "test_mask")
    ):
        converted = split_from_object(raw)
        converted.dataset = clean.dataset
        converted.split_name = split_name
        converted.protocol = "imported"
        return _validated(clean, converted)
    return _validated(
        clean,
        _make_split(
            clean.dataset,
            split_name,
            seed,
            "imported",
            _to_mask(raw["train_mask"], len(clean.labels)),
            _to_mask(raw["val_mask"], len(clean.labels)),
            _to_mask(raw["test_mask"], len(clean.labels)),
        ),
    )


def random_split(
    num_nodes: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total = train_ratio + val_ratio + test_ratio
    train_size = int(num_nodes * train_ratio / total)
    val_size = int(num_nodes * val_ratio / total)
    indices = np.random.RandomState(seed).permutation(num_nodes)
    return (
        idx_to_mask(num_nodes, indices[:train_size]),
        idx_to_mask(num_nodes, indices[train_size:train_size + val_size]),
        idx_to_mask(num_nodes, indices[train_size + val_size:]),
    )


def acm_paper_split(
    labels: torch.Tensor,
    num_classes: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels_np = labels.detach().cpu().numpy()
    scores = np.zeros(len(labels_np))
    rng = np.random.RandomState(seed)
    for class_id in range(num_classes):
        class_mask = labels_np == class_id
        scores[class_mask] = rng.permutation(np.linspace(0, 1, int(class_mask.sum())))
    return (
        idx_to_mask(len(labels_np), np.where(scores <= 0.2)[0]),
        idx_to_mask(len(labels_np), np.where((scores > 0.2) & (scores <= 0.3))[0]),
        idx_to_mask(len(labels_np), np.where(scores > 0.3)[0]),
    )


def idx_to_mask(num_nodes: int, indices: Any) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[torch.as_tensor(indices, dtype=torch.long)] = True
    return mask


def _to_mask(value: Any, num_nodes: int) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.dtype == torch.bool and tensor.numel() == num_nodes:
        return tensor.clone()
    return idx_to_mask(num_nodes, tensor.long().view(-1))


def _make_split(dataset: str, name: str, seed: int, protocol: str, *masks) -> SplitArtifact:
    train, val, test = (mask.bool().cpu() for mask in masks)
    num_nodes = len(train)
    counts = {"train": int(train.sum()), "val": int(val.sum()), "test": int(test.sum())}
    stats = {
        "num_nodes": num_nodes,
        **{f"num_{key}": value for key, value in counts.items()},
        **{f"actual_{key}_ratio": value / max(num_nodes, 1) for key, value in counts.items()},
    }
    return SplitArtifact(
        dataset=dataset,
        split_name=name,
        seed=seed,
        protocol=protocol,
        train_mask=train,
        val_mask=val,
        test_mask=test,
        train_idx=torch.nonzero(train, as_tuple=False).view(-1),
        val_idx=torch.nonzero(val, as_tuple=False).view(-1),
        test_idx=torch.nonzero(test, as_tuple=False).view(-1),
        stats=stats,
    )


def _validated(clean: CleanGraphArtifact, split: SplitArtifact) -> SplitArtifact:
    masks = [split.train_mask, split.val_mask, split.test_mask]
    if any(len(mask) != len(clean.labels) for mask in masks):
        raise ValueError("Imported split length does not match target node count")
    overlap = masks[0].int() + masks[1].int() + masks[2].int()
    if not torch.all(overlap == 1):
        raise ValueError("Split masks must be disjoint and cover every target node")
    return split


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except (RuntimeError, pickle.UnpicklingError):
        if path.suffix.lower() != ".pkl":
            raise
        with path.open("rb") as stream:
            return pickle.load(stream)
