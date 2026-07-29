"""Native attack artifact generation, import, diffing and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import scipy.sparse as sp
import torch

from .artifacts import (
    AttackArtifact,
    CleanGraphArtifact,
    SplitArtifact,
    file_sha256,
)
from .graph_adapter import pyg_attack_to_adjs


def generate_rnd_attack(
    clean: CleanGraphArtifact,
    split: SplitArtifact,
    attack_rate: float,
    seed: int,
) -> AttackArtifact:
    rng = np.random.RandomState(seed)
    perturbed = {}
    processed = set()
    for etype, value in clean.hete_adjs.items():
        if etype in processed:
            continue
        reverse = etype[::-1]
        attacked = random_flip(value, attack_rate, rng)
        perturbed[etype] = attacked
        if reverse in clean.hete_adjs:
            perturbed[reverse] = attacked.T.tocsr()
        processed.update({etype, reverse})
    for etype, value in clean.hete_adjs.items():
        perturbed.setdefault(etype, value.copy())
    return build_attack_artifact(
        clean, split, "rnd", attack_rate, seed, perturbed, None, "generated_by_dvcl_bench"
    )


def import_prbcd_like_attack(
    clean: CleanGraphArtifact,
    split: SplitArtifact,
    attack_name: str,
    attack_rate: float,
    seed: int,
    source_file: Path,
) -> AttackArtifact:
    source_file = Path(source_file)
    data = _torch_load(source_file)
    imported = pyg_attack_to_adjs(data)
    perturbed = {name: value.copy() for name, value in clean.hete_adjs.items()}
    for etype, value in imported.items():
        reverse = etype[::-1]
        if reverse in imported and value.shape[::-1] == imported[reverse].shape:
            mismatch = _binary(value.T) - _binary(imported[reverse])
            mismatch.eliminate_zeros()
            if mismatch.nnz:
                raise ValueError(f"Imported reverse relations disagree: {etype}/{reverse}")
    processed = set()
    for etype, value in imported.items():
        if etype in processed:
            continue
        if etype not in clean.hete_adjs:
            raise ValueError(f"Imported attack contains unknown relation: {etype}")
        value = _binary(value)
        if value.shape != clean.hete_adjs[etype].shape:
            raise ValueError(f"Imported relation shape mismatch for {etype}")
        perturbed[etype] = value
        reverse = etype[::-1]
        if reverse in clean.hete_adjs:
            perturbed[reverse] = value.T.tocsr()
        processed.update({etype, reverse})
    return build_attack_artifact(
        clean,
        split,
        attack_name.lower(),
        attack_rate,
        seed,
        perturbed,
        None,
        str(source_file.resolve()),
        file_sha256(source_file),
    )


def build_attack_artifact(
    clean: CleanGraphArtifact,
    split: SplitArtifact,
    attack_name: str,
    attack_rate: float,
    seed: int,
    perturbed: Mapping[str, sp.spmatrix],
    target_nodes: Optional[torch.Tensor],
    source: str,
    source_sha256: Optional[str] = None,
) -> AttackArtifact:
    normalized = {name: _binary(value) for name, value in perturbed.items()}
    stats, added, deleted = perturbation_diff(clean.hete_adjs, normalized)
    return AttackArtifact(
        dataset=clean.dataset,
        attack_name=attack_name,
        attack_rate=float(attack_rate),
        seed=int(seed),
        clean_version=clean.version,
        split_name=split.split_name,
        split_seed=split.seed,
        perturbed_hete_adjs=normalized,
        added_edges=added,
        deleted_edges=deleted,
        target_nodes=None if target_nodes is None else target_nodes.detach().cpu().long(),
        stats=stats,
        source=source,
        source_sha256=source_sha256,
    )


def random_flip(
    adjacency: sp.spmatrix,
    attack_rate: float,
    rng: np.random.RandomState,
) -> sp.csr_matrix:
    clean = _binary(adjacency)
    budget = int(clean.nnz * _rate_fraction(attack_rate))
    budget = max(0, min(budget, clean.nnz))
    if not budget:
        return clean.copy()
    num_delete = budget // 2
    num_add = budget - num_delete
    rows, cols = clean.nonzero()
    result = clean.tolil(copy=True)
    for index in rng.choice(clean.nnz, size=num_delete, replace=False):
        result[rows[index], cols[index]] = 0
    existing = set(zip(rows.tolist(), cols.tolist()))
    added = 0
    attempts = 0
    max_attempts = max(100, num_add * 20)
    while added < num_add and attempts < max_attempts:
        row = int(rng.randint(0, clean.shape[0]))
        col = int(rng.randint(0, clean.shape[1]))
        if (row, col) not in existing and result[row, col] == 0:
            result[row, col] = 1
            existing.add((row, col))
            added += 1
        attempts += 1
    return _binary(result)


def perturbation_diff(clean_adjs, perturbed_adjs):
    stats = {}
    added_edges = {}
    deleted_edges = {}
    for etype, clean_value in clean_adjs.items():
        if etype not in perturbed_adjs:
            stats[etype] = {"missing": True}
            continue
        clean = _binary(clean_value)
        perturbed = _binary(perturbed_adjs[etype])
        if clean.shape != perturbed.shape:
            stats[etype] = {"shape_mismatch": [list(clean.shape), list(perturbed.shape)]}
            continue
        added = perturbed - clean
        added.data = (added.data > 0).astype(np.int8)
        added.eliminate_zeros()
        deleted = clean - perturbed
        deleted.data = (deleted.data > 0).astype(np.int8)
        deleted.eliminate_zeros()
        changed = int(added.nnz + deleted.nnz)
        stats[etype] = {
            "clean_edges": int(clean.nnz),
            "perturbed_edges": int(perturbed.nnz),
            "n_add": int(added.nnz),
            "n_del": int(deleted.nnz),
            "actual_rate": changed / max(clean.nnz, 1),
            "missing": False,
        }
        added_edges[etype] = added.tocsr()
        deleted_edges[etype] = deleted.tocsr()
    return stats, added_edges, deleted_edges


def verify_attack(
    clean: CleanGraphArtifact,
    split: SplitArtifact,
    attack: AttackArtifact,
) -> Dict[str, object]:
    issues = []
    if clean.dataset != attack.dataset or split.dataset != attack.dataset:
        issues.append("dataset mismatch between clean, split and attack")
    if clean.version != attack.clean_version:
        issues.append("clean version mismatch")
    if split.split_name != attack.split_name or split.seed != attack.split_seed:
        issues.append("split identity mismatch")
    clean_names = set(clean.hete_adjs)
    attack_names = set(attack.perturbed_hete_adjs)
    if clean_names != attack_names:
        issues.append(
            f"relation set mismatch: missing={sorted(clean_names-attack_names)}, "
            f"extra={sorted(attack_names-clean_names)}"
        )
    for etype, clean_value in clean.hete_adjs.items():
        value = attack.perturbed_hete_adjs.get(etype)
        if value is None:
            continue
        if value.shape != clean_value.shape:
            issues.append(f"shape mismatch for {etype}")
            continue
        if value.nnz != value.astype(bool).nnz or np.any(value.data != 1):
            issues.append(f"duplicate or weighted edges for {etype}")
        reverse = etype[::-1]
        reverse_value = attack.perturbed_hete_adjs.get(reverse)
        if reverse_value is not None and value.shape[::-1] == reverse_value.shape:
            mismatch = _binary(value.T) - _binary(reverse_value)
            mismatch.eliminate_zeros()
            if mismatch.nnz:
                issues.append(f"reverse edge mismatch: {etype}/{reverse}")
    recomputed, _, _ = perturbation_diff(clean.hete_adjs, attack.perturbed_hete_adjs)
    for etype, value in recomputed.items():
        claimed = attack.stats.get(etype)
        if claimed is None:
            issues.append(f"missing stored stats for {etype}")
        elif any(claimed.get(key) != value.get(key) for key in ("n_add", "n_del")):
            issues.append(f"stored perturbation stats mismatch for {etype}")
    if attack.target_nodes is not None:
        target = attack.target_nodes.long()
        if target.numel() and (int(target.min()) < 0 or int(target.max()) >= len(clean.labels)):
            issues.append("target node index out of range")
        if target.numel() and not bool(split.test_mask[target].all()):
            issues.append("target nodes are not all in the test split")
    return {"ok": not issues, "issues": issues, "stats": recomputed}


def _rate_fraction(rate: float) -> float:
    if rate <= 0:
        raise ValueError("Attack rate must be positive")
    return rate / 100.0 if rate > 1 else rate


def _binary(value: sp.spmatrix) -> sp.csr_matrix:
    result = value.tocsr().astype(bool).astype(np.int8)
    result.eliminate_zeros()
    return result


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
