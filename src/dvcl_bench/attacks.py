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
    validate_attack_context(clean, split, data)
    provenance = dict(getattr(data, "attack_metadata", {}) or {})
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
        provenance,
    )


def validate_attack_context(
    clean: CleanGraphArtifact,
    split: SplitArtifact,
    source,
) -> None:
    node_types = getattr(source, "node_types", None)
    if node_types is None:
        return
    if clean.predict_ntype not in node_types:
        raise ValueError(f"Attack source is missing target node type: {clean.predict_ntype}")
    store = source[clean.predict_ntype]
    checks = {
        "features": (getattr(store, "x", None), clean.features, torch.float32),
        "labels": (getattr(store, "y", None), clean.labels, torch.long),
        "train_mask": (getattr(store, "train_mask", None), split.train_mask, torch.bool),
        "val_mask": (getattr(store, "val_mask", None), split.val_mask, torch.bool),
        "test_mask": (getattr(store, "test_mask", None), split.test_mask, torch.bool),
    }
    for name, (source_value, expected, dtype) in checks.items():
        if source_value is None:
            continue
        current = torch.as_tensor(source_value).detach().cpu().to(dtype=dtype)
        expected = expected.detach().cpu().to(dtype=dtype)
        if not torch.equal(current, expected):
            raise ValueError(f"Attack source {name} does not match the frozen artifact")


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
    provenance: Optional[Mapping[str, object]] = None,
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
        provenance=dict(provenance or {}),
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
    stats["_global"] = _global_perturbation_stats(clean_adjs, stats)
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
    recomputed, added_edges, deleted_edges = perturbation_diff(
        clean.hete_adjs, attack.perturbed_hete_adjs
    )
    for etype, value in recomputed.items():
        claimed = attack.stats.get(etype)
        if etype == "_global" and claimed is None:
            continue
        if claimed is None:
            issues.append(f"missing stored stats for {etype}")
        elif any(claimed.get(key) != value.get(key) for key in ("n_add", "n_del")):
            issues.append(f"stored perturbation stats mismatch for {etype}")
    budget = _budget_report(attack.attack_rate, recomputed["_global"])
    if attack.target_nodes is None and not budget["ok"]:
        issues.append(
            "global perturbation budget mismatch: "
            f"expected={budget['expected_changes']}, actual={budget['actual_changes']}"
        )
    if attack.target_nodes is not None:
        target = attack.target_nodes.long()
        if target.numel() and (int(target.min()) < 0 or int(target.max()) >= len(clean.labels)):
            issues.append("target node index out of range")
        if target.numel() and not bool(split.test_mask[target].all()):
            issues.append("target nodes are not all in the test split")
    split_stats = split_perturbation_stats(clean, split, added_edges, deleted_edges)
    warnings = _split_concentration_warnings(split_stats, attack.target_nodes)
    return {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "stats": recomputed,
        "budget": budget,
        "split_perturbation": split_stats,
    }


def split_perturbation_stats(clean, split, added_edges, deleted_edges):
    masks = {
        "train": split.train_mask.detach().cpu().numpy().astype(bool),
        "val": split.val_mask.detach().cpu().numpy().astype(bool),
        "test": split.test_mask.detach().cpu().numpy().astype(bool),
    }
    num_nodes = len(clean.labels)
    global_add = np.zeros(num_nodes, dtype=np.int64)
    global_delete = np.zeros(num_nodes, dtype=np.int64)
    relations = {}
    processed = set()
    for source_type, relation, target_type in clean.canonical_etypes:
        reverse = relation[::-1]
        pair = tuple(sorted((relation, reverse)))
        if pair in processed or clean.predict_ntype not in {source_type, target_type}:
            continue
        processed.add(pair)
        added = added_edges.get(relation)
        deleted = deleted_edges.get(relation)
        if added is None or deleted is None:
            continue
        if target_type == clean.predict_ntype:
            added = added.T.tocsr()
            deleted = deleted.T.tocsr()
        if added.shape[0] != num_nodes:
            continue
        additions = np.asarray(added.sum(axis=1)).reshape(-1).astype(np.int64)
        deletions = np.asarray(deleted.sum(axis=1)).reshape(-1).astype(np.int64)
        global_add += additions
        global_delete += deletions
        relations[relation] = _split_change_summary(additions, deletions, masks)
    return {
        "predict_ntype": clean.predict_ntype,
        "relations": relations,
        "_global": _split_change_summary(global_add, global_delete, masks),
    }


def _split_change_summary(additions, deletions, masks):
    changes = additions + deletions
    total_changes = int(changes.sum())
    num_nodes = len(changes)
    result = {"total_changes": total_changes}
    for name, mask in masks.items():
        split_changes = int(changes[mask].sum())
        node_share = float(mask.sum() / max(num_nodes, 1))
        change_share = float(split_changes / max(total_changes, 1))
        result[name] = {
            "nodes": int(mask.sum()),
            "touched_nodes": int(np.count_nonzero(changes[mask])),
            "touched_fraction": float(np.count_nonzero(changes[mask]) / max(mask.sum(), 1)),
            "n_add": int(additions[mask].sum()),
            "n_del": int(deletions[mask].sum()),
            "changes": split_changes,
            "changes_per_node": float(split_changes / max(mask.sum(), 1)),
            "node_share": node_share,
            "change_share": change_share,
            "enrichment": float(change_share / max(node_share, np.finfo(float).eps)),
        }
    return result


def _split_concentration_warnings(split_stats, target_nodes):
    if target_nodes is not None:
        return []
    global_stats = split_stats["_global"]
    train = global_stats["train"]
    if global_stats["total_changes"] < 10 or train["enrichment"] < 5.0:
        return []
    return [
        "global attack changes are highly concentrated on the training split: "
        f"change_share={train['change_share']:.4f}, enrichment={train['enrichment']:.2f}x"
    ]


def _global_perturbation_stats(clean_adjs, relation_stats):
    processed = set()
    selected = []
    for etype in clean_adjs:
        if etype in processed:
            continue
        reverse = etype[::-1]
        selected.append(etype)
        processed.add(etype)
        if reverse in clean_adjs:
            processed.add(reverse)
    clean_edges = sum(int(relation_stats[name]["clean_edges"]) for name in selected)
    perturbed_edges = sum(int(relation_stats[name]["perturbed_edges"]) for name in selected)
    n_add = sum(int(relation_stats[name]["n_add"]) for name in selected)
    n_del = sum(int(relation_stats[name]["n_del"]) for name in selected)
    return {
        "relations": selected,
        "clean_edges": clean_edges,
        "perturbed_edges": perturbed_edges,
        "n_add": n_add,
        "n_del": n_del,
        "actual_rate": (n_add + n_del) / max(clean_edges, 1),
    }


def _budget_report(attack_rate, global_stats):
    clean_edges = int(global_stats["clean_edges"])
    expected_changes = int(clean_edges * _rate_fraction(float(attack_rate)))
    actual_changes = int(global_stats["n_add"] + global_stats["n_del"])
    tolerance = max(1, int(np.ceil(expected_changes * 0.02)))
    shortfall = expected_changes - actual_changes
    return {
        "ok": 0 <= shortfall <= tolerance,
        "expected_changes": expected_changes,
        "actual_changes": actual_changes,
        "shortfall": shortfall,
        "tolerance": tolerance,
        "expected_rate": _rate_fraction(float(attack_rate)),
        "actual_rate": float(global_stats["actual_rate"]),
    }


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
