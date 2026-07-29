"""Golden comparisons for legacy-to-native migration audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch


def compare_clean(reference, current) -> Dict[str, object]:
    issues = []
    for name in ("dataset", "predict_ntype", "num_classes", "node_counts", "meta_paths"):
        if getattr(reference, name) != getattr(current, name):
            issues.append(f"clean field mismatch: {name}")
    if not torch.equal(reference.features, current.features):
        issues.append("feature tensor mismatch")
    if not torch.equal(reference.labels, current.labels):
        issues.append("label tensor mismatch")
    issues.extend(_compare_adjs(reference.hete_adjs, current.hete_adjs, "clean"))
    return {"ok": not issues, "issues": issues}


def compare_split(reference, current) -> Dict[str, object]:
    issues = []
    for name in ("train_mask", "val_mask", "test_mask"):
        if not torch.equal(getattr(reference, name).bool(), getattr(current, name).bool()):
            issues.append(f"split mask mismatch: {name}")
    return {"ok": not issues, "issues": issues}


def compare_attack(reference, current) -> Dict[str, object]:
    issues = []
    for name in ("dataset", "attack_name", "attack_rate", "seed"):
        if getattr(reference, name) != getattr(current, name):
            issues.append(f"attack field mismatch: {name}")
    issues.extend(_compare_adjs(
        reference.perturbed_hete_adjs, current.perturbed_hete_adjs, "attack"
    ))
    return {"ok": not issues, "issues": issues}


def compare_metrics(reference_path: Path, current_path: Path, tolerance: float):
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    current = json.loads(Path(current_path).read_text(encoding="utf-8"))
    reference = reference.get("metrics", reference)
    current = current.get("metrics", current)
    issues = []
    differences = {}
    for name in ("accuracy", "micro_f1", "macro_f1"):
        if name not in reference or name not in current:
            issues.append(f"missing metric: {name}")
            continue
        difference = abs(float(reference[name]) - float(current[name]))
        differences[name] = difference
        if difference > tolerance:
            issues.append(f"{name} differs by {difference:.6f} > {tolerance:.6f}")
    return {"ok": not issues, "issues": issues, "differences": differences}


def _compare_adjs(reference, current, label):
    issues = []
    if set(reference) != set(current):
        issues.append(f"{label} relation set mismatch")
        return issues
    for name in reference:
        if reference[name].shape != current[name].shape:
            issues.append(f"{label} shape mismatch: {name}")
        elif (reference[name] != current[name]).nnz:
            issues.append(f"{label} adjacency mismatch: {name}")
    return issues
