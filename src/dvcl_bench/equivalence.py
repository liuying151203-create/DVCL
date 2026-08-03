"""Golden comparisons for legacy-to-native migration audits."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict

import torch


LEGACY_VALIDATION_RE = re.compile(
    r"^\s*(?P<epoch>\d+)\s+\|VAL Micro-F1:\s*(?P<micro>[^,]+),\s*"
    r"Macro-F1:\s*(?P<macro>\S+)"
)
LEGACY_TEST_RE = re.compile(
    r"@@@@test:\s*(?P<accuracy>\S+)\s+(?P<micro>\S+)\s+(?P<macro>\S+)"
)


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


def compare_legacy_training_log(
    reference_path: Path,
    current_history_path: Path,
    current_metrics_path: Path,
    tolerance: float,
):
    reference_history, reference_metrics = parse_legacy_training_log(reference_path)
    with Path(current_history_path).open(newline="", encoding="utf-8") as stream:
        current_history = list(csv.DictReader(stream))
    current_payload = json.loads(Path(current_metrics_path).read_text(encoding="utf-8"))
    current_metrics = current_payload.get("metrics", current_payload)
    issues = []
    if len(reference_history) != len(current_history):
        issues.append(
            f"history length mismatch: {len(reference_history)} != {len(current_history)}"
        )
    max_differences = {"micro_f1": 0.0, "macro_f1": 0.0}
    for reference, current in zip(reference_history, current_history):
        if int(current["epoch"]) != reference["epoch"]:
            issues.append(
                f"epoch mismatch: {reference['epoch']} != {int(current['epoch'])}"
            )
            continue
        for name in ("micro_f1", "macro_f1"):
            difference = abs(reference[name] - float(current[f"val_{name}"]))
            max_differences[name] = max(max_differences[name], difference)
    metric_differences = {}
    for name in ("accuracy", "micro_f1", "macro_f1"):
        difference = abs(reference_metrics[name] - float(current_metrics[name]))
        metric_differences[name] = difference
        if difference > tolerance:
            issues.append(f"final {name} differs by {difference:.6f} > {tolerance:.6f}")
    for name, difference in max_differences.items():
        if difference > tolerance:
            issues.append(f"validation {name} differs by {difference:.6f} > {tolerance:.6f}")
    return {
        "ok": not issues,
        "issues": issues,
        "epochs_compared": min(len(reference_history), len(current_history)),
        "max_validation_differences": max_differences,
        "final_metric_differences": metric_differences,
    }


def parse_legacy_training_log(path: Path):
    history = []
    metrics = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        validation = LEGACY_VALIDATION_RE.search(line)
        if validation:
            history.append({
                "epoch": int(validation.group("epoch")),
                "micro_f1": float(validation.group("micro")),
                "macro_f1": float(validation.group("macro")),
            })
        test = LEGACY_TEST_RE.search(line)
        if test:
            metrics = {
                "accuracy": float(test.group("accuracy")),
                "micro_f1": float(test.group("micro")),
                "macro_f1": float(test.group("macro")),
            }
    if not history:
        raise ValueError(f"No validation history found in legacy log: {path}")
    if metrics is None:
        raise ValueError(f"No final test metrics found in legacy log: {path}")
    return history, metrics


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
