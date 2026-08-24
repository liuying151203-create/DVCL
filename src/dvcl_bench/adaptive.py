"""Model-aware target evasion utilities."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from .artifacts import CleanGraphArtifact
from .attacks import TARGET_ATTACK_SPECS, apply_target_change


def adversarial_margin(logits: torch.Tensor, true_label: int) -> float:
    values = logits.detach().flatten()
    if values.numel() <= 1:
        return float("-inf")
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[int(true_label)] = False
    return float(values[mask].max() - values[int(true_label)])


def target_candidates(
    clean: CleanGraphArtifact,
    target: int,
    seed: int,
    max_additions: int,
    max_deletions: int,
) -> List[Tuple[str, List[int]]]:
    spec = TARGET_ATTACK_SPECS[clean.dataset]
    relation = spec["relation"]
    position = int(spec["target_position"])
    adjacency = clean.hete_adjs[relation].tocsr()
    rng = np.random.RandomState(_target_seed(seed, target))
    if position == 0:
        existing = adjacency.getrow(target).indices.astype(np.int64)
        endpoint_count = adjacency.shape[1]
        edge = lambda endpoint: [int(target), int(endpoint)]
    else:
        existing = adjacency.getcol(target).indices.astype(np.int64)
        endpoint_count = adjacency.shape[0]
        edge = lambda endpoint: [int(endpoint), int(target)]
    existing_set = set(existing.tolist())
    deleted = _sample(existing, max_deletions, rng)
    additions = []
    attempts = 0
    maximum_attempts = max(100, max_additions * 50)
    while len(additions) < min(max_additions, endpoint_count - len(existing_set)):
        endpoint = int(rng.randint(endpoint_count))
        attempts += 1
        if endpoint not in existing_set and endpoint not in additions:
            additions.append(endpoint)
        if attempts >= maximum_attempts:
            break
    return [
        *(("deleted", edge(endpoint)) for endpoint in deleted),
        *(("added", edge(endpoint)) for endpoint in additions),
    ]


def greedy_query_target_changes(
    clean: CleanGraphArtifact,
    targets: Iterable[int],
    labels: torch.Tensor,
    clean_logits: torch.Tensor,
    forward_for_adjs: Callable[[Mapping[str, object], Mapping[str, object]], torch.Tensor],
    budget: int,
    seed: int,
    max_additions: int = 16,
    max_deletions: int = 16,
):
    if budget not in {1, 3, 5}:
        raise ValueError("Adaptive target budget must be one of 1, 3 or 5")
    spec = TARGET_ATTACK_SPECS[clean.dataset]
    labels = labels.detach()
    clean_logits = clean_logits.detach()
    records = []
    query_count = 0
    changed_targets = 0
    for raw_target in targets:
        target = int(raw_target)
        record = {
            "target": target,
            "relation": spec["relation"],
            "reverse_relation": spec["reverse"],
            "target_position": int(spec["target_position"]),
            "deleted": [],
            "added": [],
        }
        true_label = int(labels[target])
        current_margin = adversarial_margin(clean_logits[target], true_label)
        candidates = target_candidates(
            clean, target, seed, max_additions, max_deletions
        )
        for _ in range(budget):
            best = None
            best_margin = current_margin
            for kind, edge in candidates:
                candidate_record = _with_change(record, kind, edge)
                with torch.no_grad():
                    logits = forward_for_adjs(
                        apply_target_change(clean, candidate_record), candidate_record
                    )
                query_count += 1
                margin = adversarial_margin(logits[target], true_label)
                if margin > best_margin:
                    best = (kind, edge)
                    best_margin = margin
            if best is None:
                break
            kind, edge = best
            record[kind].append(edge)
            candidates.remove(best)
            current_margin = best_margin
        if record["deleted"] or record["added"]:
            changed_targets += 1
        records.append(record)
    diagnostics = {
        "targets": len(records),
        "changed_targets": changed_targets,
        "queries": query_count,
        "budget_per_target": int(budget),
        "candidate_additions": int(max_additions),
        "candidate_deletions": int(max_deletions),
        "objective": "maximize max_other_logit_minus_true_logit",
        "algorithm": "greedy_score_based_query",
    }
    return records, diagnostics


def _with_change(record: Mapping[str, object], kind: str, edge: Sequence[int]):
    value = {
        "target": int(record["target"]),
        "relation": str(record["relation"]),
        "reverse_relation": str(record["reverse_relation"]),
        "target_position": int(record["target_position"]),
        "deleted": [list(item) for item in record.get("deleted", [])],
        "added": [list(item) for item in record.get("added", [])],
    }
    value[kind].append(list(edge))
    return value


def _sample(values: np.ndarray, count: int, rng) -> List[int]:
    if count <= 0 or not len(values):
        return []
    if len(values) <= count:
        return values.tolist()
    return values[rng.choice(len(values), size=count, replace=False)].tolist()


def _target_seed(seed: int, target: int) -> int:
    return int((int(seed) * 1_000_003 + int(target) * 97) % (2**32 - 1))
