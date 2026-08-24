from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch

from dvcl_bench.adaptive import (
    adversarial_margin,
    greedy_query_target_changes,
    target_candidates,
)

from scripts.prepare_dvcl_adaptive_requests import stratified_targets


def clean_graph():
    pa = sp.csr_matrix(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int8))
    return SimpleNamespace(
        dataset="acm",
        hete_adjs={"pa": pa, "ap": pa.T.tocsr()},
    )


def test_adversarial_margin_uses_strongest_wrong_class():
    assert adversarial_margin(torch.tensor([2.0, 1.5, 3.0]), 0) == 1.0


def test_target_candidates_touch_target_and_respect_edge_state():
    clean = clean_graph()
    values = target_candidates(clean, target=0, seed=1, max_additions=2, max_deletions=2)
    assert ("deleted", [0, 0]) in values
    assert all(edge[0] == 0 for _, edge in values)
    assert all(bool(clean.hete_adjs["pa"][tuple(edge)]) == (kind == "deleted") for kind, edge in values)


def test_greedy_query_selects_margin_increasing_change():
    clean = clean_graph()
    labels = torch.tensor([0, 1])
    clean_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])

    def forward(adjs, record):
        logits = clean_logits.clone()
        if record["added"]:
            logits[int(record["target"])] = torch.tensor([0.0, 3.0])
        return logits

    records, diagnostics = greedy_query_target_changes(
        clean, [0], labels, clean_logits, forward, budget=1, seed=2,
        max_additions=2, max_deletions=1,
    )
    assert records[0]["added"]
    assert diagnostics["queries"] == 3
    assert diagnostics["changed_targets"] == 1


def test_stratified_targets_is_deterministic_and_class_balanced():
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    first = stratified_targets(range(9), labels, count=6, seed=7)
    second = stratified_targets(range(9), labels, count=6, seed=7)
    assert torch.equal(first, second)
    selected_labels = labels[first.numpy()]
    assert {label: int((selected_labels == label).sum()) for label in range(3)} == {
        0: 2, 1: 2, 2: 2,
    }
