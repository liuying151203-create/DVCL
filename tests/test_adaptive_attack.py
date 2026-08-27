from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch

from dvcl_bench.adaptive import (
    adversarial_margin,
    greedy_query_target_changes,
    record_query_count,
    records_at_budget,
    target_candidates,
)

from scripts.prepare_dvcl_adaptive_requests import stratified_targets


def clean_graph():
    pa = sp.csr_matrix(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int8))
    return SimpleNamespace(
        dataset="acm",
        hete_adjs={"pa": pa, "ap": pa.T.tocsr()},
    )


def dblp_clean_graph():
    paper_author = sp.csr_matrix(np.array([
        [0, 1],
        [1, 0],
        [0, 1],
    ], dtype=np.int8))
    return SimpleNamespace(
        dataset="dblp",
        hete_adjs={"pa": paper_author, "ap": paper_author.T.tocsr()},
    )


def test_adversarial_margin_uses_strongest_wrong_class():
    assert adversarial_margin(torch.tensor([2.0, 1.5, 3.0]), 0) == 1.0


def test_target_candidates_touch_target_and_respect_edge_state():
    clean = clean_graph()
    values = target_candidates(clean, target=0, seed=1, max_additions=2, max_deletions=2)
    assert ("deleted", [0, 0]) in values
    assert all(edge[0] == 0 for _, edge in values)
    assert all(bool(clean.hete_adjs["pa"][tuple(edge)]) == (kind == "deleted") for kind, edge in values)


def test_column_target_candidates_use_original_row_indices():
    clean = dblp_clean_graph()
    values = target_candidates(
        clean, target=1, seed=1, max_additions=2, max_deletions=2
    )
    deleted = [edge for kind, edge in values if kind == "deleted"]
    assert sorted(deleted) == [[0, 1], [2, 1]]
    assert all(edge[1] == 1 for _, edge in values)
    assert all(
        bool(clean.hete_adjs["pa"][tuple(edge)]) == (kind == "deleted")
        for kind, edge in values
    )


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
    assert diagnostics["total_changes"] == 1
    assert diagnostics["mean_changes_per_target"] == 1.0
    assert diagnostics["budget_utilization"] == 1.0
    assert len(diagnostics["candidate_pool_sha256"]) == 64

    _, repeated = greedy_query_target_changes(
        clean, [0], labels, clean_logits, forward, budget=1, seed=2,
        max_additions=2, max_deletions=1,
    )
    assert repeated["candidate_pool_sha256"] == diagnostics["candidate_pool_sha256"]


def test_candidate_pool_hash_is_independent_of_selected_change():
    clean = clean_graph()
    labels = torch.tensor([0, 1])
    clean_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])

    def prefer_deletion(adjs, record):
        logits = clean_logits.clone()
        if record["deleted"]:
            logits[0] = torch.tensor([0.0, 3.0])
        return logits

    def prefer_addition(adjs, record):
        logits = clean_logits.clone()
        if record["added"]:
            logits[0] = torch.tensor([0.0, 3.0])
        return logits

    deletion_records, deletion_diagnostics = greedy_query_target_changes(
        clean, [0], labels, clean_logits, prefer_deletion, budget=1, seed=2,
        max_additions=1, max_deletions=1,
    )
    addition_records, addition_diagnostics = greedy_query_target_changes(
        clean, [0], labels, clean_logits, prefer_addition, budget=1, seed=2,
        max_additions=1, max_deletions=1,
    )

    assert deletion_records != addition_records
    assert deletion_diagnostics["candidate_pool_sha256"] == addition_diagnostics[
        "candidate_pool_sha256"
    ]


def test_search_skips_clean_incorrect_targets():
    clean = clean_graph()
    labels = torch.tensor([0, 1])
    clean_logits = torch.tensor([[0.0, 2.0], [0.0, 2.0]])
    calls = 0

    def forward(adjs, record):
        nonlocal calls
        calls += 1
        return clean_logits

    records, diagnostics = greedy_query_target_changes(
        clean, [0], labels, clean_logits, forward, budget=5, seed=2,
        max_additions=2, max_deletions=1,
    )
    assert calls == 0
    assert not records[0]["added"] and not records[0]["deleted"]
    assert diagnostics["skipped_clean_incorrect"] == 1


def test_search_stops_after_first_successful_change():
    clean = clean_graph()
    labels = torch.tensor([0, 1])
    clean_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])

    def forward(adjs, record):
        logits = clean_logits.clone()
        if record["added"]:
            logits[0] = torch.tensor([0.0, 3.0])
        return logits

    records, diagnostics = greedy_query_target_changes(
        clean, [0], labels, clean_logits, forward, budget=5, seed=2,
        max_additions=2, max_deletions=1,
    )
    assert len(records[0]["added"]) == 1
    assert diagnostics["queries"] == 3
    assert diagnostics["early_stopped_successes"] == 1


def test_search_trajectory_can_be_reused_for_smaller_budgets():
    clean = clean_graph()
    labels = torch.tensor([0, 1])
    clean_logits = torch.tensor([[4.0, 0.0], [0.0, 2.0]])

    def forward(adjs, record):
        logits = clean_logits.clone()
        changes = len(record["added"]) + len(record["deleted"])
        logits[0] = torch.tensor([4.0 - changes, float(changes)])
        return logits

    records, diagnostics = greedy_query_target_changes(
        clean, [0], labels, clean_logits, forward, budget=5, seed=2,
        max_additions=2, max_deletions=1,
    )
    one = records_at_budget(records, 1)
    three = records_at_budget(records, 3)
    assert len(one[0]["sequence"]) == 1
    assert len(three[0]["sequence"]) == len(records[0]["sequence"])
    assert record_query_count(one) <= record_query_count(three)
    assert record_query_count(records) == diagnostics["queries"]


def test_query_count_includes_terminal_no_improvement_round():
    clean = clean_graph()
    labels = torch.tensor([0, 1])
    clean_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])

    def forward(adjs, record):
        return clean_logits

    records, diagnostics = greedy_query_target_changes(
        clean, [0], labels, clean_logits, forward, budget=5, seed=2,
        max_additions=2, max_deletions=1,
    )
    assert records[0]["sequence"] == []
    assert records[0]["terminal_queries"] == 3
    assert record_query_count(records) == diagnostics["queries"] == 3
    assert record_query_count(records_at_budget(records, 1)) == 3


def test_stratified_targets_is_deterministic_and_class_balanced():
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    first = stratified_targets(range(9), labels, count=6, seed=7)
    second = stratified_targets(range(9), labels, count=6, seed=7)
    assert torch.equal(first, second)
    selected_labels = labels[first.numpy()]
    assert {label: int((selected_labels == label).sum()) for label in range(3)} == {
        0: 2, 1: 2, 2: 2,
    }
