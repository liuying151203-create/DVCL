import numpy as np
import scipy.sparse as sp
import torch

from dvcl_bench.models.semantic import (
    feature_similarity,
    semantic_graph,
    transition_edges,
)


def test_cached_feature_similarity_preserves_transition_edges():
    features = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    adjacency = sp.csr_matrix(
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    )
    adjs = {"pa": adjacency, "ap": adjacency.T.tocsr()}
    uncached = transition_edges(features, adjs, [["pa", "ap"]])
    cached = transition_edges(
        features, adjs, [["pa", "ap"]], feature_similarity(features)
    )
    assert torch.equal(uncached[0][0], cached[0][0])
    assert torch.equal(uncached[0][1], cached[0][1])
    assert uncached[0][2] == cached[0][2]


def test_semantic_graph_can_disable_second_stage_filter():
    transitions = [(
        torch.tensor([[0, 0], [1, 2]]),
        torch.tensor([0.001, 0.003]),
        3,
    )]
    filtered = semantic_graph(
        transitions, torch.tensor([1.0]), 0.002, apply_filter=True
    )
    unfiltered = semantic_graph(
        transitions, torch.tensor([1.0]), 0.002, apply_filter=False
    )
    assert filtered.nnz == 1
    assert unfiltered.nnz == 2
