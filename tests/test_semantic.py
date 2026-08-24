import numpy as np
import scipy.sparse as sp
import torch

from dvcl_bench.models.semantic import feature_similarity, transition_edges


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
