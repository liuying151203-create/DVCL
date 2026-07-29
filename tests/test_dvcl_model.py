import pytest

torch = pytest.importorskip("torch")
dgl = pytest.importorskip("dgl")

from dvcl_bench.models.dvcl import (
    DualViewContrastiveDefense,
    build_feature_knn_graph,
    cross_view_contrastive_loss,
)


def test_dvcl_forward_and_ablation_modes():
    features = torch.randn(6, 4)
    graph = build_feature_knn_graph(features, 2)
    model = DualViewContrastiveDefense(4, 3, 2, 2, 0.0, view_mode="both")
    logits, topology, feature = model(features, graph, graph)
    assert logits.shape == (6, 2)
    assert topology.shape == feature.shape == (6, 6)
    assert cross_view_contrastive_loss(
        topology, feature, torch.ones(6, dtype=torch.bool), 0.5
    ).ndim == 0


def test_directed_knn_has_exact_out_degree():
    features = torch.eye(5)
    graph = build_feature_knn_graph(features, 2, "directed")
    assert graph.num_edges() == 10
