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


@pytest.mark.parametrize("fusion_mode", ["concat", "gate", "gated_concat"])
def test_dvcl_view_diagnostics_match_classifier_shape(fusion_mode):
    features = torch.randn(6, 4)
    graph = build_feature_knn_graph(features, 2)
    model = DualViewContrastiveDefense(
        4, 3, 2, 2, 0.0, view_mode="both", fusion_mode=fusion_mode
    )
    _, topology, feature = model(features, graph, graph)
    diagnostics = model.diagnostic_views(topology, feature)
    assert diagnostics["topology_logits"].shape == (6, 2)
    assert diagnostics["feature_logits"].shape == (6, 2)
    assert ("gate_weight" in diagnostics) == (fusion_mode != "concat")
