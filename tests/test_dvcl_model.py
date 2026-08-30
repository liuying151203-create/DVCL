import pytest

torch = pytest.importorskip("torch")
dgl = pytest.importorskip("dgl")

from dvcl_bench.models.dvcl import (
    DualViewContrastiveDefense,
    build_feature_knn_graph,
    cross_view_contrastive_loss,
    perturb_topology_graph,
    reliability_signals,
)
from dvcl_bench.models.semantic import SemanticHAN


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


def test_han_semantic_topology_receives_fused_classification_gradients():
    features = torch.randn(6, 4)
    graph = build_feature_knn_graph(features, 2)
    semantic = SemanticHAN(1, 4, 3, 2, 2, 0.0)
    model = DualViewContrastiveDefense(
        4, 3, 2, 2, 0.0, view_mode="both", fusion_mode="concat"
    )
    topology = semantic.encode(features, [graph])
    logits, _, _ = model.forward_with_topology_embedding(
        features, topology, graph
    )
    torch.nn.functional.cross_entropy(
        logits, torch.tensor([0, 1, 0, 1, 0, 1])
    ).backward()
    assert semantic.layer.gat_layers[0].fc.weight.grad is not None


@pytest.mark.parametrize(
    "fusion_mode", ["concat", "gate", "gated_concat", "reliability_gate"]
)
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


def test_reliability_gate_uses_reference_free_signals_and_losses():
    features = torch.randn(6, 4)
    graph = build_feature_knn_graph(features, 2)
    model = DualViewContrastiveDefense(
        4, 3, 2, 2, 0.0, view_mode="both",
        fusion_mode="reliability_gate",
    )
    logits, topology, feature = model(features, graph, graph)
    signals = reliability_signals(
        model.topology_classifier(topology),
        model.feature_classifier(feature),
        topology,
        feature,
    )
    losses = model.reliability_losses(
        topology, feature, torch.tensor([0, 1, 0, 1, 0, 1]),
        torch.ones(6, dtype=torch.bool),
    )
    assert logits.shape == (6, 2)
    assert signals.shape == (6, 7)
    assert torch.isfinite(signals).all()
    assert model.last_gate_weight.shape == (6, 1)
    assert losses["auxiliary_loss"].ndim == 0
    assert losses["route_loss"].ndim == 0
    (losses["auxiliary_loss"] + losses["route_loss"]).backward()
    assert model.gate[0].weight.grad is not None


def test_non_reliability_feature_only_mode_has_zero_route_losses():
    features = torch.randn(6, 4)
    graph = build_feature_knn_graph(features, 2)
    model = DualViewContrastiveDefense(
        4, 3, 2, 2, 0.0, view_mode="feat", fusion_mode="concat"
    )
    _, topology, feature = model(features, graph, graph)
    losses = model.reliability_losses(
        topology, feature, torch.zeros(6, dtype=torch.long),
        torch.ones(6, dtype=torch.bool),
    )
    assert losses["auxiliary_loss"].item() == 0.0
    assert losses["route_loss"].item() == 0.0


def test_topology_augmentation_preserves_node_and_edge_counts():
    features = torch.randn(8, 4)
    graph = build_feature_knn_graph(features, 2)
    torch.manual_seed(3)
    augmented = perturb_topology_graph(graph, 0.25)
    assert augmented.num_nodes() == graph.num_nodes()
    assert augmented.num_edges() == graph.num_edges()
    assert not torch.equal(
        torch.stack(graph.edges()), torch.stack(augmented.edges())
    )
