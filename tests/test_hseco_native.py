import pytest

torch = pytest.importorskip("torch")

from dvcl_bench.models.hseco import (
    HSeCo,
    HSeCoConfig,
    cosine_edge_weights,
    paper_contrastive_loss,
    prune_edges,
)


def test_edge_softmax_is_normalized_per_source():
    x = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    edges = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
    weights = cosine_edge_weights(x, edges)
    assert torch.allclose(weights[:2].sum(), torch.tensor(1.0))
    assert torch.allclose(weights[2:], torch.tensor([1.0]))


def test_threshold_prunes_low_confidence_edges():
    edges = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    weights = torch.tensor([0.8, 0.2])
    kept_edges, kept_weights = prune_edges(edges, weights, 0.5)
    assert kept_edges.tolist() == [[0], [1]]
    assert kept_weights.tolist() == pytest.approx([0.8])


def test_contrastive_loss_prefers_aligned_positive_view():
    anchor = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    negative = -anchor
    aligned = paper_contrastive_loss(anchor, [anchor], negative)
    misaligned = paper_contrastive_loss(anchor, [negative], anchor)
    assert aligned < misaligned


def test_model_forward_exposes_paper_outputs():
    config = HSeCoConfig(input_dim=3, hidden_dim=8, heads=2, view_thresholds=(0.0, 0.0))
    model = HSeCo(config, num_classes=2)
    features = torch.randn(4, 3)
    view_a = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    view_b = torch.tensor([[0, 2, 3], [2, 1, 0]], dtype=torch.long)
    outputs = model(features, [view_a, view_b])
    assert outputs["logits"].shape == (4, 2)
    assert outputs["embedding"].shape == (4, 8)
    assert outputs["semantic_weights"].shape == (2,)
    assert torch.allclose(outputs["semantic_weights"].sum(), torch.tensor(1.0))
