import pytest

torch = pytest.importorskip("torch")
dgl = pytest.importorskip("dgl")

from dvcl_bench.models.baselines import HeteroSAGE
from dvcl_bench.models.semantic import SemanticHAN
from dvcl_bench.registry import build_model_config, get_native_trainer


def test_han_baseline_forward_shape():
    features = torch.randn(5, 3)
    graph = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]), num_nodes=5)
    model = SemanticHAN(2, 3, 4, 3, 2, 0.0)
    logits = model(features, [graph, graph])
    assert logits.shape == (5, 3)


def test_heterosage_baseline_forward_shape():
    canonical = [("author", "ap", "paper"), ("paper", "pa", "author")]
    graph = dgl.heterograph({
        canonical[0]: ([0, 1, 2], [0, 1, 1]),
        canonical[1]: ([0, 1, 1], [0, 1, 2]),
    }, num_nodes_dict={"author": 3, "paper": 2})
    features = {"author": torch.randn(3, 4), "paper": torch.zeros(2, 4)}
    model = HeteroSAGE(canonical, 4, 5, 3, 2, 0.0)
    logits = model(graph, features)
    assert logits["author"].shape == (3, 3)


def test_baselines_are_registered_with_strict_configs():
    assert build_model_config("han", {"hidden_dim": 32}).hidden_dim == 32
    assert build_model_config("heterosage", {"num_layers": 3}).num_layers == 3
    assert callable(get_native_trainer("han"))
    assert callable(get_native_trainer("heterosage"))
