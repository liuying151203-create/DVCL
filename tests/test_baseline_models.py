import pytest

torch = pytest.importorskip("torch")
dgl = pytest.importorskip("dgl")

from dvcl_bench.models.baselines import HeteroGuard, HeteroSAGE
from dvcl_bench.models.semantic import SemanticHAN
from dvcl_bench.models.rohe import RoHe
from dvcl_bench.models.fastrohgcn import FastRoHGCN
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
    assert build_model_config("heteroguard", {"attention_threshold": 0.2}).attention_threshold == 0.2
    assert callable(get_native_trainer("heteroguard"))
    assert build_model_config("rohe", {"top_t": [2, 5]}).top_t == [2, 5]
    assert callable(get_native_trainer("rohe"))
    assert build_model_config("fastrohgcn", {"topk_similarity": 3}).topk_similarity == 3
    assert callable(get_native_trainer("fastrohgcn"))


def test_heteroguard_forward_shape_and_attention_filter():
    canonical = [("author", "ap", "paper"), ("paper", "pa", "author")]
    edges = {
        canonical[0]: torch.tensor([[0, 1, 2], [0, 1, 1]]),
        canonical[1]: torch.tensor([[0, 1, 1], [0, 1, 2]]),
    }
    features = {"author": torch.randn(3, 4), "paper": torch.randn(2, 4)}
    model = HeteroGuard(canonical, 4, 5, 3, 2, 0.0, 0.1, True)
    logits = model(features, edges)
    assert logits["author"].shape == (3, 3)
    assert set(model.last_attention_density) == {"author__ap__paper", "paper__pa__author"}


def test_rohe_forward_shape():
    import scipy.sparse as sp

    features = torch.randn(5, 4)
    transition = sp.csr_matrix(torch.eye(5).numpy())
    model = RoHe(2, 4, 3, 2, 2, 0.0, [2, 2])
    logits = model(features, [transition, transition])
    assert logits.shape == (5, 2)


def test_fastrohgcn_forward_shape():
    graph = dgl.graph(([0, 1, 2, 3], [1, 0, 3, 2]), num_nodes=4)
    model = FastRoHGCN(
        {"paper": 2, "author": 2}, "paper", 3, 4, 5, 1, 2, 0.0
    )
    logits = model(torch.randn(2, 3), graph, torch.ones(4))
    assert logits.shape == (4, 2)
