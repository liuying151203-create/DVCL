from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sp = pytest.importorskip("scipy.sparse")
pytest.importorskip("dgl")
pytest.importorskip("openhgnn")

from dvcl_bench.artifacts import AttackArtifact, CleanGraphArtifact, SplitArtifact
from dvcl_bench.openhgnn_adapter import (
    build_openhgnn_config,
    train_openhgnn,
)


@pytest.fixture
def toy_protocol():
    paper_author = sp.csr_matrix([
        [1, 0],
        [1, 1],
        [0, 1],
        [1, 0],
    ], dtype="int8")
    clean = CleanGraphArtifact(
        dataset="acm",
        version="toy-v1",
        predict_ntype="paper",
        node_counts={"paper": 4, "author": 2},
        hete_adjs={"pa": paper_author, "ap": paper_author.T},
        features=torch.randn(4, 6),
        labels=torch.tensor([0, 1, 0, 1]),
        num_classes=2,
        meta_paths=[["pa", "ap"]],
        canonical_etypes=[
            ("paper", "pa", "author"),
            ("author", "ap", "paper"),
        ],
        stats={},
    )
    split = SplitArtifact(
        dataset="acm",
        split_name="paper_seed_1",
        seed=1,
        protocol="paper",
        train_mask=torch.tensor([1, 1, 0, 0], dtype=torch.bool),
        val_mask=torch.tensor([0, 0, 1, 0], dtype=torch.bool),
        test_mask=torch.tensor([0, 0, 0, 1], dtype=torch.bool),
        train_idx=torch.tensor([0, 1]),
        val_idx=torch.tensor([2]),
        test_idx=torch.tensor([3]),
        stats={},
    )
    return clean, split


@pytest.mark.parametrize("name,values", [
    ("hgt", {"hidden_dim": 8, "num_heads": 2, "num_layers": 2, "dropout": 0.0}),
    ("simplehgn", {
        "hidden_dim": 8, "num_heads": 2, "num_layers": 2,
        "edge_dim": 4, "dropout": 0.0,
    }),
    ("magnn", {
        "hidden_dim": 8, "num_heads": 2, "num_layers": 2,
        "inter_attention_dim": 4, "dropout": 0.0, "encoder_type": "Average",
        "instances_per_node": 2,
    }),
    ("heco", {
        "hidden_dim": 8, "feature_dropout": 0.0, "attention_dropout": 0.0,
        "schema_sample_size": 1, "positive_topk": 2,
    }),
])
def test_openhgnn_models_train_from_frozen_artifacts(
    tmp_path: Path, toy_protocol, name, values
):
    clean, split = toy_protocol
    result = train_openhgnn(
        clean=clean,
        split=split,
        attack=None,
        config=build_openhgnn_config(name, values),
        train_seed=1,
        epochs=1,
        patience=1,
        device="cpu",
        checkpoint_path=tmp_path / f"{name}.pt",
        model_name=name,
    )
    assert 0.0 <= result.metrics["micro_f1"] <= 1.0
    assert (tmp_path / f"{name}.pt").is_file()
    assert result.diagnostics["openhgnn_revision"].startswith("27a483e")


def test_openhgnn_config_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown hgt configuration"):
        build_openhgnn_config("hgt", {"unknown": 1})


def test_magnn_evaluates_target_evasion(tmp_path: Path, toy_protocol):
    clean, split = toy_protocol
    empty = {
        name: sp.csr_matrix(value.shape, dtype="int8")
        for name, value in clean.hete_adjs.items()
    }
    attack = AttackArtifact(
        dataset="acm",
        attack_name="hg_baseline",
        attack_rate=1,
        seed=1,
        clean_version="toy-v1",
        split_name="paper_seed_1",
        split_seed=1,
        perturbed_hete_adjs=clean.hete_adjs,
        added_edges=empty,
        deleted_edges=empty,
        target_nodes=torch.tensor([3]),
        stats={},
        source="toy",
        threat_model="evasion",
        scope="target",
        target_changes=[{
            "target": 3,
            "relation": "pa",
            "reverse_relation": "ap",
            "target_position": 0,
            "deleted": [[3, 0]],
            "added": [[3, 1]],
        }],
    )
    result = train_openhgnn(
        clean=clean,
        split=split,
        attack=attack,
        config=build_openhgnn_config("magnn", {
            "hidden_dim": 8,
            "num_heads": 2,
            "num_layers": 2,
            "inter_attention_dim": 4,
            "dropout": 0.0,
            "encoder_type": "Average",
            "instances_per_node": 2,
        }),
        train_seed=1,
        epochs=1,
        patience=1,
        device="cpu",
        checkpoint_path=tmp_path / "magnn-target.pt",
        model_name="magnn",
    )
    assert result.diagnostics["evaluation_scope"] == "target"
    assert result.diagnostics["target_count"] == 1
