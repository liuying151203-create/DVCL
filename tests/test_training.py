import pytest

torch = pytest.importorskip("torch")
sp = pytest.importorskip("scipy.sparse")

from dvcl_bench.adapters import (
    _train_supervised,
    _validate_dvcl_strategy,
    train_dvcl,
)
from dvcl_bench.artifacts import AttackArtifact, CleanGraphArtifact, SplitArtifact
from dvcl_bench.registry import build_model_config
from dvcl_bench.training import (
    DVCLTrainConfig,
    HANTrainConfig,
    LegacyEarlyStopping,
    _checkpoint_config_matches,
)


def test_legacy_early_stopping_restores_joint_best_state():
    model = torch.nn.Linear(2, 1, bias=False)
    stopper = LegacyEarlyStopping(patience=2)
    with torch.no_grad():
        model.weight.fill_(1)
    assert not stopper.step(1.0, 0.5, model, 0)
    with torch.no_grad():
        model.weight.fill_(2)
    assert not stopper.step(1.2, 0.4, model, 1)
    assert stopper.step(1.3, 0.3, model, 2)
    stopper.restore(model)
    assert torch.all(model.weight == 1)
    assert stopper.best_epoch == 0


def test_supervised_training_can_reuse_exact_checkpoint(tmp_path):
    labels = torch.tensor([0, 1, 0])
    masks = {
        "train": torch.tensor([1, 1, 0], dtype=torch.bool),
        "val": torch.tensor([0, 1, 0], dtype=torch.bool),
        "test": torch.tensor([0, 0, 1], dtype=torch.bool),
    }
    features = torch.eye(3)
    config = HANTrainConfig(learning_rate=0.01)
    source = tmp_path / "source.pt"
    first_model = torch.nn.Linear(3, 2)
    _train_supervised(
        first_model, lambda: first_model(features), labels, masks,
        config, epochs=1, patience=1, checkpoint_path=source,
    )
    destination = tmp_path / "reused.pt"
    reused_model = torch.nn.Linear(3, 2)
    result = _train_supervised(
        reused_model, lambda: reused_model(features), labels, masks,
        config, epochs=1, patience=1, checkpoint_path=destination,
        checkpoint_source=source,
    )
    assert result.diagnostics["checkpoint_reused"] is True
    assert result.diagnostics["optimizer_steps"] == 0
    assert source.read_bytes() == destination.read_bytes()


def test_dvcl_reliability_gate_config_is_registered():
    config = build_model_config("dvcl", {
        "fusion_mode": "reliability_gate",
        "gate_hidden_dim": 8,
        "route_temperature": 0.75,
        "beta_aux": 0.25,
        "lambda_route": 0.5,
    })
    assert config.fusion_mode == "reliability_gate"
    assert config.gate_hidden_dim == 8
    assert config.route_temperature == 0.75
    assert config.beta_aux == 0.25
    assert config.lambda_route == 0.5


def test_old_dvcl_checkpoint_config_accepts_new_default_fields_only():
    config = DVCLTrainConfig()
    old = {
        name: value
        for name, value in config.__dict__.items()
        if name not in {
            "gate_hidden_dim", "route_temperature", "beta_aux", "lambda_route",
            "structure_augment_rate", "lambda_aug", "topology_source",
            "semantic_topology_filter",
        }
    }
    assert _checkpoint_config_matches(old, config)
    assert not _checkpoint_config_matches(old, DVCLTrainConfig(lambda_route=0.5))


def test_han_semantic_strategy_requires_compatible_full_checkpoint():
    _validate_dvcl_strategy(DVCLTrainConfig(topology_source="han_semantic"))
    with pytest.raises(ValueError, match="dimension mismatch"):
        _validate_dvcl_strategy(DVCLTrainConfig(
            topology_source="han_semantic", semantic_hidden_dim=32
        ))
    with pytest.raises(ValueError, match="full semantic and DVCL checkpoint"):
        _validate_dvcl_strategy(DVCLTrainConfig(
            topology_source="han_semantic", legacy_checkpoint_semantics=True
        ))


def test_dvcl_strategy_rejects_unknown_topology_switches():
    with pytest.raises(ValueError, match="Unsupported DVCL topology source"):
        _validate_dvcl_strategy(DVCLTrainConfig(topology_source="unknown"))
    with pytest.raises(ValueError, match="Unsupported semantic topology filter"):
        _validate_dvcl_strategy(DVCLTrainConfig(
            semantic_topology_filter="soft"
        ))


def test_feature_only_training_is_invariant_to_structural_attack(tmp_path):
    forward = sp.csr_matrix([
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
    ])
    attacked = sp.csr_matrix([
        [1, 1, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 0, 1, 1, 0, 1],
        [1, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 1],
    ])
    clean = CleanGraphArtifact(
        dataset="acm",
        version="toy-v1",
        predict_ntype="paper",
        node_counts={"paper": 6, "author": 6},
        hete_adjs={"pa": forward, "ap": forward.T.tocsr()},
        features=torch.eye(6),
        labels=torch.tensor([0, 1, 0, 1, 0, 1]),
        num_classes=2,
        meta_paths=[["pa", "ap"]],
        canonical_etypes=[
            ("paper", "pa", "author"),
            ("author", "ap", "paper"),
        ],
        stats={},
    )
    masks = {
        "train": torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool),
        "val": torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.bool),
        "test": torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool),
    }
    split = SplitArtifact(
        dataset="acm",
        split_name="toy",
        seed=1,
        protocol="toy",
        train_mask=masks["train"],
        val_mask=masks["val"],
        test_mask=masks["test"],
        train_idx=torch.tensor([0, 1, 2]),
        val_idx=torch.tensor([3]),
        test_idx=torch.tensor([4, 5]),
        stats={},
    )
    attack = AttackArtifact(
        dataset="acm",
        attack_name="prbcd",
        attack_rate=25,
        seed=1,
        clean_version="toy-v1",
        split_name="toy",
        split_seed=1,
        perturbed_hete_adjs={"pa": attacked, "ap": attacked.T.tocsr()},
        added_edges={},
        deleted_edges={},
        target_nodes=None,
        stats={},
        source="test",
    )
    config = DVCLTrainConfig(
        hidden_dim=4,
        heads=2,
        semantic_hidden_dim=4,
        semantic_heads=2,
        knn_k=2,
        view_mode="feat",
        lambda_dvcl=0.0,
    )
    clean_checkpoint = tmp_path / "clean.pt"
    attacked_checkpoint = tmp_path / "attacked.pt"
    clean_result = train_dvcl(
        clean, split, None, config, 7, 3, 3, "cpu", clean_checkpoint
    )
    attacked_result = train_dvcl(
        clean, split, attack, config, 7, 3, 3, "cpu", attacked_checkpoint
    )
    assert clean_result.metrics == attacked_result.metrics
    assert clean_result.history == attacked_result.history
    assert clean_result.diagnostics["topology_branch_active"] is False
    assert clean_result.diagnostics["semantic_attention"] == []
    clean_state = torch.load(clean_checkpoint, map_location="cpu")["state_dict"]
    attacked_state = torch.load(attacked_checkpoint, map_location="cpu")["state_dict"]
    assert clean_state.keys() == attacked_state.keys()
    assert all(
        torch.equal(clean_state[name], attacked_state[name]) for name in clean_state
    )
