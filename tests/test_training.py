import pytest

torch = pytest.importorskip("torch")

from dvcl_bench.adapters import _train_supervised
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
            "structure_augment_rate", "lambda_aug",
        }
    }
    assert _checkpoint_config_matches(old, config)
    assert not _checkpoint_config_matches(old, DVCLTrainConfig(lambda_route=0.5))
