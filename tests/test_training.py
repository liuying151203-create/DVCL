import pytest

torch = pytest.importorskip("torch")

from dvcl_bench.adapters import _train_supervised
from dvcl_bench.training import HANTrainConfig, LegacyEarlyStopping


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
