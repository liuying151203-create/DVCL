import pytest

torch = pytest.importorskip("torch")

from dvcl_bench.training import LegacyEarlyStopping


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
