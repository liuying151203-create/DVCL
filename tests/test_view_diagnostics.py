import pytest

torch = pytest.importorskip("torch")

from dvcl_bench.view_diagnostics import target_view_diagnostics


def test_target_view_diagnostics_record_success_drift_and_disagreement():
    labels = torch.tensor([0, 1])
    clean = {
        "fused_logits": torch.tensor([[3.0, 1.0], [1.0, 3.0]]),
        "topology_logits": torch.tensor([[3.0, 1.0], [1.0, 3.0]]),
        "feature_logits": torch.tensor([[2.0, 1.0], [1.0, 2.0]]),
        "topology_embedding": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "feature_embedding": torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        "gate_weight": torch.tensor([[0.7], [0.6]]),
    }
    attacked = [
        {
            "target": 0,
            "fused_logits": torch.tensor([0.0, 2.0]),
            "topology_logits": torch.tensor([0.0, 2.0]),
            "feature_logits": torch.tensor([2.0, 1.0]),
            "topology_embedding": torch.tensor([0.0, 1.0]),
            "feature_embedding": torch.tensor([1.0, 1.0]),
            "gate_weight": torch.tensor([0.4]),
        },
        {
            "target": 1,
            "fused_logits": torch.tensor([1.0, 3.0]),
            "topology_logits": torch.tensor([3.0, 1.0]),
            "feature_logits": torch.tensor([1.0, 2.0]),
            "topology_embedding": torch.tensor([1.0, 0.0]),
            "feature_embedding": torch.tensor([1.0, 1.0]),
            "gate_weight": torch.tensor([0.5]),
        },
    ]
    diagnostics = target_view_diagnostics(clean, attacked, labels)
    assert diagnostics["clean"]["fused_target_micro_f1"] == 1.0
    assert diagnostics["attacked"]["fused_target_micro_f1"] == 0.5
    assert diagnostics["drift"]["topology_l2_mean"] > 0
    assert diagnostics["drift"]["feature_l2_mean"] == 0
    assert diagnostics["attacked"]["view_disagreement_rate"] == 1.0
    assert diagnostics["per_target"][0]["attack_success"] is True
