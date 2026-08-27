from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sp = pytest.importorskip("scipy.sparse")
pytest.importorskip("dgl")

from dvcl_bench.adapters import train_han
from dvcl_bench.artifacts import (
    CleanGraphArtifact,
    SplitArtifact,
    file_sha256,
    load_attack_artifact,
)
from dvcl_bench.attacks import build_attack_artifact
from dvcl_bench.training import HANTrainConfig


def test_han_materializes_model_specific_adaptive_attack(tmp_path: Path):
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
        hete_adjs={"pa": paper_author, "ap": paper_author.T.tocsr()},
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
    source = tmp_path / "targets.pt"
    source.write_bytes(b"targets")
    request = build_attack_artifact(
        clean=clean,
        split=split,
        attack_name="adaptive_query",
        attack_rate=5,
        seed=2,
        perturbed=clean.hete_adjs,
        target_nodes=torch.tensor([3]),
        source=str(source),
        source_sha256=file_sha256(source),
        provenance={
            "request_only": True,
            "candidate_additions": 1,
            "candidate_deletions": 1,
        },
        threat_model="evasion",
        scope="target",
        adaptive=True,
        target_changes=[{
            "target": 3,
            "relation": "pa",
            "reverse_relation": "ap",
            "target_position": 0,
            "deleted": [],
            "added": [],
        }],
    )
    checkpoint = tmp_path / "checkpoint.pt"
    result = train_han(
        clean=clean,
        split=split,
        attack=request,
        config=HANTrainConfig(hidden_dim=4, heads=2, dropout=0.0),
        train_seed=1,
        epochs=1,
        patience=1,
        device="cpu",
        checkpoint_path=checkpoint,
    )
    generated = load_attack_artifact(tmp_path / "adaptive_attack.pt")
    assert generated.provenance["victim_model"] == "han"
    assert generated.provenance["request_only"] is False
    assert len(generated.provenance["candidate_pool_sha256"]) == 64
    assert result.diagnostics["adaptive_attack"]["victim_model"] == "han"
    assert "attack_success_rate" in result.diagnostics
    assert set(result.diagnostics["budget_evaluations"]) == {"1", "3", "5"}
    assert result.diagnostics["search_budget"] == 5
    for budget in (1, 3, 5):
        evaluation = result.diagnostics["budget_evaluations"][str(budget)]
        assert evaluation["adaptive_attack"]["budget_per_target"] == budget
        assert evaluation["adaptive_attack"]["budget_utilization"] <= 1.0
        assert 0.0 <= evaluation["metrics"]["micro_f1"] <= 1.0
    assert (tmp_path / "adaptive_attacks" / "rate_1" / "attack.pt").is_file()
    assert (tmp_path / "adaptive_attacks" / "rate_3" / "attack.pt").is_file()
