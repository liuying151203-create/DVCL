from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sp = pytest.importorskip("scipy.sparse")

from dvcl_bench.artifacts import (
    AttackArtifact,
    CleanGraphArtifact,
    load_attack_artifact,
    load_clean_artifact,
    save_attack_artifact,
    save_clean_artifact,
)


def test_clean_artifact_round_trip_is_legacy_independent(tmp_path: Path):
    adjacency = sp.csr_matrix([[0, 1], [1, 0]])
    artifact = CleanGraphArtifact(
        dataset="toy",
        version="toy-v1",
        predict_ntype="node",
        node_counts={"node": 2},
        hete_adjs={"nn": adjacency},
        features=torch.eye(2),
        labels=torch.tensor([0, 1]),
        num_classes=2,
        meta_paths=[["nn"]],
        canonical_etypes=[("node", "nn", "node")],
        stats={"edge_counts": {"nn": 2}},
    )
    path = tmp_path / "clean.pt"
    save_clean_artifact(artifact, path)
    loaded = load_clean_artifact(path)
    assert loaded.version == "toy-v1"
    assert loaded.canonical_etypes == [("node", "nn", "node")]
    assert (loaded.hete_adjs["nn"] != adjacency).nnz == 0
    assert (tmp_path / "meta.json").is_file()


def test_attack_artifact_preserves_generation_provenance(tmp_path: Path):
    adjacency = sp.csr_matrix([[0, 1], [1, 0]])
    artifact = AttackArtifact(
        dataset="toy",
        attack_name="heteprbcd",
        attack_rate=5,
        seed=1,
        clean_version="toy-v1",
        split_name="paper_seed_1",
        split_seed=1,
        perturbed_hete_adjs={"nn": adjacency},
        added_edges={"nn": sp.csr_matrix(adjacency.shape)},
        deleted_edges={"nn": sp.csr_matrix(adjacency.shape)},
        target_nodes=None,
        stats={},
        source="source.pt",
        provenance={"constrained": True, "biased": True, "lambda": 0.4},
    )
    path = tmp_path / "attack.pt"
    save_attack_artifact(artifact, path)
    loaded = load_attack_artifact(path)
    assert loaded.provenance == artifact.provenance
