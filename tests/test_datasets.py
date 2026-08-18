from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sp = pytest.importorskip("scipy.sparse")

from dvcl_bench.datasets import build_clean_artifact


def test_aminer_loader_requires_the_frozen_paper_protocol_shape(tmp_path: Path):
    directory = tmp_path / "aminer"
    directory.mkdir()
    papers = 6564
    sp.save_npz(directory / "pa.npz", sp.csr_matrix((papers, 3)))
    sp.save_npz(directory / "pr.npz", sp.csr_matrix((papers, 4)))
    sp.save_npz(directory / "pos.npz", sp.eye(papers, 2, format="csr"))
    np.save(directory / "labels.npy", np.arange(papers) % 4)
    artifact = build_clean_artifact("aminer", tmp_path)
    assert artifact.predict_ntype == "paper"
    assert artifact.node_counts == {"paper": papers, "author": 3, "research": 4}
    assert artifact.features.shape == (papers, 2)
    assert artifact.num_classes == 4


def test_aminer_loader_rejects_a_different_preprocessing(tmp_path: Path):
    directory = tmp_path / "aminer"
    directory.mkdir()
    sp.save_npz(directory / "pa.npz", sp.csr_matrix((10, 3)))
    sp.save_npz(directory / "pr.npz", sp.csr_matrix((10, 4)))
    sp.save_npz(directory / "pos.npz", sp.eye(10, format="csr"))
    np.save(directory / "labels.npy", np.arange(10) % 4)
    with pytest.raises(ValueError, match="6564"):
        build_clean_artifact("aminer", tmp_path)
