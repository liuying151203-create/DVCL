import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
sp = pytest.importorskip("scipy.sparse")

from dvcl_bench.artifacts import CleanGraphArtifact
from dvcl_bench.attacks import generate_rnd_attack, verify_attack
from dvcl_bench.splits import build_split_artifact


def clean_fixture():
    forward = sp.csr_matrix(
        np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]])
    )
    return CleanGraphArtifact(
        dataset="acm",
        version="acm-test",
        predict_ntype="paper",
        node_counts={"paper": 4, "author": 4},
        hete_adjs={"pa": forward, "ap": forward.T.tocsr()},
        features=torch.eye(4),
        labels=torch.tensor([0, 0, 1, 1]),
        num_classes=2,
        meta_paths=[["pa", "ap"]],
        canonical_etypes=[("paper", "pa", "author"), ("author", "ap", "paper")],
        stats={},
    )


def test_split_seed_is_deterministic_and_masks_cover_nodes():
    clean = clean_fixture()
    one = build_split_artifact(clean, 7, "random", 0.5, 0.25, 0.25)
    two = build_split_artifact(clean, 7, "random", 0.5, 0.25, 0.25)
    assert torch.equal(one.train_idx, two.train_idx)
    assert torch.all(one.train_mask.int() + one.val_mask.int() + one.test_mask.int() == 1)


def test_random_attack_is_deterministic_and_reverse_consistent():
    clean = clean_fixture()
    split = build_split_artifact(clean, 1, "random", 0.5, 0.25, 0.25)
    one = generate_rnd_attack(clean, split, 25, 11)
    two = generate_rnd_attack(clean, split, 25, 11)
    assert (one.perturbed_hete_adjs["pa"] != two.perturbed_hete_adjs["pa"]).nnz == 0
    assert (one.perturbed_hete_adjs["pa"].T != one.perturbed_hete_adjs["ap"]).nnz == 0
    assert verify_attack(clean, split, one)["ok"]
