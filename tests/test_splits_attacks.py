import pickle

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
sp = pytest.importorskip("scipy.sparse")

from dvcl_bench.artifacts import CleanGraphArtifact
from dvcl_bench.attacks import (
    _budget_report,
    build_attack_artifact,
    generate_rnd_attack,
    validate_attack_context,
    verify_attack,
)
from dvcl_bench.splits import build_split_artifact, import_split_artifact


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
    report = verify_attack(clean, split, one)
    assert report["ok"]
    assert one.stats["_global"]["actual_rate"] == pytest.approx(0.25)
    assert report["budget"]["ok"]
    assert report["split_perturbation"]["predict_ntype"] == "paper"


def test_attack_verification_warns_about_training_split_concentration():
    size = 20
    forward = sp.eye(size, format="csr", dtype=np.int8)
    clean = CleanGraphArtifact(
        dataset="acm",
        version="acm-concentration-test",
        predict_ntype="paper",
        node_counts={"paper": size, "author": size},
        hete_adjs={"pa": forward, "ap": forward.T.tocsr()},
        features=torch.eye(size),
        labels=torch.arange(size) % 2,
        num_classes=2,
        meta_paths=[["pa", "ap"]],
        canonical_etypes=[("paper", "pa", "author"), ("author", "ap", "paper")],
        stats={},
    )
    split = build_split_artifact(clean, 1, "random", 0.1, 0.1, 0.8)
    train_node = int(split.train_idx[0])
    perturbed = forward.tolil(copy=True)
    candidates = [index for index in range(size) if index != train_node][:10]
    perturbed[train_node, candidates] = 1
    perturbed = perturbed.tocsr()
    attack = build_attack_artifact(
        clean,
        split,
        "heteprbcd",
        50,
        1,
        {"pa": perturbed, "ap": perturbed.T.tocsr()},
        None,
        "test",
    )
    report = verify_attack(clean, split, attack)
    train_stats = report["split_perturbation"]["_global"]["train"]
    assert report["ok"]
    assert train_stats["change_share"] == pytest.approx(1.0)
    assert train_stats["enrichment"] == pytest.approx(10.0)
    assert report["warnings"]


def test_budget_report_allows_small_underuse_but_rejects_overuse():
    under_budget = _budget_report(
        10, {"clean_edges": 1000, "n_add": 99, "n_del": 0, "actual_rate": 0.099}
    )
    over_budget = _budget_report(
        10, {"clean_edges": 1000, "n_add": 101, "n_del": 0, "actual_rate": 0.101}
    )
    excessive_shortfall = _budget_report(
        10, {"clean_edges": 1000, "n_add": 97, "n_del": 0, "actual_rate": 0.097}
    )
    assert under_budget["ok"]
    assert under_budget["shortfall"] == 1
    assert not over_budget["ok"]
    assert not excessive_shortfall["ok"]


def test_attack_context_rejects_a_different_split():
    clean = clean_fixture()
    split = build_split_artifact(clean, 1, "random", 0.5, 0.25, 0.25)

    class Store:
        x = clean.features
        y = clean.labels
        train_mask = split.val_mask
        val_mask = split.train_mask
        test_mask = split.test_mask

    class Source:
        node_types = ["paper"]

        def __getitem__(self, name):
            assert name == "paper"
            return Store()

    with pytest.raises(ValueError, match="train_mask"):
        validate_attack_context(clean, split, Source())


def test_imports_legacy_hseco_split_list(tmp_path):
    clean = clean_fixture()
    train = torch.tensor([True, False, False, False])
    val = torch.tensor([False, True, False, False])
    test = torch.tensor([False, False, True, True])
    source = tmp_path / "legacy_split.pkl"
    with source.open("wb") as stream:
        pickle.dump(
            [
                torch.nonzero(train).view(-1).numpy(),
                torch.nonzero(val).view(-1).numpy(),
                torch.nonzero(test).view(-1).numpy(),
                train.numpy(),
                val.numpy(),
                test.numpy(),
                clean.labels.numpy(),
            ],
            stream,
        )
    split = import_split_artifact(clean, source, "legacy", seed=3)
    assert split.protocol == "imported"
    assert torch.equal(split.train_mask, train)
    assert torch.equal(split.val_mask, val)
    assert torch.equal(split.test_mask, test)
