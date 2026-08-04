from copy import deepcopy

import pytest

from dvcl_bench.reporting import _best_cells, summarize_family, validate_matrix


def _row(seed, attack, rate, value):
    return {
        "dataset": "acm",
        "model": "dvcl",
        "variant": "default",
        "attack": attack,
        "rate": rate,
        "train_seed": seed,
        "accuracy": value,
        "micro_f1": value,
    }


def test_family_average_is_computed_within_each_seed_first():
    rows = [
        _row(1, "prbcd", 5, 0.6),
        _row(1, "prbcd", 10, 0.8),
        _row(2, "prbcd", 5, 0.8),
        _row(2, "prbcd", 10, 1.0),
    ]
    mean, std = summarize_family(rows, ("prbcd",))
    assert mean == pytest.approx(0.8)
    assert std == pytest.approx(2 ** 0.5 / 10)


def test_matrix_validation_rejects_duplicate_or_missing_seed():
    rows = [_row(1, "clean", 0, 0.8), _row(1, "clean", 0, 0.9)]
    identities = [("acm", "dvcl", "clean", 0.0)]
    with pytest.raises(ValueError, match="train seeds"):
        validate_matrix(rows, identities, (1, 2))


def test_matrix_validation_rejects_accuracy_micro_mismatch():
    row = _row(1, "clean", 0, 0.8)
    changed = deepcopy(row)
    changed["micro_f1"] = 0.7
    with pytest.raises(ValueError, match="Accuracy and Micro-F1 differ"):
        validate_matrix([changed], [("acm", "dvcl", "clean", 0.0)], (1,))


def test_best_cells_bolds_the_best_model_only():
    values = {"han": (0.8, 0.1), "dvcl": (0.9, 0.2)}
    assert _best_cells(values, ("han", "dvcl")) == [
        "80.00 ± 10.00",
        "**90.00 ± 20.00**",
    ]
