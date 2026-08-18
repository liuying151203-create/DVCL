import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_results", ROOT / "scripts" / "summarize_results.py"
)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def row(seed, attack, rate, accuracy):
    return {
        "protocol": "main",
        "dataset": "acm",
        "model": "dvcl",
        "variant": "default",
        "attack": attack,
        "attack_variant": "default",
        "rate": rate,
        "split_seed": 1,
        "attack_seed": 1,
        "train_seed": seed,
        "accuracy": accuracy,
        "micro_f1": accuracy,
        "macro_f1": accuracy,
    }


def test_summary_groups_train_seeds_and_attack_conditions():
    rows = [
        row(1, "prbcd", 5, 0.8),
        row(1, "heteprbcd", 5, 0.6),
        row(2, "prbcd", 5, 1.0),
        row(2, "heteprbcd", 5, 0.8),
    ]
    grouped = SUMMARY.aggregate(rows)
    assert len(grouped) == 2
    attack_average = SUMMARY.attack_averages(rows)
    assert len(attack_average) == 1
    assert attack_average[0]["accuracy_mean"] == 0.8


def test_attack_variants_are_not_merged():
    rows = [
        {**row(1, "prbcd", 5, 0.8), "attack_variant": "default"},
        {**row(1, "prbcd", 5, 0.5), "attack_variant": "unconstrained"},
    ]
    assert len(SUMMARY.aggregate(rows)) == 2
    assert len(SUMMARY.attack_averages(rows)) == 2
