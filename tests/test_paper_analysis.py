import pytest

from scripts.analyze_paper_results import _ablation_rows, _ablation_table_lines
from dvcl_bench.paper_analysis import (
    average_ranks,
    holm_adjust,
    paired_significance,
    target_summary,
)


def test_holm_adjust_is_monotonic_in_sorted_order():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_paired_significance_preserves_pairing_and_counts_outcomes():
    rows = []
    for seed, dvcl, han in ((1, 0.9, 0.8), (2, 0.7, 0.7), (3, 0.6, 0.8)):
        for model, score in (("dvcl", dvcl), ("han", han)):
            rows.append({
                "dataset": "acm",
                "attack": "prbcd",
                "rate": 5.0,
                "attack_seed": 1,
                "train_seed": seed,
                "model": model,
                "micro_f1": score,
            })
    result = paired_significance(rows)
    family = next(row for row in result if row["attack"] == "prbcd")
    assert family["n"] == 3
    assert (family["wins"], family["ties"], family["losses"]) == (1, 1, 1)
    assert family["effect_pp"] == pytest.approx(-100 / 30)


def test_average_ranks_handles_ties():
    rows = [
        {
            "dataset": "acm", "attack": "prbcd", "rate": 5.0,
            "attack_seed": 1, "train_seed": 1, "model": model,
            "micro_f1": score,
        }
        for model, score in (("dvcl", 0.9), ("han", 0.8), ("hseco", 0.8))
    ]
    result = average_ranks(rows)
    assert next(row for row in result if row["dataset"] == "acm" and row["model"] == "dvcl")["average_rank"] == 1
    assert next(row for row in result if row["dataset"] == "acm" and row["model"] == "han")["average_rank"] == 2.5


def test_target_summary_uses_paired_clean_target_scores():
    rows = [
        {
            "protocol": "target", "dataset": "acm", "model": "dvcl",
            "attack": "adaptive", "rate": 5.0, "micro_f1": attacked,
            "diagnostics": {"clean_target_metrics": {"micro_f1": clean}},
        }
        for clean, attacked in ((0.9, 0.8), (0.8, 0.6))
    ]
    result = target_summary(rows)[0]
    assert result["clean_micro_f1_mean"] == pytest.approx(0.85)
    assert result["micro_f1_mean"] == pytest.approx(0.7)
    assert result["drop_pp_mean"] == pytest.approx(15.0)


def test_ablation_summary_keeps_acm_and_dblp_separate():
    rows = []
    for dataset, clean, attacked in (
        ("acm", 0.9, 0.8),
        ("dblp", 0.7, 0.6),
    ):
        for variant in ("full", "no_cl", "topology_only", "feature_only"):
            rows.extend([
                {
                    "protocol": f"{dataset}_poisoning_ablation_v1",
                    "dataset": dataset,
                    "variant": variant,
                    "attack": "clean",
                    "train_seed": 1,
                    "micro_f1": clean,
                },
                {
                    "protocol": f"{dataset}_poisoning_ablation_v1",
                    "dataset": dataset,
                    "variant": variant,
                    "attack": "prbcd",
                    "train_seed": 1,
                    "micro_f1": attacked,
                },
                {
                    "protocol": f"{dataset}_poisoning_ablation_v1",
                    "dataset": dataset,
                    "variant": variant,
                    "attack": "heteprbcd",
                    "train_seed": 1,
                    "micro_f1": attacked,
                },
            ])
    summary = _ablation_rows(rows)
    assert next(
        row for row in summary
        if row["dataset"] == "acm"
        and row["variant"] == "full"
        and row["condition"] == "clean"
    )["micro_f1_mean"] == pytest.approx(0.9)
    dblp_lines = _ablation_table_lines(summary, "dblp")
    assert dblp_lines[0].startswith("| Full DVCL | 70.00")
