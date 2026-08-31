import pytest

from scripts.analyze_paper_results import (
    STATISTICAL_CONFIG,
    _ablation_significance,
    _ablation_rows,
    _ablation_table_lines,
    _audit_figure_outputs,
    _load_statistical_protocol,
    _model_efficiency_lines,
    _poisoning_drop_rows,
)
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
    assert family["effect_ci_low_pp"] <= family["effect_pp"]
    assert family["effect_ci_high_pp"] >= family["effect_pp"]
    assert family["correction_family"] == "acm:prbcd"


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


def test_model_efficiency_lines_separate_training_and_inference_claims():
    summary = []
    queries = []
    for dataset in ("acm", "dblp", "aminer"):
        for model, training, inference in (
            ("hseco", 20.0, 2.0),
            ("dvcl", 10.0, 3.0),
        ):
            summary.append({
                "dataset": dataset,
                "model": model,
                "parameter_millions": 1.0,
                "training_seconds_mean": training,
                "training_seconds_std": 0.1,
                "seconds_per_iteration_mean": training / 100,
                "seconds_per_iteration_std": 0.001,
                "inference_latency_ms_mean_mean": inference,
                "inference_latency_ms_mean_std": 0.1,
                "peak_allocated_mib_mean": 100.0,
                "peak_allocated_mib_std": 1.0,
                "micro_f1_mean": 0.8,
                "micro_f1_std": 0.01,
            })
            queries.append({
                "dataset": dataset,
                "model": model,
                "rate": 5,
                "queries_per_target_mean": 100.0,
                "queries_per_target_std": 2.0,
            })
    result = {
        "summary": summary,
        "queries": queries,
        "capacity": [
            {
                "heads": heads,
                "state_elements": heads * 1000,
                "relative_to_k4": heads / 4,
            }
            for heads in (1, 2, 4, 8)
        ],
    }
    document = "\n".join(_model_efficiency_lines(result))
    assert "## 6. 效率与资源" in document
    assert "Micro-F1" in document
    assert "Macro" not in document
    assert "训练更快" in document
    assert "不支持“训练与推理全面更高效”" in document


def test_frozen_statistical_protocol_declares_post_hoc_families():
    config = _load_statistical_protocol(STATISTICAL_CONFIG)
    assert config["status"] == "post_hoc_frozen"
    assert config["poisoning"]["expected_comparisons"] == 18
    assert config["ablation"]["expected_comparisons"] == 12
    assert config["adaptive"]["expected_comparisons"] == 30
    assert len(config["figures"]) == 9


def test_ablation_significance_preserves_five_seed_pairing():
    rows = []
    for dataset in ("acm", "dblp"):
        for train_seed in (1, 2, 3, 4, 5):
            for variant, offset in (
                ("full", 0.0),
                ("no_cl", -0.01),
                ("topology_only", -0.02),
                ("feature_only", -0.03),
            ):
                for attack, score in (
                    ("clean", 0.9 + offset),
                    ("prbcd", 0.8 + offset),
                    ("heteprbcd", 0.7 + offset),
                ):
                    rows.append({
                        "protocol": f"{dataset}_poisoning_ablation_v1",
                        "dataset": dataset,
                        "variant": variant,
                        "attack": attack,
                        "train_seed": train_seed,
                        "micro_f1": score,
                    })
    result = _ablation_significance(rows)
    assert len(result) == 12
    assert all(row["n"] == 5 for row in result)
    assert all(row["effect_ci_low_pp"] <= row["effect_pp"] for row in result)
    assert {row["correction_family"] for row in result} == {
        "acm:clean", "acm:all", "dblp:clean", "dblp:all",
    }


def test_poisoning_drop_summary_has_fifteen_paired_repeats():
    models = ("han", "heterosage", "hseco", "dvcl")
    clean = []
    attacked = []
    for dataset in ("acm", "dblp"):
        protocol = f"{dataset}_poisoning_main_v1"
        for model in models:
            for train_seed in (1, 2, 3, 4, 5):
                clean.append({
                    "protocol": protocol,
                    "dataset": dataset,
                    "model": model,
                    "attack": "clean",
                    "train_seed": train_seed,
                    "micro_f1": 0.9,
                })
                for attack in ("prbcd", "heteprbcd"):
                    for rate in (5, 15, 25):
                        for attack_seed in (1, 2, 3):
                            attacked.append({
                                "dataset": dataset,
                                "model": model,
                                "attack": attack,
                                "rate": rate,
                                "attack_seed": attack_seed,
                                "train_seed": train_seed,
                                "micro_f1": 0.8,
                            })
    result = _poisoning_drop_rows(attacked, clean)
    assert len(result) == 48
    assert all(row["n"] == 15 for row in result)
    assert all(row["drop_pp_mean"] == pytest.approx(10.0) for row in result)


def test_figure_audit_requires_png_and_pdf(tmp_path):
    for name in ("first", "second"):
        for suffix in (".png", ".pdf"):
            (tmp_path / f"{name}{suffix}").write_bytes(b"figure")
    audit = _audit_figure_outputs(tmp_path, ("first", "second"))
    assert audit["ok"] is True
    assert audit["completed_files"] == 4
