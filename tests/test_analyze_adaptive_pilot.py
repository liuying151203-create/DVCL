import pytest

from scripts.analyze_adaptive_pilot import (
    adaptive_average_ranks,
    aggregate_rows,
    choose_candidate,
    paired_reference_comparisons,
    validate_candidate_hashes,
)


def pilot_rows():
    rows = []
    for size, attacked, success in (
        (16, 0.60, 0.30),
        (64, 0.50, 0.50),
        (128, 0.495, 0.505),
    ):
        for attack_seed in (1, 2):
            rows.append({
                "dataset": "acm",
                "model": "han",
                "candidate_size": size,
                "rate": 5,
                "train_seed": 1,
                "attack_seed": attack_seed,
                "clean_target_micro_f1": 0.8,
                "attacked_target_micro_f1": attacked,
                "micro_f1_drop": 0.8 - attacked,
                "attack_success_rate": success,
                "clean_correct_targets": 40,
                "attack_success_count": 20,
                "total_changes": 200,
                "budget_utilization": 0.8,
                "queries": size * 10,
                "candidate_pool_sha256": f"hash-{size}-{attack_seed}",
                "run_dir": "run",
            })
    return rows


def test_pilot_aggregation_and_selection_choose_smallest_stable_pool():
    summary = aggregate_rows(pilot_rows())
    selected, diagnostics = choose_candidate(summary, tolerance=0.02)
    assert selected == 64
    assert diagnostics[-1]["stable"] is True


def test_candidate_hash_audit_is_model_independent():
    rows = pilot_rows()
    duplicate = dict(rows[0])
    duplicate["model"] = "dvcl"
    rows.append(duplicate)
    assert validate_candidate_hashes(rows) == []
    rows[-1]["candidate_pool_sha256"] = "different"
    assert len(validate_candidate_hashes(rows)) == 1


def test_adaptive_ranks_and_paired_comparisons_preserve_seed_pairing():
    rows = []
    for seed, dvcl, han in ((1, 0.9, 0.7), (2, 0.8, 0.7), (3, 0.7, 0.7)):
        for rate in (1, 3, 5):
            for model, score in (("dvcl", dvcl), ("han", han)):
                rows.append({
                    "dataset": "acm",
                    "model": model,
                    "candidate_size": 64,
                    "rate": rate,
                    "train_seed": seed,
                    "attack_seed": seed,
                    "clean_target_micro_f1": 0.9,
                    "attacked_target_micro_f1": score,
                    "micro_f1_drop": 0.9 - score,
                    "attack_success_rate": 0.5,
                    "clean_correct_targets": 45,
                    "attack_success_count": 20,
                    "total_changes": 50,
                    "budget_utilization": 0.5,
                    "queries": 100,
                    "candidate_pool_sha256": f"hash-{seed}",
                    "run_dir": "run",
                })
    ranks = adaptive_average_ranks(rows)
    assert next(row for row in ranks if row["model"] == "dvcl")["average_rank"] == 7 / 6
    comparison = paired_reference_comparisons(rows)[0]
    assert comparison["n"] == 3
    assert comparison["effect_pp"] == pytest.approx(10)
    assert (comparison["wins"], comparison["ties"], comparison["losses"]) == (2, 1, 0)
    assert comparison["effect_ci_low_pp"] <= comparison["effect_pp"]
    assert comparison["effect_ci_high_pp"] >= comparison["effect_pp"]
