from scripts.analyze_adaptive_pilot import (
    aggregate_rows,
    choose_candidate,
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
