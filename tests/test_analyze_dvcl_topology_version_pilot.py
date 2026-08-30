from scripts import analyze_dvcl_topology_version_pilot as ANALYZER


def _training_rows(clean_offsets=None, poisoning_offsets=None):
    clean_offsets = clean_offsets or {}
    poisoning_offsets = poisoning_offsets or {}
    rows = []
    for variant in ANALYZER.VARIANTS:
        for attack, rate in ANALYZER.CONDITIONS:
            for attack_seed, train_seed in ANALYZER.SEED_PAIRS:
                offset = (
                    clean_offsets.get(variant, 0.0)
                    if attack == "clean"
                    else poisoning_offsets.get(variant, 0.0)
                )
                rows.append({
                    "variant": variant,
                    "attack": attack,
                    "rate": rate,
                    "attack_seed": attack_seed,
                    "train_seed": train_seed,
                    "micro_f1": 0.8 + offset + train_seed * 0.001,
                })
    return rows


def _adaptive_summary(graph_no_filter=0.47):
    attacked = {
        "graph_hard": 0.40,
        "graph_no_filter": graph_no_filter,
    }
    rows = []
    for variant in ANALYZER.VARIANTS:
        for rate in (1, 3, 5):
            value = attacked[variant] + (5 - rate) * 0.01
            rows.append({
                "dataset": "dblp",
                "variant": variant,
                "attack": "adaptive_query",
                "rate": rate,
                "clean_target_micro_f1_mean": 0.9,
                "attacked_target_micro_f1_mean": value,
                "attacked_target_micro_f1_std": 0.01,
                "micro_f1_drop_mean": 0.9 - value,
            })
    return rows


def test_matrix_validation_and_training_summary():
    rows = _training_rows()
    assert ANALYZER.validate_training_matrix(rows) == []
    summary = {
        (row["variant"], row["attack"]): row
        for row in ANALYZER.summarize_training(rows)
    }
    assert summary[("graph_hard", "clean")]["n"] == 3
    assert abs(summary[("graph_hard", "clean")]["micro_f1_mean"] - 0.802) < 1e-12


def test_decision_selects_candidate_that_passes_all_gates():
    decision = ANALYZER.topology_version_decision(
        _training_rows(), _adaptive_summary()
    )
    assert decision["selected_variant"] == "graph_no_filter"
    assert decision["method_change"] is True


def test_decision_retains_reference_when_clean_gate_fails():
    decision = ANALYZER.topology_version_decision(
        _training_rows(clean_offsets={"graph_no_filter": -0.02}),
        _adaptive_summary(),
    )
    assert decision["selected_variant"] == "graph_hard"
    assert decision["method_change"] is False


def test_render_report_uses_only_micro_f1_result_metric():
    training = ANALYZER.summarize_training(_training_rows())
    adaptive = _adaptive_summary()
    decision = ANALYZER.topology_version_decision(
        _training_rows(), adaptive
    )
    audit = {
        "training_physical_runs": "18/18",
        "adaptive_physical_runs": "9/9",
        "adaptive_logical_results": "27/27",
        "issues": [],
    }
    report = ANALYZER.render_report(training, adaptive, decision, audit)
    assert "Micro-F1" in report
    assert "Macro" not in report
    assert "## 版本判定" in report
