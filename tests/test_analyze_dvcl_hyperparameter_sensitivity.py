from scripts import analyze_dvcl_hyperparameter_sensitivity as ANALYZER


def _rows():
    rows = []
    variants = [("reference", "reference", None)]
    for factor, spec in ANALYZER.FACTOR_SPECS.items():
        variants.extend(
            (f"{factor}_{value}", factor, float(value))
            for value in spec["values"]
            if float(value) != float(spec["default"])
        )
    for variant, factor, value in variants:
        for attack, rate in ANALYZER.CONDITIONS:
            for attack_seed, train_seed in ANALYZER.SEED_PAIRS:
                offset = 0.0 if factor == "reference" else -0.0001 * float(value)
                rows.append({
                    "variant": variant,
                    "factor": factor,
                    "value": value,
                    "attack": attack,
                    "rate": rate,
                    "attack_seed": attack_seed,
                    "train_seed": train_seed,
                    "micro_f1": 0.9 - (0.05 if attack != "clean" else 0.0)
                    + offset + 0.001 * train_seed,
                    "best_epoch": 1,
                    "run_dir": "unused",
                })
    return rows


def test_matrix_summary_and_stability_cover_frozen_design():
    rows = _rows()
    assert len(rows) == 120
    assert ANALYZER.validate_matrix(rows) == []
    summary = ANALYZER.summarize_rows(rows)
    assert len(summary) == 48
    stability = ANALYZER.stability_assessment(summary)
    assert len(stability) == 5
    assert all(row["locally_stable"] for row in stability)


def test_stability_loss_is_zero_when_neighbors_outperform_reference():
    summary = ANALYZER.summarize_rows(_rows())
    for attack, rate in ANALYZER.CONDITIONS:
        reference = next(
            row for row in summary
            if row["factor"] == "heads"
            and row["value"] == 4.0
            and row["attack"] == attack
            and row["rate"] == rate
        )
        for value in (2.0, 8.0):
            neighbor = next(
                row for row in summary
                if row["factor"] == "heads"
                and row["value"] == value
                and row["attack"] == attack
                and row["rate"] == rate
            )
            neighbor["micro_f1_mean"] = reference["micro_f1_mean"] + 0.01

    stability = ANALYZER.stability_assessment(summary)
    heads = next(row for row in stability if row["factor"] == "heads")
    assert heads["max_local_loss"] == 0.0


def test_variant_definitions_require_one_factor_only():
    base = {
        spec["config_key"]: spec["default"]
        for spec in ANALYZER.FACTOR_SPECS.values()
    }
    variants = [{"name": "reference", "factor": "reference", "value": None,
                 "model_config": {}}]
    for factor, spec in ANALYZER.FACTOR_SPECS.items():
        for value in spec["values"]:
            if float(value) == float(spec["default"]):
                continue
            variants.append({
                "name": f"{factor}_{value}",
                "factor": factor,
                "value": value,
                "model_config": {spec["config_key"]: value},
            })
    definitions, issues = ANALYZER.variant_definitions(
        {"variants": variants}, base
    )
    assert len(definitions) == 20
    assert issues == []


def test_report_uses_only_micro_f1_and_freezes_reference():
    rows = _rows()
    summary = ANALYZER.summarize_rows(rows)
    stability = ANALYZER.stability_assessment(summary)
    audit = {
        "physical_runs": "120/120",
        "manifest_count": 120,
        "dirty_manifests": 0,
        "issues": [],
    }
    report = ANALYZER.render_report(summary, stability, audit)
    assert "Micro-F1" in report
    assert "Macro" not in report
    assert "配对 Δ" in report
    assert "不根据结果修改已冻结的论文主模型" in report


def test_partial_paired_effects_wait_for_matching_reference():
    rows = [row for row in _rows() if row["factor"] != "reference"][:1]
    assert ANALYZER.paired_effects(rows) == []
