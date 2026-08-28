from scripts.analyze_dvcl_view_diagnosis import render_report, stage_e_decision


def _rows():
    clean = []
    summary = []
    clean_values = {
        "topo": 0.80,
        "feat": 0.78,
        "concat": 0.85,
        "gate": 0.845,
        "gated_concat": 0.82,
    }
    for dataset in ("acm", "dblp", "aminer"):
        for variant, value in clean_values.items():
            clean.append({
                "dataset": dataset,
                "variant": variant,
                "full_test_micro_f1": value,
            })
            for attack in ("hg_baseline", "adaptive_query"):
                for rate in (1, 3, 5):
                    attacked = 0.8
                    if dataset == "dblp" and attack == "adaptive_query":
                        attacked = {
                            "topo": 0.3,
                            "feat": 0.75,
                            "concat": 0.4,
                            "gate": 0.5,
                            "gated_concat": 0.35,
                        }[variant]
                    summary.append({
                        "dataset": dataset,
                        "variant": variant,
                        "attack": attack,
                        "rate": rate,
                        "clean_target_micro_f1_mean": 0.8,
                        "attacked_target_micro_f1_mean": attacked,
                        "micro_f1_drop_mean": 0.8 - attacked,
                        "attack_success_rate_mean": 0.1,
                        "drift_topology_l2_mean_mean": 1.0,
                        "drift_feature_l2_mean_mean": 0.0,
                        "clean_view_disagreement_rate_mean": 0.2,
                        "attacked_view_disagreement_rate_mean": 0.3,
                        "gate_clean_mean_mean": 0.6,
                        "gate_attacked_mean_mean": 0.5,
                    })
    return clean, summary


def test_stage_e_decision_selects_existing_gate_when_thresholds_pass():
    clean, summary = _rows()
    decision = stage_e_decision(clean, summary)
    assert decision["variant"] == "gate"
    assert decision["passes"] is True


def test_render_report_contains_all_required_sections():
    clean, summary = _rows()
    report = render_report(clean, summary)
    assert "# DVCL 视图失效诊断结果" in report
    assert "## 2. Clean Micro-F1" in report
    assert "## 3. HG Baseline 目标逃逸" in report
    assert "## 4. 模型自适应目标逃逸" in report
    assert "## 5. 视图诊断" in report
    assert "将 `gate` 扩展到 3 个配对种子" in report
