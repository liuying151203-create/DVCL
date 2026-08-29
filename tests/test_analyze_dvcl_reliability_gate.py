import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_dvcl_reliability_gate",
    ROOT / "scripts" / "analyze_dvcl_reliability_gate.py",
)
ANALYZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZE)


def _clean(variant, acm, dblp, aminer, gate_std=0.1):
    return [
        {
            "dataset": dataset,
            "variant": variant,
            "full_test_micro_f1": value,
            "gate_mean": 0.5,
            "gate_std": gate_std,
            "gate_topology_fraction": 0.0,
            "gate_feature_fraction": 0.0,
        }
        for dataset, value in zip(ANALYZE.DATASETS, (acm, dblp, aminer))
    ]


def _summary(variant, dblp=0.42, acm=0.85, aminer=0.70):
    rows = []
    for dataset, value in (("acm", acm), ("dblp", dblp), ("aminer", aminer)):
        rows.append({
            "dataset": dataset,
            "variant": variant,
            "attack": "adaptive_query",
            "rate": 5,
            "attacked_target_micro_f1_mean": value,
        })
    return rows


def test_stage_e_decision_requires_gain_clean_safety_and_noncollapse():
    baseline_clean = _clean("concat", 0.89, 0.89, 0.88)
    candidate_clean = _clean("reliability_gate", 0.88, 0.88, 0.87)
    baseline = _summary("concat")
    candidate = _summary("reliability_gate", dblp=0.48)
    decision = ANALYZE.stage_e_decision(
        candidate_clean, candidate, baseline_clean, baseline
    )
    assert decision["passes"] is True
    collapsed = _clean(
        "reliability_gate", 0.88, 0.88, 0.87, gate_std=0.0
    )
    decision = ANALYZE.stage_e_decision(
        collapsed, candidate, baseline_clean, baseline
    )
    assert decision["passes"] is False
    assert decision["next_action"] == "implement_reliability_gate_aug_pilot"


def test_failed_augmented_candidate_closes_model_iteration():
    baseline_clean = _clean("concat", 0.89, 0.89, 0.88)
    candidate_clean = _clean("reliability_gate_aug", 0.88, 0.88, 0.87)
    baseline = _summary("concat")
    candidate = _summary("reliability_gate_aug", dblp=0.40)
    decision = ANALYZE.stage_e_decision(
        candidate_clean, candidate, baseline_clean, baseline,
        "reliability_gate_aug",
    )
    assert decision["passes"] is False
    assert decision["next_action"] == "retain_concat_and_close_stage_e"


def test_stage_e_decision_supports_augmented_candidate_variant():
    baseline_clean = _clean("concat", 0.89, 0.89, 0.88)
    candidate_clean = _clean("reliability_gate_aug", 0.88, 0.88, 0.87)
    baseline = _summary("concat")
    candidate = _summary("reliability_gate_aug", dblp=0.48)
    decision = ANALYZE.stage_e_decision(
        candidate_clean, candidate, baseline_clean, baseline,
        "reliability_gate_aug",
    )
    assert decision["passes"] is True
