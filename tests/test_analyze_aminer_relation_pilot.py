import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "analyze_aminer_relation_pilot.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_aminer_relation_pilot", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _summary(pa_drop=0.03, pr_drop=0.01, joint_drop=0.02):
    rows = []
    for scope, drop in (("pa", pa_drop), ("pr", pr_drop), ("joint", joint_drop)):
        rows.append({
            "relation_scope": scope,
            "n": 4,
            "mean_downstream_micro_f1_drop": drop,
            "min_downstream_micro_f1_drop": drop - 0.01,
            "prbcd_surrogate_micro_f1_drop": 0.01,
            "heteprbcd_surrogate_micro_f1_drop": 0.02,
        })
    return rows


def test_choose_scope_uses_common_downstream_score():
    decision = MODULE.choose_scope(_summary())
    assert decision["selected_relation_scope"] == "pa"
    assert decision["passes"] is True


def test_choose_scope_rejects_negative_surrogate_drop():
    summary = _summary()
    summary[0]["prbcd_surrogate_micro_f1_drop"] = -0.001
    decision = MODULE.choose_scope(summary)
    assert decision["selected_relation_scope"] == "pa"
    assert decision["passes"] is False


def test_choose_scope_requires_complete_matrix():
    summary = _summary()
    summary[1]["n"] = 3
    assert MODULE.choose_scope(summary) is None
