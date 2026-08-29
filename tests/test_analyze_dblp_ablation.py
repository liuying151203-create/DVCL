import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_dblp_ablation.py"
SPEC = importlib.util.spec_from_file_location("analyze_dblp_ablation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _rows():
    rows = []
    offsets = {
        "full": 0.0,
        "no_cl": -0.01,
        "topology_only": -0.02,
        "feature_only": -0.03,
    }
    for variant in MODULE.VARIANTS:
        for attack, rate in MODULE.CONDITIONS:
            for seed in MODULE.TRAIN_SEEDS:
                value = 0.90 + offsets[variant] + seed * 0.001
                if variant != "feature_only" and attack != "clean":
                    value -= rate * 0.0001
                rows.append({
                    "variant": variant,
                    "attack": attack,
                    "rate": rate,
                    "train_seed": seed,
                    "micro_f1": value,
                })
    return rows


def test_validate_matrix_and_feature_invariance():
    rows = _rows()
    assert MODULE.validate_matrix(rows) == []
    assert MODULE.validate_feature_invariance(rows) == []


def test_family_summary_averages_within_seed_first():
    summary = {
        (row["variant"], row["family"]): row
        for row in MODULE.family_summary(_rows())
    }
    full_prbcd = summary[("full", "prbcd")]
    assert full_prbcd["n"] == 5
    assert abs(full_prbcd["micro_f1_mean"] - 0.9015) < 1e-12


def test_paired_effects_report_full_minus_ablation():
    effects = {
        (row["variant"], row["family"]): row
        for row in MODULE.paired_effects(_rows())
    }
    assert abs(effects[("no_cl", "clean")]["full_gain_pp_mean"] - 1.0) < 1e-12
    assert effects[("no_cl", "all")]["wins"] == 5


def test_validate_matrix_rejects_missing_row():
    issues = MODULE.validate_matrix(_rows()[:-1])
    assert issues and "missing matrix rows" in issues[0]
