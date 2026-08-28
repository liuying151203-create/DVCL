import json
from dataclasses import asdict

from dvcl_bench.paths import ExperimentLayout
from dvcl_bench.specs import (
    AttackSpec,
    ExperimentSpec,
    ModelSpec,
    SeedSpec,
)
from scripts import analyze_dvcl_view_diagnosis as ANALYZER
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


def _manifest_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(ANALYZER, "ROOT", tmp_path)
    layout = ExperimentLayout(tmp_path)
    clean = layout.clean_path("acm")
    split = layout.split_path("acm", "paper_seed_1")
    for path, content in ((clean, b"clean"), (split, b"split")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    spec = ExperimentSpec(
        protocol="diagnosis",
        dataset="acm",
        split_name="paper_seed_1",
        seeds=SeedSpec(split=1, attack=1, train=1),
        attack=AttackSpec(),
        model=ModelSpec(
            name="dvcl", backend="native", config={"variant": "concat"}
        ),
        device="cuda:0",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    experiment = asdict(spec)
    experiment["device"] = "cuda:3"
    manifest = {
        "schema_version": 2,
        "experiment": experiment,
        "inputs": {
            "clean": {"path": str(clean), "sha256": ANALYZER.file_sha256(clean)},
            "split": {"path": str(split), "sha256": ANALYZER.file_sha256(split)},
        },
        "git_commit": "abc123",
        "git_dirty": True,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return spec, run_dir, clean


def test_manifest_audit_accepts_device_migration(tmp_path, monkeypatch):
    spec, run_dir, _ = _manifest_fixture(tmp_path, monkeypatch)
    issues = []
    manifest = ANALYZER._audit_manifest(run_dir, spec, [], issues, {})
    assert manifest is not None
    assert issues == []


def test_manifest_audit_rejects_tampered_input(tmp_path, monkeypatch):
    spec, run_dir, clean = _manifest_fixture(tmp_path, monkeypatch)
    clean.write_bytes(b"tampered")
    issues = []
    ANALYZER._audit_manifest(run_dir, spec, [], issues, {})
    assert any("manifest clean hash mismatch" in issue for issue in issues)
