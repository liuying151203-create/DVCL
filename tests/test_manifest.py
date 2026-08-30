import subprocess
from pathlib import Path

from dvcl_bench.manifest import build_manifest, git_dirty
from dvcl_bench.specs import AttackSpec, ExperimentSpec, ModelSpec, SeedSpec


def test_manifest_records_runtime_environment(tmp_path: Path):
    artifact = tmp_path / "clean.pt"
    artifact.write_bytes(b"artifact")
    spec = ExperimentSpec(
        protocol="test",
        dataset="acm",
        split_name="paper_seed_1",
        seeds=SeedSpec(split=1, attack=1, train=1),
        attack=AttackSpec(name="clean"),
        model=ModelSpec(name="hseco", backend="native"),
        device="cpu",
    )
    manifest = build_manifest(spec, tmp_path, {"clean": artifact})
    assert manifest["schema_version"] == 2
    assert manifest["environment"]["python"]["version"]
    assert "torch" in manifest["environment"]["packages"]
    assert "openhgnn" in manifest["environment"]["packages"]
    assert "cuda_available" in manifest["environment"]["accelerator"]
    assert "dgl_backend" in manifest["environment"]["accelerator"]


def test_git_dirty_ignores_untracked_files_but_detects_tracked_changes(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "DVCL Test"],
        cwd=tmp_path,
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "test fixture"], cwd=tmp_path, check=True
    )

    (tmp_path / "untracked.txt").write_text("ignored", encoding="utf-8")
    assert git_dirty(tmp_path) is False

    tracked.write_text("changed", encoding="utf-8")
    assert git_dirty(tmp_path) is True
