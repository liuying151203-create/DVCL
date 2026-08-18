from pathlib import Path

from dvcl_bench.manifest import build_manifest
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
