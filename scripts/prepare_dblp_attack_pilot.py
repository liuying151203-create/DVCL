import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.artifacts import (
    load_clean_artifact,
    load_split_artifact,
    save_attack_artifact,
)
from dvcl_bench.attacks import import_prbcd_like_attack, verify_attack
from dvcl_bench.manifest import save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare corrected attack artifacts.")
    parser.add_argument("--dataset", choices=["acm", "dblp"], default="dblp")
    parser.add_argument("--legacy-root", default=str(ROOT.parent / "HSeCo"))
    parser.add_argument(
        "--output-root", default=str(ROOT / "outputs" / "pilots" / "attack_protocol")
    )
    parser.add_argument("--rates", nargs="+", type=int, default=[5, 25])
    parser.add_argument(
        "--attacks",
        nargs="+",
        choices=["prbcd", "heteprbcd"],
        default=["prbcd", "heteprbcd"],
    )
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--attack-seed", type=int, default=1)
    parser.add_argument("--conda-env", default="hseco")
    parser.add_argument("--skip-generation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    legacy_root = Path(args.legacy_root).resolve()
    output_root = Path(args.output_root).resolve()
    clean = load_clean_artifact(
        ROOT / "data" / "processed" / args.dataset / "clean.pt"
    )
    split = load_split_artifact(
        ROOT / "data" / "splits" / args.dataset / "paper_seed_1.pt"
    )
    if not args.skip_generation:
        require_gpu()
    for attack in args.attacks:
        for rate in args.rates:
            source = (
                output_root
                / "sources"
                / args.dataset
                / attack
                / f"rate_{rate}"
                / "source.pt"
            )
            artifact_path = (
                output_root
                / "artifacts"
                / args.dataset
                / attack
                / f"rate_{rate}"
                / "attack.pt"
            )
            clear_published_artifact(artifact_path)
            if not args.skip_generation:
                generate_source(
                    legacy_root,
                    source,
                    attack,
                    rate,
                    args.dataset,
                    args.attack_seed,
                    args.gpu_id,
                    args.conda_env,
                )
            if not source.is_file():
                raise FileNotFoundError(f"Missing pilot source: {source}")
            artifact = import_prbcd_like_attack(
                clean, split, attack, rate, args.attack_seed, source
            )
            validate_provenance(
                artifact.provenance, args.dataset, attack, rate, args.attack_seed
            )
            report = verify_attack(clean, split, artifact)
            rejected_report = (
                output_root
                / "audits"
                / args.dataset
                / attack
                / f"rate_{rate}"
                / "verification.json"
            )
            save_json(report, rejected_report)
            if not report["ok"]:
                raise RuntimeError("; ".join(report["issues"]))
            save_attack_artifact(artifact, artifact_path)
            save_json(report, artifact_path.with_name("verification.json"))
            print(f"Prepared {attack} {rate}%: {artifact_path}")
            for warning in report["warnings"]:
                print(f"Warning: {warning}")
    return 0


def require_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("NVIDIA driver check failed; restore GPU access before generation") from exc
    if result.returncode:
        raise RuntimeError("NVIDIA driver is unavailable; restore GPU access before generation")


def generate_source(
    legacy_root, output, attack, rate, dataset, attack_seed, gpu_id, conda_env
):
    output.parent.mkdir(parents=True, exist_ok=True)
    script = legacy_root / "scripts" / f"gen_{dataset}_{attack}.sh"
    environment = os.environ.copy()
    environment.update({
        "ATK_RATE": str(rate),
        "SEED": str(attack_seed),
        "GPU_ID": str(gpu_id),
        "CONDA_ENV": conda_env,
        "DATA_ROOT": str(legacy_root / "data"),
        "HETEROGUARD_ROOT": str(legacy_root.parent / "Hetero-Guard"),
        "OUTPUT": str(output),
    })
    subprocess.run(["bash", str(script)], cwd=legacy_root, env=environment, check=True)


def validate_provenance(provenance, dataset, attack, rate, attack_seed):
    expected_biased = attack == "heteprbcd"
    expected = {
        "attack": "HetePRBCD" if expected_biased else "PRBCD",
        "dataset": dataset,
        "rate": rate,
        "seed": attack_seed,
        "constrained": True,
        "biased": expected_biased,
    }
    mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Pilot source provenance mismatch: {json.dumps(mismatches)}")


def clear_published_artifact(artifact_path):
    for path in (
        artifact_path,
        artifact_path.with_name("meta.json"),
        artifact_path.with_name("verification.json"),
    ):
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
