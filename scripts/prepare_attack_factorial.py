import argparse
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


VARIANTS = {
    "prbcd_unconstrained": {
        "attack": "prbcd", "constrained": False, "biased": False,
    },
    "heteprbcd_unbiased": {
        "attack": "heteprbcd", "constrained": True, "biased": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare constraint and bias factorial attack artifacts."
    )
    parser.add_argument("--datasets", nargs="+", default=["acm", "dblp"])
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=sorted(VARIANTS))
    parser.add_argument("--rates", nargs="+", type=int, default=[5, 15, 25])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for dataset in args.datasets:
        clean = load_clean_artifact(ROOT / "data" / "processed" / dataset / "clean.pt")
        split = load_split_artifact(ROOT / "data" / "splits" / dataset / "paper_seed_1.pt")
        for variant in args.variants:
            setting = VARIANTS[variant]
            for rate in args.rates:
                source = ROOT / "outputs" / "attack_sources" / variant / dataset / f"rate_{rate}" / "source.pt"
                artifact_path = ROOT / "data" / "attack_diagnostics" / dataset / variant / f"rate_{rate}" / f"seed_{args.seed}" / "attack.pt"
                if artifact_path.exists() and not args.force:
                    print(f"Skip existing {artifact_path}")
                    continue
                if not args.skip_generation:
                    _generate(source, dataset, rate, args.seed, args.cuda, setting)
                if not source.is_file():
                    raise FileNotFoundError(source)
                artifact = import_prbcd_like_attack(
                    clean, split, setting["attack"], rate, args.seed, source
                )
                artifact.provenance["diagnostic_variant"] = variant
                report = verify_attack(clean, split, artifact)
                if not report["ok"]:
                    raise RuntimeError("; ".join(report["issues"]))
                save_attack_artifact(artifact, artifact_path)
                save_json(report, artifact_path.with_name("verification.json"))
                print(f"Prepared {artifact_path}")
    return 0


def _generate(output, dataset, rate, seed, cuda, setting):
    command = [
        sys.executable,
        str(ROOT / "scripts" / "generate_prbcd_diagnostic_source.py"),
        "--dataset", dataset,
        "--attack", setting["attack"],
        "--rate", str(rate),
        "--seed", str(seed),
        "--cuda", str(cuda),
        "--output", str(output),
    ]
    if setting["constrained"]:
        command.append("--constrained")
    if setting["biased"]:
        command.append("--biased")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
