import argparse
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
from dvcl_bench.attacks import generate_rnd_attack, import_prbcd_like_attack
from dvcl_bench.paths import ExperimentLayout


def parse_args():
    parser = argparse.ArgumentParser(description="Generate or import an attack artifact.")
    parser.add_argument("--dataset", required=True, choices=["acm", "dblp"])
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--clean-path")
    parser.add_argument("--split", default="paper_seed_1")
    parser.add_argument("--split-path")
    parser.add_argument("--attack", required=True, choices=["rnd", "prbcd", "heteprbcd"])
    parser.add_argument("--attack-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mode", default="generate", choices=["generate", "import"])
    parser.add_argument("--source-file")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    layout = ExperimentLayout(ROOT)
    clean_path = Path(args.clean_path) if args.clean_path else (
        layout.clean_path(args.dataset)
        if data_root == ROOT / "data"
        else data_root / "processed" / args.dataset / "clean.pt"
    )
    split_path = Path(args.split_path) if args.split_path else (
        layout.split_path(args.dataset, args.split)
        if data_root == ROOT / "data"
        else data_root / "splits" / args.dataset / f"{args.split}.pt"
    )
    clean = load_clean_artifact(clean_path)
    split = load_split_artifact(split_path)
    if args.attack == "rnd" and args.mode == "generate":
        artifact = generate_rnd_attack(clean, split, args.attack_rate, args.seed)
    elif args.attack in {"prbcd", "heteprbcd"} and args.mode == "import":
        if not args.source_file:
            raise ValueError("--source-file is required when importing PRBCD/HetePRBCD")
        artifact = import_prbcd_like_attack(
            clean, split, args.attack, args.attack_rate, args.seed, Path(args.source_file)
        )
    else:
        raise ValueError(f"Unsupported attack/mode: {args.attack}/{args.mode}")
    output = Path(args.output) if args.output else (
        layout.attack_path(args.dataset, args.attack, args.attack_rate, args.seed)
        if data_root == ROOT / "data"
        else data_root / "attacks" / args.dataset / args.attack
        / f"rate_{args.attack_rate:g}" / f"seed_{args.seed}" / "attack.pt"
    )
    save_attack_artifact(artifact, output)
    print(f"Saved attack artifact: {output}")
    print(f"Stats: {artifact.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
