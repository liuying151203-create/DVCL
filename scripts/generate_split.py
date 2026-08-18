import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.artifacts import load_clean_artifact, save_split_artifact
from dvcl_bench.paths import ExperimentLayout
from dvcl_bench.splits import build_split_artifact, import_split_artifact


def parse_args():
    parser = argparse.ArgumentParser(description="Generate or import a frozen data split.")
    parser.add_argument("--dataset", required=True, choices=["acm", "dblp", "aminer"])
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--clean-path")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--protocol", default="paper", choices=["paper", "random", "imported"])
    parser.add_argument("--train-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.8)
    parser.add_argument("--source-file")
    parser.add_argument("--split-name")
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
    clean = load_clean_artifact(clean_path)
    if args.protocol == "imported":
        if not args.source_file:
            raise ValueError("--source-file is required for imported splits")
        split = import_split_artifact(
            clean,
            Path(args.source_file),
            args.split_name or f"imported_seed_{args.seed}",
            args.seed,
        )
    else:
        split = build_split_artifact(
            clean,
            args.seed,
            args.protocol,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
        )
        if args.split_name:
            split.split_name = args.split_name
    output = Path(args.output) if args.output else (
        layout.split_path(args.dataset, split.split_name)
        if data_root == ROOT / "data"
        else data_root / "splits" / args.dataset / f"{split.split_name}.pt"
    )
    save_split_artifact(split, output)
    print(f"Saved split artifact: {output}")
    print(f"Stats: {split.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
