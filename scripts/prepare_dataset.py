import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.artifacts import save_clean_artifact
from dvcl_bench.datasets import build_clean_artifact
from dvcl_bench.paths import ExperimentLayout


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a versioned clean graph artifact.")
    parser.add_argument("--dataset", required=True, choices=["acm", "dblp"])
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--version", default="v1")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    artifact = build_clean_artifact(args.dataset, data_root, args.version)
    output = Path(args.output) if args.output else ExperimentLayout(ROOT).clean_path(args.dataset)
    if args.data_root != str(ROOT / "data") and not args.output:
        output = data_root / "processed" / args.dataset / "clean.pt"
    save_clean_artifact(artifact, output)
    print(f"Saved clean artifact: {output}")
    print(f"Stats: {artifact.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
