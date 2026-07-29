import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.artifacts import import_legacy_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a legacy HSeCo artifact.")
    parser.add_argument("--kind", required=True, choices=["clean", "split", "attack"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--legacy-root")
    args = parser.parse_args()
    if args.legacy_root:
        sys.path.insert(0, str(Path(args.legacy_root).resolve()))
    import_legacy_artifact(Path(args.source), args.kind, Path(args.output))
    print(f"Converted {args.kind} artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
