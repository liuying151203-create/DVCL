import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_adaptive_requests import prepare_requests


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare all candidate-pool requests for adaptive attack pilot."
    )
    parser.add_argument("--datasets", nargs="+", default=["acm", "dblp"])
    parser.add_argument("--rates", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--target-seed", type=int, default=1)
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--candidate-sizes", nargs="+", type=int, default=[16, 64, 128])
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "outputs" / "pilots" / "adaptive_requests"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for size in args.candidate_sizes:
        prepare_requests(SimpleNamespace(
            datasets=args.datasets,
            rates=args.rates,
            seeds=args.seeds,
            target_seed=args.target_seed,
            target_count=args.target_count,
            candidate_additions=size,
            candidate_deletions=size,
            output_root=str(
                Path(args.output_root) / f"cand_{size}"
            ),
            force=args.force,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
