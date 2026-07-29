import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args():
    parser = argparse.ArgumentParser(description="Compare legacy and native golden artifacts/runs.")
    parser.add_argument("--kind", required=True, choices=["clean", "split", "attack", "metrics"])
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--legacy-root")
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.legacy_root:
        sys.path.insert(0, str(Path(args.legacy_root).resolve()))
    from dvcl_bench import equivalence
    if args.kind == "metrics":
        report = equivalence.compare_metrics(
            Path(args.reference), Path(args.current), args.tolerance
        )
    else:
        from dvcl_bench.artifacts import (
            load_attack_artifact,
            load_clean_artifact,
            load_split_artifact,
        )
        loaders = {
            "clean": load_clean_artifact,
            "split": load_split_artifact,
            "attack": load_attack_artifact,
        }
        comparisons = {
            "clean": equivalence.compare_clean,
            "split": equivalence.compare_split,
            "attack": equivalence.compare_attack,
        }
        report = comparisons[args.kind](
            loaders[args.kind](Path(args.reference)),
            loaders[args.kind](Path(args.current)),
        )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
