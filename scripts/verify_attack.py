import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.artifacts import load_attack_artifact, load_clean_artifact, load_split_artifact
from dvcl_bench.attacks import verify_attack
from dvcl_bench.paths import ExperimentLayout


def parse_args():
    parser = argparse.ArgumentParser(description="Verify an attack artifact.")
    parser.add_argument("--dataset", required=True, choices=["acm", "dblp"])
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--clean-path")
    parser.add_argument("--split", default="paper_seed_1")
    parser.add_argument("--split-path")
    parser.add_argument("--attack-path", required=True)
    parser.add_argument("--report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    layout = ExperimentLayout(ROOT)
    data_root = Path(args.data_root)
    default_clean = (
        layout.clean_path(args.dataset)
        if data_root == ROOT / "data"
        else data_root / "processed" / args.dataset / "clean.pt"
    )
    default_split = (
        layout.split_path(args.dataset, args.split)
        if data_root == ROOT / "data"
        else data_root / "splits" / args.dataset / f"{args.split}.pt"
    )
    clean = load_clean_artifact(Path(args.clean_path) if args.clean_path else default_clean)
    split = load_split_artifact(
        Path(args.split_path) if args.split_path else default_split
    )
    attack = load_attack_artifact(Path(args.attack_path))
    report = verify_attack(clean, split, attack)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
