import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.artifacts import (
    load_attack_artifact,
    load_clean_artifact,
    load_split_artifact,
)
from dvcl_bench.graph_adapter import hete_adjs_to_dgl


def parse_args():
    parser = argparse.ArgumentParser(description="Run legacy training with native artifacts.")
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--entrypoint", required=True)
    return parser.parse_known_args()


def load_artifact_experiment_data(args):
    clean = load_clean_artifact(Path(args.clean_artifact_path))
    split = load_split_artifact(Path(args.split_artifact_path))
    if clean.dataset != args.dataname or split.dataset != args.dataname:
        raise ValueError(f"Golden artifact dataset mismatch: {args.dataname}")

    hete_adjs = clean.hete_adjs
    if args.attack_artifact_path:
        attack = load_attack_artifact(Path(args.attack_artifact_path))
        if attack.dataset != args.dataname:
            raise ValueError(f"Golden attack dataset mismatch: {attack.dataset}")
        if attack.clean_version != clean.version:
            raise ValueError("Golden attack clean version mismatch")
        if attack.split_name != split.split_name:
            raise ValueError("Golden attack split mismatch")
        hete_adjs = attack.perturbed_hete_adjs

    graph = hete_adjs_to_dgl(hete_adjs, clean.canonical_etypes, clean.node_counts)
    return (
        graph,
        hete_adjs,
        clean.features,
        clean.labels,
        clean.num_classes,
        split.train_mask,
        split.val_mask,
        split.test_mask,
    )


def main() -> None:
    args, remaining = parse_args()
    reference_root = Path(args.reference_root).resolve()
    entrypoint = reference_root / args.entrypoint
    if not entrypoint.is_file():
        raise FileNotFoundError(f"Legacy entrypoint not found: {entrypoint}")

    sys.path.insert(0, str(reference_root))
    from core import experiment_data

    experiment_data.load_artifact_experiment_data = load_artifact_experiment_data
    sys.argv = [str(entrypoint), *remaining]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
