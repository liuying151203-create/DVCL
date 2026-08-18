import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.artifacts import (
    load_clean_artifact,
    load_split_artifact,
    save_attack_artifact,
)
from dvcl_bench.attacks import (
    TARGET_ATTACK_SPECS,
    import_hg_baseline_attack,
    verify_attack,
)
from dvcl_bench.manifest import save_json
from dvcl_bench.paths import ExperimentLayout


SOURCE_NAMES = {
    "acm": "adv_acm_pap_pa_{rate}.pkl",
    "dblp": "adv_dblp_apa_pa_{rate}.pkl",
    "aminer": "adv_aminer_prp_pr_{rate}.pkl",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Freeze auditable HG Baseline target-evasion artifacts."
    )
    parser.add_argument("--dataset", required=True, choices=sorted(SOURCE_NAMES))
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--split", default="paper_seed_1")
    parser.add_argument("--rates", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--num-targets", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    layout = ExperimentLayout(ROOT)
    clean = load_clean_artifact(layout.clean_path(args.dataset))
    split = load_split_artifact(layout.split_path(args.dataset, args.split))
    source_root = Path(args.source_root)
    records_by_rate = {}
    sources = {}
    for rate in args.rates:
        source = source_root / SOURCE_NAMES[args.dataset].format(rate=rate)
        if not source.is_file():
            raise FileNotFoundError(source)
        sources[rate] = source
        records_by_rate[rate] = _load_unique_records(source)

    candidates = set(split.test_idx.tolist()).intersection(
        *(set(records) for records in records_by_rate.values())
    )
    valid = [
        target for target in sorted(candidates)
        if all(
            _valid_record(clean, records_by_rate[rate][target], rate)
            for rate in args.rates
        )
    ]
    if not valid:
        raise ValueError("No valid HG Baseline targets remain after protocol filtering")
    if args.num_targets > 0 and len(valid) > args.num_targets:
        rng = np.random.default_rng(args.seed)
        targets = sorted(
            int(value) for value in rng.choice(valid, args.num_targets, replace=False)
        )
    else:
        targets = valid[: args.num_targets] if args.num_targets > 0 else valid

    target_root = ROOT / "data" / "attacks" / args.dataset / "hg_baseline"
    target_root.mkdir(parents=True, exist_ok=True)
    save_json({
        "dataset": args.dataset,
        "split": args.split,
        "selection_seed": args.seed,
        "requested_targets": args.num_targets,
        "candidate_targets": len(candidates),
        "valid_targets": len(valid),
        "selected_targets": len(targets),
        "target_nodes": targets,
        "rates": args.rates,
        "sources": {str(rate): str(path.resolve()) for rate, path in sources.items()},
    }, target_root / "target_set.json")

    for rate in args.rates:
        output = layout.attack_path(args.dataset, "hg_baseline", rate, args.seed)
        if output.exists() and not args.force:
            raise FileExistsError(f"Artifact exists; pass --force to replace: {output}")
        artifact = import_hg_baseline_attack(
            clean, split, rate, args.seed, sources[rate], targets
        )
        report = verify_attack(clean, split, artifact)
        if not report["ok"]:
            raise ValueError("HG Baseline verification failed: " + "; ".join(report["issues"]))
        save_attack_artifact(artifact, output)
        save_json(report, output.with_name("verification.json"))
        print(f"Saved {output} targets={len(targets)} budget={rate}")
    return 0


def _load_unique_records(path):
    with Path(path).open("rb") as stream:
        values = pickle.load(stream)
    records = {}
    for value in values:
        target = int(value[0])
        records.setdefault(target, value)
    return records


def _valid_record(clean, record, budget):
    spec = TARGET_ATTACK_SPECS[clean.dataset]
    target = int(record[0])
    deleted = record[2]
    added = record[3]
    if len(deleted) + len(added) > budget:
        return False
    adjacency = clean.hete_adjs[spec["relation"]]
    position = int(spec["target_position"])
    for kind, edges in (("deleted", deleted), ("added", added)):
        for raw_row, raw_column in edges:
            row, column = int(raw_row), int(raw_column)
            if row < 0 or column < 0 or row >= adjacency.shape[0] or column >= adjacency.shape[1]:
                return False
            if (row, column)[position] != target:
                return False
            exists = bool(adjacency[row, column])
            if (kind == "deleted" and not exists) or (kind == "added" and exists):
                return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
