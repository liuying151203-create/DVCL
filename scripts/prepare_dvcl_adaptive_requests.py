import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.artifacts import (
    file_sha256,
    load_attack_artifact,
    load_clean_artifact,
    load_split_artifact,
    save_attack_artifact,
)
from dvcl_bench.attacks import TARGET_ATTACK_SPECS, build_attack_artifact, verify_attack
from dvcl_bench.manifest import save_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare frozen target sets for DVCL adaptive query attacks."
    )
    parser.add_argument("--datasets", nargs="+", default=["acm", "dblp"])
    parser.add_argument("--rates", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--candidate-additions", type=int, default=16)
    parser.add_argument("--candidate-deletions", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def stratified_targets(candidates, labels, count, seed):
    candidates = np.asarray(sorted(set(int(value) for value in candidates)), dtype=np.int64)
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    selected = []
    classes = sorted(np.unique(labels[candidates]).tolist())
    base, remainder = divmod(min(count, len(candidates)), len(classes))
    for index, label in enumerate(classes):
        values = candidates[labels[candidates] == label].copy()
        rng.shuffle(values)
        selected.extend(values[:base + (index < remainder)].tolist())
    if len(selected) < min(count, len(candidates)):
        remaining = np.asarray(sorted(set(candidates.tolist()) - set(selected)))
        rng.shuffle(remaining)
        selected.extend(remaining[:count - len(selected)].tolist())
    return torch.tensor(sorted(selected), dtype=torch.long)


def main() -> int:
    args = parse_args()
    for dataset in args.datasets:
        clean_path = ROOT / "data" / "processed" / dataset / "clean.pt"
        split_path = ROOT / "data" / "splits" / dataset / "paper_seed_1.pt"
        hg_path = (
            ROOT / "data" / "attacks" / dataset / "hg_baseline"
            / "rate_5" / "seed_1" / "attack.pt"
        )
        clean = load_clean_artifact(clean_path)
        split = load_split_artifact(split_path)
        hg = load_attack_artifact(hg_path)
        candidates = hg.target_nodes.tolist() if hg.target_nodes is not None else []
        targets = stratified_targets(candidates, clean.labels.numpy(), args.target_count, args.seed)
        if not bool(split.test_mask[targets].all()):
            raise ValueError("Adaptive request targets must belong to the test split")
        spec = TARGET_ATTACK_SPECS[dataset]
        records = [{
            "target": int(target),
            "relation": spec["relation"],
            "reverse_relation": spec["reverse"],
            "target_position": int(spec["target_position"]),
            "deleted": [],
            "added": [],
        } for target in targets.tolist()]
        for rate in args.rates:
            output = (
                ROOT / "data" / "attacks" / dataset / "dvcl_adaptive_query"
                / f"rate_{rate}" / f"seed_{args.seed}" / "attack.pt"
            )
            if output.is_file() and not args.force:
                print(f"Skip existing {output}")
                continue
            artifact = build_attack_artifact(
                clean, split, "dvcl_adaptive_query", rate, args.seed,
                clean.hete_adjs, targets, str(hg_path.resolve()), file_sha256(hg_path),
                provenance={
                    "generator": "scripts/prepare_dvcl_adaptive_requests.py",
                    "request_only": True,
                    "target_source": str(hg_path.resolve()),
                    "target_source_sha256": file_sha256(hg_path),
                    "target_count": len(targets),
                    "candidate_additions": args.candidate_additions,
                    "candidate_deletions": args.candidate_deletions,
                    "victim_model": "dvcl",
                    "objective": "maximize max_other_logit_minus_true_logit",
                },
                threat_model="evasion", scope="target", adaptive=True,
                target_changes=records,
            )
            report = verify_attack(clean, split, artifact)
            if not report["ok"]:
                raise RuntimeError("; ".join(report["issues"]))
            save_attack_artifact(artifact, output)
            save_json(report, output.with_name("verification.json"))
            print(f"Prepared {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
