import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
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
from dvcl_bench.attacks import verify_attack
from dvcl_bench.paths import ExperimentLayout


FIELDS = [
    "protocol", "dataset", "model", "attack", "attack_variant", "rate",
    "runs", "micro_f1_mean", "micro_f1_std", "clean_micro_f1_mean",
    "drop_pp", "actual_rate", "changes", "train_change_share",
    "train_enrichment", "constrained", "biased", "adaptive",
    "surrogate_before_micro_f1", "surrogate_after_micro_f1",
    "surrogate_drop_pp", "optimization_loss_first", "optimization_loss_last",
    "generation_diagnostics",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit whether frozen attacks produce measurable model degradation."
    )
    parser.add_argument("--protocol", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strict-generation-diagnostics", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    layout = ExperimentLayout(ROOT)
    groups = defaultdict(list)
    for protocol in args.protocol:
        root = layout.outputs / "runs" / protocol
        for path in root.rglob("metrics.json") if root.is_dir() else []:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "metrics" not in payload or "micro_f1" not in payload["metrics"]:
                continue
            key = (
                protocol, payload["dataset"], payload["model"], payload["attack"],
                payload.get("attack_variant", "default"), float(payload["rate"]),
            )
            groups[key].append(float(payload["metrics"]["micro_f1"]))

    clean = {}
    for key, values in groups.items():
        protocol, dataset, model, attack, _, rate = key
        if attack == "clean" and rate == 0:
            clean[(protocol, dataset, model)] = statistics.mean(values)

    rows = []
    missing_diagnostics = []
    for key in sorted(groups):
        protocol, dataset, model, attack, attack_variant, rate = key
        if attack == "clean":
            continue
        values = groups[key]
        row = {
            "protocol": protocol,
            "dataset": dataset,
            "model": model,
            "attack": attack,
            "attack_variant": attack_variant,
            "rate": rate,
            "runs": len(values),
            "micro_f1_mean": statistics.mean(values),
            "micro_f1_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "clean_micro_f1_mean": clean.get((protocol, dataset, model)),
        }
        if row["clean_micro_f1_mean"] is not None:
            row["drop_pp"] = 100 * (
                row["clean_micro_f1_mean"] - row["micro_f1_mean"]
            )
        artifact_path = layout.attack_path(dataset, attack, rate, 1)
        if artifact_path.is_file():
            artifact = load_attack_artifact(artifact_path)
            split = load_split_artifact(layout.split_path(dataset, "paper_seed_1"))
            report = verify_attack(load_clean_artifact(layout.clean_path(dataset)), split, artifact)
            global_stats = artifact.stats.get("_global", {})
            train_stats = report["split_perturbation"]["_global"]["train"]
            provenance = artifact.provenance
            row.update({
                "actual_rate": global_stats.get("actual_rate"),
                "changes": global_stats.get("n_add", 0) + global_stats.get("n_del", 0),
                "train_change_share": train_stats.get("change_share"),
                "train_enrichment": train_stats.get("enrichment"),
                "constrained": provenance.get("constrained"),
                "biased": provenance.get("biased"),
                "adaptive": artifact.adaptive,
            })
            _generation_diagnostics(row, provenance)
            if attack in {"prbcd", "heteprbcd"} and row["generation_diagnostics"] != "complete":
                missing_diagnostics.append(str(artifact_path))
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps({
            "rows": rows,
            "missing_generation_diagnostics": sorted(set(missing_diagnostics)),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output} rows={len(rows)}")
    if args.strict_generation_diagnostics and missing_diagnostics:
        print(
            f"Missing surrogate/history diagnostics for {len(set(missing_diagnostics))} artifacts",
            file=sys.stderr,
        )
        return 2
    return 0


def _generation_diagnostics(row, provenance):
    before = provenance.get("surrogate_before", {})
    after = provenance.get("surrogate_after", {})
    history = provenance.get("optimization_history", [])
    row["surrogate_before_micro_f1"] = before.get("micro_f1")
    row["surrogate_after_micro_f1"] = after.get("micro_f1")
    if row["surrogate_before_micro_f1"] is not None and row["surrogate_after_micro_f1"] is not None:
        row["surrogate_drop_pp"] = 100 * (
            row["surrogate_before_micro_f1"] - row["surrogate_after_micro_f1"]
        )
    row["optimization_loss_first"] = history[0] if history else None
    row["optimization_loss_last"] = history[-1] if history else None
    row["generation_diagnostics"] = (
        "complete" if before and after and history else "missing"
    )


if __name__ == "__main__":
    raise SystemExit(main())
