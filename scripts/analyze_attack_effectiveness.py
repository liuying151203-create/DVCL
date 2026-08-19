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
    parser.add_argument("--clean-protocol", action="append", default=[])
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
            manifest_path = path.with_name("manifest.json")
            manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file() else {}
            )
            groups[key].append({
                "micro_f1": float(payload["metrics"]["micro_f1"]),
                "inputs": manifest.get("inputs", {}),
            })

    clean = {}
    for key, samples in groups.items():
        protocol, dataset, model, attack, _, rate = key
        if attack == "clean" and rate == 0:
            clean[(protocol, dataset, model)] = statistics.mean(
                sample["micro_f1"] for sample in samples
            )
    clean_fallback = _clean_fallback(args.clean_protocol)

    rows = []
    missing_diagnostics = []
    for key in sorted(groups):
        protocol, dataset, model, attack, attack_variant, rate = key
        if attack == "clean":
            continue
        samples = groups[key]
        values = [sample["micro_f1"] for sample in samples]
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
            "clean_micro_f1_mean": clean.get(
                (protocol, dataset, model), clean_fallback.get((dataset, model))
            ),
        }
        if row["clean_micro_f1_mean"] is not None:
            row["drop_pp"] = 100 * (
                row["clean_micro_f1_mean"] - row["micro_f1_mean"]
            )
        artifact_path, clean_path, split_path = _input_paths(
            samples, layout, dataset, attack, rate
        )
        if artifact_path.is_file():
            artifact = load_attack_artifact(artifact_path)
            split = load_split_artifact(split_path)
            report = verify_attack(load_clean_artifact(clean_path), split, artifact)
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


def _clean_fallback(protocols):
    result = defaultdict(list)
    for protocol in protocols:
        root = ExperimentLayout(ROOT).outputs / "runs" / protocol
        for path in root.rglob("metrics.json") if root.is_dir() else []:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("attack") == "clean"
                and float(payload.get("rate", 0)) == 0
                and "micro_f1" in payload.get("metrics", {})
            ):
                result[(payload["dataset"], payload["model"])].append(
                    float(payload["metrics"]["micro_f1"])
                )
    return {key: statistics.mean(values) for key, values in result.items()}


def _input_paths(samples, layout, dataset, attack, rate):
    names = ("attack", "clean", "split")
    discovered = {
        name: {
            sample["inputs"].get(name, {}).get("path")
            for sample in samples
            if sample["inputs"].get(name, {}).get("path")
        }
        for name in names
    }
    for name, paths in discovered.items():
        if len(paths) > 1:
            raise RuntimeError(
                f"Experiment group uses multiple {name} artifacts: {sorted(paths)}"
            )
    defaults = {
        "attack": layout.attack_path(dataset, attack, rate, 1),
        "clean": layout.clean_path(dataset),
        "split": layout.split_path(dataset, "paper_seed_1"),
    }
    return tuple(
        Path(next(iter(discovered[name]))) if discovered[name] else defaults[name]
        for name in names
    )


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
