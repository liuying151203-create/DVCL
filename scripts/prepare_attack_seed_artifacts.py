import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.artifacts import (
    load_clean_artifact,
    load_split_artifact,
    save_attack_artifact,
)
from dvcl_bench.attacks import import_prbcd_like_attack, verify_attack
from dvcl_bench.manifest import save_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and verify PRBCD/HetePRBCD artifacts for attack seeds."
    )
    parser.add_argument("--datasets", nargs="+", default=["acm", "dblp"])
    parser.add_argument(
        "--attacks", nargs="+", choices=["prbcd", "heteprbcd"],
        default=["prbcd", "heteprbcd"],
    )
    parser.add_argument("--rates", nargs="+", type=int, default=[5, 15, 25])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2, 3])
    parser.add_argument(
        "--relation-scopes", nargs="+",
        choices=["default", "pa", "pr", "joint"],
        default=["default"],
    )
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=100000)
    parser.add_argument("--data-root")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def source_path(dataset, attack, rate, seed, relation_scope="default"):
    if relation_scope != "default":
        return (
            ROOT / "data" / "attack_diagnostics" / dataset / "f1_relations"
            / relation_scope / attack / f"rate_{rate}" / f"seed_{seed}" / "source.pt"
        )
    return (
        ROOT / "data" / "attack_diagnostics" / dataset / attack
        / f"rate_{rate}" / f"seed_{seed}" / "source.pt"
    )


def artifact_path(dataset, attack, rate, seed, relation_scope="default"):
    if relation_scope != "default":
        return (
            ROOT / "data" / "attacks" / dataset / "f1_relations"
            / relation_scope / attack / f"rate_{rate}" / f"seed_{seed}" / "attack.pt"
        )
    return (
        ROOT / "data" / "attacks" / dataset / attack
        / f"rate_{rate}" / f"seed_{seed}" / "attack.pt"
    )


def generator_command(
    dataset, attack, rate, seed, cuda, block_size, output,
    relation_scope="default", data_root=None,
):
    command = [
        sys.executable,
        str(ROOT / "scripts" / "generate_prbcd_diagnostic_source.py"),
        "--dataset", dataset,
        "--attack", attack,
        "--rate", str(rate),
        "--seed", str(seed),
        "--cuda", str(cuda),
        "--block-size", str(block_size),
        "--output", str(output),
        "--constrained",
    ]
    if attack == "heteprbcd":
        command.append("--biased")
    if relation_scope != "default":
        command.extend(["--relation-scope", relation_scope])
    if data_root is not None:
        command.extend(["--data-root", str(Path(data_root).resolve())])
    return command


def validate_provenance(
    provenance, dataset, attack, rate, seed, relation_scope="default"
):
    expected = {
        "dataset": dataset,
        "attack": "PRBCD" if attack == "prbcd" else "HetePRBCD",
        "rate": rate,
        "seed": seed,
        "constrained": True,
        "biased": attack == "heteprbcd",
    }
    if relation_scope != "default":
        expected["relation_scope"] = relation_scope
    mismatches = {
        key: {"actual": provenance.get(key), "expected": value}
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise ValueError("Attack provenance mismatch: " + json.dumps(mismatches))
    if relation_scope != "default":
        expected_relations = {
            "pa": {"pa"},
            "pr": {"pr"},
            "joint": {"pa", "pr"},
        }[relation_scope]
        actual_relations = {
            edge_type[1] for edge_type in provenance.get("budget", [])
            if len(edge_type) == 3
        }
        if actual_relations != expected_relations:
            raise ValueError(
                "Attack provenance budget mismatch: "
                + json.dumps({
                    "actual": sorted(actual_relations),
                    "expected": sorted(expected_relations),
                })
            )


def main() -> int:
    args = parse_args()
    for dataset in args.datasets:
        invalid_scopes = [
            scope for scope in args.relation_scopes
            if scope != "default" and dataset != "aminer"
        ]
        if invalid_scopes:
            raise ValueError(
                "Explicit relation scopes are currently supported only for AMiner"
            )
        clean = load_clean_artifact(ROOT / "data" / "processed" / dataset / "clean.pt")
        split = load_split_artifact(
            ROOT / "data" / "splits" / dataset / "paper_seed_1.pt"
        )
        for relation_scope in args.relation_scopes:
            for attack in args.attacks:
                for rate in args.rates:
                    for seed in args.seeds:
                        source = source_path(
                            dataset, attack, rate, seed, relation_scope
                        )
                        destination = artifact_path(
                            dataset, attack, rate, seed, relation_scope
                        )
                        if destination.is_file() and not args.force:
                            print(f"Skip existing {destination}")
                            continue
                        if not args.skip_generation:
                            source.parent.mkdir(parents=True, exist_ok=True)
                            subprocess.run(
                                generator_command(
                                    dataset, attack, rate, seed, args.cuda,
                                    args.block_size, source, relation_scope,
                                    args.data_root,
                                ),
                                cwd=ROOT,
                                check=True,
                            )
                        if not source.is_file():
                            raise FileNotFoundError(source)
                        artifact = import_prbcd_like_attack(
                            clean, split, attack, rate, seed, source
                        )
                        validate_provenance(
                            artifact.provenance, dataset, attack, rate, seed,
                            relation_scope,
                        )
                        report = verify_attack(clean, split, artifact)
                        if not report["ok"]:
                            raise RuntimeError("; ".join(report["issues"]))
                        save_attack_artifact(artifact, destination)
                        save_json(report, destination.with_name("verification.json"))
                        print(f"Prepared {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
