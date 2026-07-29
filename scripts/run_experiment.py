import argparse
import csv
import json
import shlex
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.legacy import build_legacy_command, run_legacy
from dvcl_bench.manifest import build_manifest, save_json
from dvcl_bench.paths import ExperimentLayout
from dvcl_bench.specs import AttackSpec, ExperimentSpec, ModelSpec, SeedSpec


def parse_args():
    parser = argparse.ArgumentParser(description="Run one frozen DVCL benchmark experiment.")
    parser.add_argument("--protocol", default="dvcl_main")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="native", choices=["legacy", "native", "openhgnn"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--attack", default="clean")
    parser.add_argument("--rate", type=float, default=0)
    parser.add_argument("--threat-model", default="poisoning", choices=["poisoning", "evasion"])
    parser.add_argument("--scope", default="global", choices=["global", "target"])
    parser.add_argument("--split-name", default="paper_seed_1")
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--attack-seed", type=int, default=1)
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--model-config-json", default="{}")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args, extra = parser.parse_known_args()
    return args, extra


def build_spec(args, extra):
    return ExperimentSpec(
        protocol=args.protocol,
        dataset=args.dataset.lower(),
        split_name=args.split_name,
        seeds=SeedSpec(args.split_seed, args.attack_seed, args.train_seed),
        attack=AttackSpec(args.attack.lower(), args.rate, args.threat_model, args.scope),
        model=ModelSpec(args.model.lower(), args.backend, json.loads(args.model_config_json)),
        device=args.device,
        epochs=args.epochs,
        patience=args.patience,
        extra_args=tuple(extra),
    )


def input_paths(spec, layout):
    values = {
        "clean": layout.clean_path(spec.dataset),
        "split": layout.split_path(spec.dataset, spec.split_name),
    }
    if spec.attack.name != "clean":
        values["attack"] = layout.attack_path(
            spec.dataset, spec.attack.name, spec.attack.rate, spec.seeds.attack
        )
    return values


def main() -> int:
    args, extra = parse_args()
    spec = build_spec(args, extra)
    layout = ExperimentLayout(ROOT)
    run_dir = layout.run_dir(spec)
    print(json.dumps(asdict(spec), ensure_ascii=False, indent=2))
    print(f"run_dir: {run_dir}")
    if args.dry_run:
        return 0
    if spec.model.backend == "native" and spec.extra_args:
        raise SystemExit(
            "Native runs reject unknown CLI arguments; use --model-config-json for model options"
        )
    if spec.model.backend == "openhgnn":
        raise SystemExit("OpenHGNN model execution is not implemented yet")
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "metrics.json"
    if not args.force and status_path.exists() and metrics_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") == "completed":
            print(f"Skip completed run: {run_dir}")
            return 0

    inputs = input_paths(spec, layout)
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_json({
            "state": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": "Missing frozen experiment artifacts: " + ", ".join(missing),
        }, status_path)
        print("Missing frozen experiment artifacts: " + ", ".join(missing), file=sys.stderr)
        return 1
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(build_manifest(spec, ROOT, inputs), run_dir / "manifest.json")
    save_json({
        "state": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }, status_path)
    try:
        if spec.model.backend == "legacy":
            command = build_legacy_command(spec, layout, args.python_bin)
            print(shlex.join(command))
            returncode = run_legacy(command, layout.legacy_hseco)
            if returncode:
                raise RuntimeError(f"Legacy runner returned {returncode}")
            result_payload = {"backend": "legacy"}
        else:
            result_payload = run_native(spec, inputs, run_dir)
        save_json(result_payload, metrics_path)
        save_json({
            "state": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "best_epoch": result_payload.get("best_epoch"),
        }, status_path)
        return 0
    except Exception as exc:
        save_json({
            "state": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }, status_path)
        print(traceback.format_exc(), file=sys.stderr)
        return 1


def run_native(spec, inputs, run_dir):
    from dvcl_bench.artifacts import (
        load_attack_artifact,
        load_clean_artifact,
        load_split_artifact,
    )
    from dvcl_bench.attacks import verify_attack
    from dvcl_bench.registry import build_model_config, get_native_trainer

    clean = load_clean_artifact(inputs["clean"])
    split = load_split_artifact(inputs["split"])
    attack = None
    if "attack" in inputs:
        attack = load_attack_artifact(inputs["attack"])
        report = verify_attack(clean, split, attack)
        save_json(report, run_dir / "attack_verification.json")
        if not report["ok"]:
            raise ValueError("Attack verification failed: " + "; ".join(report["issues"]))
    config = build_model_config(spec.model.name, spec.model.config)
    trainer = get_native_trainer(spec.model.name)
    result = trainer(
        clean=clean,
        split=split,
        attack=attack,
        config=config,
        train_seed=spec.seeds.train,
        epochs=spec.epochs,
        patience=spec.patience,
        device=spec.device,
        checkpoint_path=run_dir / "checkpoint.pt",
    )
    write_history(result.history, run_dir / "history.csv")
    return {
        "protocol": spec.protocol,
        "dataset": spec.dataset,
        "model": spec.model.name,
        "variant": spec.model.config.get("variant", "default"),
        "attack": spec.attack.name,
        "rate": spec.attack.rate,
        "split_seed": spec.seeds.split,
        "attack_seed": spec.seeds.attack,
        "train_seed": spec.seeds.train,
        "metrics": result.metrics,
        "best_epoch": result.best_epoch,
        "stopped_epoch": result.stopped_epoch,
        "diagnostics": result.diagnostics,
    }


def write_history(rows, path):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
