import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.equivalence import compare_legacy_training_log
from dvcl_bench.golden import (
    audit_payload,
    build_current_command,
    build_reference_command,
    build_spec,
    input_paths,
    load_golden_cases,
    load_golden_config,
    resolve_model_config,
    resolve_reference_root,
)
from dvcl_bench.manifest import save_json
from dvcl_bench.paths import ExperimentLayout


def parse_args():
    parser = argparse.ArgumentParser(description="Run audited legacy/native golden cases.")
    parser.add_argument("--config", default="configs/golden/hseco_dvcl.yaml")
    parser.add_argument("--reference-root")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_golden_config(config_path)
    reference_root = resolve_reference_root(args.reference_root, ROOT)
    layout = ExperimentLayout(ROOT)
    output_root = ROOT / str(config.get("output_root", "outputs/equivalence/model_golden"))
    protocol = str(config.get("protocol", "model_golden"))
    device = str(config.get("device", "cuda:0"))
    training = dict(config.get("training", {}))
    epochs = int(training.get("epochs", 200))
    patience = int(training.get("patience", 100))
    tolerance = float(config.get("tolerance", 0))
    failures = 0
    summary = []

    for case in load_golden_cases(config):
        model_config = resolve_model_config(config, case.model, ROOT)
        spec = build_spec(case, model_config, protocol, device, epochs, patience)
        inputs = input_paths(spec, layout)
        missing = [str(path) for path in inputs.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing golden input artifacts: " + ", ".join(missing))
        case_dir = output_root / case.case_id
        reference_stdout = case_dir / "reference_stdout.log"
        reference_metrics = case_dir / "reference_metrics.log"
        current_dir = layout.run_dir(spec)
        reference_command = build_reference_command(
            spec, reference_root, args.python_bin, inputs, reference_metrics
        )
        current_command = build_current_command(spec, ROOT, args.python_bin, args.force)
        print(f"[{case.case_id}] reference: {shlex.join(reference_command)}")
        print(f"[{case.case_id}] current:   {shlex.join(current_command)}")
        if args.dry_run:
            continue

        case_dir.mkdir(parents=True, exist_ok=True)
        audit = audit_payload(
            case, spec, inputs, reference_root, reference_command, current_command
        )
        dirty = [
            name for name, state in audit["repositories"].items() if state["dirty"]
        ]
        if dirty and not args.allow_dirty:
            raise RuntimeError(
                "Strict golden runs require clean repositories; dirty: " + ", ".join(dirty)
            )
        save_json(audit, case_dir / "audit.json")
        try:
            if args.force or not reference_stdout.is_file() or not reference_metrics.is_file():
                with reference_stdout.open("w", encoding="utf-8") as stream:
                    completed = subprocess.run(
                        reference_command,
                        cwd=str(reference_root),
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if completed.returncode:
                    raise RuntimeError(
                        f"Reference run returned {completed.returncode}: {reference_stdout}"
                    )
            completed = subprocess.run(current_command, cwd=str(ROOT), check=False)
            if completed.returncode:
                raise RuntimeError(f"Current run returned {completed.returncode}: {current_dir}")
            report = compare_legacy_training_log(
                reference_stdout,
                current_dir / "history.csv",
                current_dir / "metrics.json",
                tolerance,
            )
            report.update({"case_id": case.case_id, "tolerance": tolerance})
            save_json(report, case_dir / "report.json")
            summary.append(report)
            if not report["ok"]:
                raise RuntimeError("Golden comparison failed: " + "; ".join(report["issues"]))
        except Exception as exc:
            failures += 1
            failure = {"case_id": case.case_id, "ok": False, "error": str(exc)}
            if not any(item.get("case_id") == case.case_id for item in summary):
                summary.append(failure)
            save_json(failure, case_dir / "failure.json")
            if not args.continue_on_error:
                save_json({"cases": summary}, output_root / "summary.json")
                raise

    if not args.dry_run:
        save_json({"cases": summary}, output_root / "summary.json")
    print(f"Golden cases: {len(load_golden_cases(config))}, failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
