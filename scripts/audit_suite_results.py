import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.paths import ExperimentLayout
from dvcl_bench.specs import AttackSpec, ExperimentSpec, ModelSpec, SeedSpec


def parse_args():
    parser = argparse.ArgumentParser(description="Audit completion of an expanded suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    module = _load_run_suite()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = module.load_config(config_path)
    expected = []
    for command in module.commands(config, sys.executable, ROOT):
        options = _options(command[2:])
        spec = ExperimentSpec(
            protocol=options["--protocol"],
            dataset=options["--dataset"],
            split_name=options["--split-name"],
            seeds=SeedSpec(
                int(options["--split-seed"]),
                int(options["--attack-seed"]),
                int(options["--train-seed"]),
            ),
            attack=AttackSpec(
                options["--attack"], float(options["--rate"]),
                options["--threat-model"], options["--scope"],
                "--adaptive" in command, options["--attack-variant"],
            ),
            model=ModelSpec(
                options["--model"], options["--backend"],
                json.loads(options["--model-config-json"]),
            ),
            device=options["--device"],
            epochs=int(options["--epochs"]),
            patience=int(options["--patience"]),
        )
        expected.append(ExperimentLayout(ROOT).run_dir(spec))

    completed, failed, incomplete, missing = [], [], [], []
    required = ("manifest.json", "metrics.json", "history.csv", "checkpoint.pt", "status.json")
    for run_dir in expected:
        status_path = run_dir / "status.json"
        if not status_path.is_file():
            missing.append(str(run_dir))
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") == "failed":
            failed.append({"run_dir": str(run_dir), "error": status.get("error")})
            continue
        absent = [name for name in required if not (run_dir / name).is_file()]
        if status.get("state") != "completed" or absent:
            incomplete.append({"run_dir": str(run_dir), "missing_files": absent})
            continue
        completed.append(str(run_dir))
    report = {
        "config": str(config_path.resolve()),
        "expected": len(expected),
        "completed": len(completed),
        "failed": failed,
        "incomplete": incomplete,
        "missing": missing,
        "ok": len(completed) == len(expected),
    }
    output = Path(args.output) if args.output else (
        ROOT / "outputs" / "audits" / f"{config_path.stem}_results.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"expected={report['expected']} completed={report['completed']} "
        f"failed={len(failed)} incomplete={len(incomplete)} missing={len(missing)}"
    )
    print(f"Wrote {output}")
    return 0 if report["ok"] else 1


def _options(values):
    result = {}
    index = 0
    while index < len(values):
        value = values[index]
        if value.startswith("--") and index + 1 < len(values) and not values[index + 1].startswith("--"):
            result[value] = values[index + 1]
            index += 2
        else:
            index += 1
    return result


def _load_run_suite():
    path = ROOT / "scripts" / "run_suite.py"
    spec = importlib.util.spec_from_file_location("dvcl_run_suite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
