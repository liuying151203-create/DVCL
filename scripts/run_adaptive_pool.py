import argparse
import importlib.util
import json
import sys
import threading
from pathlib import Path
from queue import Empty, Queue


ROOT = Path(__file__).resolve().parents[1]
MODEL_PRIORITY = (
    "dvcl", "hseco", "simplehgn", "heco", "magnn", "hgt",
    "fastrohgcn", "heteroguard", "rohe", "heterosage", "han",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an audited adaptive suite through a persistent multi-GPU pool."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", action="append", dest="devices", required=True)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--attack-variant", action="append", dest="attack_variants")
    parser.add_argument("--rate", action="append", type=float, dest="rates")
    parser.add_argument("--split-seed", action="append", type=int, dest="split_seeds")
    parser.add_argument("--attack-seed", action="append", type=int, dest="attack_seeds")
    parser.add_argument("--train-seed", action="append", type=int, dest="train_seeds")
    parser.add_argument(
        "--confirmation-audit",
        default="outputs/analysis/adaptive_attack_strength_confirmation_v1/audit.json",
    )
    parser.add_argument(
        "--input-audit",
        default="outputs/audits/adaptive_target_evasion_v1-inputs.json",
    )
    parser.add_argument("--oom-retries", type=int, default=12)
    parser.add_argument("--oom-retry-delay", type=float, default=300.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_run_suite():
    path = ROOT / "scripts" / "run_suite.py"
    spec = importlib.util.spec_from_file_location("adaptive_pool_run_suite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def options(command):
    return {
        command[index]: command[index + 1]
        for index in range(len(command) - 1)
        if command[index].startswith("--")
    }


def replace_device(command, device):
    value = list(command)
    index = value.index("--device")
    value[index + 1] = device
    return value


def prioritize_commands(commands):
    priorities = {name: index for index, name in enumerate(MODEL_PRIORITY)}

    def key(command):
        value = options(command)
        return (
            priorities.get(value.get("--model"), len(priorities)),
            int(value.get("--train-seed", 0)),
            int(value.get("--attack-seed", 0)),
            value.get("--dataset", ""),
        )

    return sorted(commands, key=key)


def validate_audits(confirmation_path: Path, input_path: Path):
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    if confirmation.get("ok") is not True:
        raise ValueError(f"Confirmation audit is not complete: {confirmation_path}")
    inputs = json.loads(input_path.read_text(encoding="utf-8"))
    summary = inputs.get("summary", {})
    if summary.get("failed") != 0 or summary.get("passed") != summary.get("total"):
        raise ValueError(f"Formal input audit is not complete: {input_path}")


def run_pool(
    commands, devices, run_suite, force=False, continue_on_error=False,
    oom_retries=12, oom_retry_delay=300.0,
):
    queue = Queue()
    for command in commands:
        queue.put(command)
    failures = []
    lock = threading.Lock()
    stop = threading.Event()

    def worker(device):
        while not stop.is_set():
            try:
                command = queue.get_nowait()
            except Empty:
                return
            value = replace_device(command, device)
            if force:
                value.append("--force")
            print(f"[{device}] {' '.join(value)}", flush=True)
            result = run_suite.run_with_oom_retries(
                value,
                cwd=str(ROOT),
                retries=oom_retries,
                delay=oom_retry_delay,
            )
            if result.returncode:
                with lock:
                    failures.append({
                        "device": device,
                        "returncode": result.returncode,
                        "command": value,
                    })
                if not continue_on_error:
                    stop.set()
            queue.task_done()

    workers = [
        threading.Thread(target=worker, args=(device,), name=device)
        for device in devices
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    return failures, queue.qsize()


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def select_config(config, args, run_suite):
    config = run_suite.select_models(config, args.models)
    config = run_suite.select_variants(config, args.variants)
    config = run_suite.select_datasets(config, args.datasets)
    config = run_suite.select_attacks(config, args.attack_variants, args.rates)
    return run_suite.select_seeds(
        config, args.split_seeds, args.attack_seeds, args.train_seeds
    )


def main() -> int:
    args = parse_args()
    validate_audits(
        resolve(args.confirmation_audit),
        resolve(args.input_audit),
    )
    run_suite = load_run_suite()
    config_path = resolve(args.config)
    config = run_suite.load_config(config_path)
    config = select_config(config, args, run_suite)
    commands = prioritize_commands(
        list(run_suite.commands(config, sys.executable, ROOT))
    )
    print(f"physical_runs={len(commands)} devices={args.devices}")
    if args.dry_run:
        for index, command in enumerate(commands):
            device = args.devices[index % len(args.devices)]
            value = replace_device(command, device)
            print(f"[{device}] {' '.join(value)}")
        return 0
    failures, remaining = run_pool(
        commands,
        args.devices,
        run_suite,
        force=args.force,
        continue_on_error=args.continue_on_error,
        oom_retries=args.oom_retries,
        oom_retry_delay=args.oom_retry_delay,
    )
    print(f"failures={len(failures)} unstarted={remaining}")
    return 1 if failures or remaining else 0


if __name__ == "__main__":
    raise SystemExit(main())
