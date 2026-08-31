import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit exclusive NVIDIA GPU availability for efficiency runs."
    )
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--output",
        default=str(
            ROOT / "outputs/audits/model_efficiency_v1-hardware.json"
        ),
    )
    return parser.parse_args()


def parse_device_index(device):
    prefix, separator, index = str(device).partition(":")
    if prefix != "cuda" or not separator or not index.isdigit():
        raise ValueError(f"Expected CUDA device such as cuda:4, got {device!r}")
    return int(index)


def parse_csv_rows(value):
    return [
        [column.strip() for column in line.split(",")]
        for line in value.splitlines() if line.strip()
    ]


def collect_hardware_audit(device, runner=subprocess.run):
    index = parse_device_index(device)
    gpu_result = runner(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if gpu_result.returncode:
        raise RuntimeError(gpu_result.stderr.strip() or "nvidia-smi failed")
    gpus = []
    for row in parse_csv_rows(gpu_result.stdout):
        if len(row) != 5:
            raise ValueError(f"Unexpected GPU row: {row}")
        gpus.append({
            "index": int(row[0]),
            "uuid": row[1],
            "name": row[2],
            "driver_version": row[3],
            "memory_total_mib": int(row[4]),
        })
    target = next((gpu for gpu in gpus if gpu["index"] == index), None)
    if target is None:
        raise ValueError(f"CUDA device index {index} is unavailable")
    process_result = runner(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if process_result.returncode:
        raise RuntimeError(
            process_result.stderr.strip() or "nvidia-smi process query failed"
        )
    processes = []
    for row in parse_csv_rows(process_result.stdout):
        if len(row) != 4 or row[0] != target["uuid"]:
            continue
        processes.append({
            "gpu_uuid": row[0],
            "pid": int(row[1]),
            "process_name": row[2],
            "used_memory_mib": int(row[3]),
        })
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "gpu": target,
        "existing_compute_processes": processes,
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "ok": not processes and commit.returncode == 0 and dirty.returncode == 0
        and not dirty.stdout.strip(),
    }


def main():
    args = parse_args()
    report = collect_hardware_audit(args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"device={report['device']} gpu={report['gpu']['name']} "
        f"processes={len(report['existing_compute_processes'])} "
        f"git_dirty={report['git_dirty']} ok={report['ok']}"
    )
    print(f"Wrote {output}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
