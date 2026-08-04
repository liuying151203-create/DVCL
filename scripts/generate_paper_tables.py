import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dvcl_bench.reporting import audit_manifests, load_run_rows, render_paper_tables


def parse_args():
    parser = argparse.ArgumentParser(description="Generate audited Markdown paper tables.")
    parser.add_argument(
        "--runs",
        default=str(ROOT / "outputs" / "summaries" / "dvcl_main" / "runs.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "docs" / "paper-experiment-tables.md"),
    )
    parser.add_argument(
        "--baseline-runs",
        default=str(ROOT / "outputs" / "summaries" / "baseline_main" / "runs.csv"),
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_rows = load_run_rows(Path(args.baseline_runs))
    baseline_audit = audit_manifests(baseline_rows)
    rendered = render_paper_tables(
        load_run_rows(Path(args.runs)), baseline_rows, baseline_audit
    )
    output = Path(args.output)
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            print(f"Paper tables are stale: {output}", file=sys.stderr)
            return 1
        print(f"Paper tables are current: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Generated paper tables: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
