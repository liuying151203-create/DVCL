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
    parser.add_argument(
        "--corrected-dblp-runs",
        default=str(
            ROOT / "outputs" / "summaries" / "dblp_poisoning_main_v1" / "runs.csv"
        ),
    )
    parser.add_argument(
        "--corrected-acm-runs",
        default=str(
            ROOT / "outputs" / "summaries" / "acm_poisoning_main_v1" / "runs.csv"
        ),
    )
    parser.add_argument(
        "--corrected-acm-ablation-runs",
        default=str(
            ROOT / "outputs" / "summaries" / "acm_poisoning_ablation_v1" / "runs.csv"
        ),
    )
    parser.add_argument(
        "--robust-runs",
        default=str(
            ROOT / "outputs" / "summaries" / "robust_baselines_poisoning_v1" / "runs.csv"
        ),
    )
    parser.add_argument(
        "--openhgnn-runs",
        default=str(
            ROOT / "outputs" / "summaries" / "openhgnn_baselines_poisoning_v1" / "runs.csv"
        ),
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corrected_dblp = load_run_rows(Path(args.corrected_dblp_runs))
    corrected_acm = load_run_rows(Path(args.corrected_acm_runs))
    corrected_ablation = load_run_rows(Path(args.corrected_acm_ablation_runs))
    rows = [
        row for row in load_run_rows(Path(args.runs))
        if not (
            row["dataset"] in {"acm", "dblp"}
            and row["model"] in {"hseco", "dvcl"}
            and row["variant"] == "default"
        )
        and not (
            row["dataset"] == "acm"
            and row["model"] == "dvcl"
            and row["variant"] in {"full", "no_cl", "topology_only", "feature_only"}
        )
    ]
    rows.extend(
        row for row in (*corrected_acm, *corrected_dblp)
        if row["model"] in {"hseco", "dvcl"} and row["variant"] == "default"
    )
    rows.extend(corrected_ablation)
    baseline_rows = [
        row for row in load_run_rows(Path(args.baseline_runs))
        if not (
            row["dataset"] in {"acm", "dblp"}
            and row["model"] in {"han", "heterosage"}
        )
    ]
    baseline_rows.extend(
        row for row in (*corrected_acm, *corrected_dblp)
        if row["model"] in {"han", "heterosage"}
    )
    baseline_audit = audit_manifests(baseline_rows)
    extended_rows = [
        *load_run_rows(Path(args.robust_runs)),
        *load_run_rows(Path(args.openhgnn_runs)),
    ]
    rendered = render_paper_tables(
        rows,
        baseline_rows,
        baseline_audit,
        extended_rows=extended_rows,
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
