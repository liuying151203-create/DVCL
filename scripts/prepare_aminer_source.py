import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import scipy.sparse as sp


PAPER_COUNT = 6564
AUTHOR_COUNT = 13329
RESEARCH_COUNT = 35890
SOURCE_FILES = ("pa.txt", "pr.txt", "labels.npy", "pos.npz")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the official HeCo AMiner release into DVCL raw files."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", default="data/aminer")
    parser.add_argument("--source-url", default="https://github.com/liun-online/HeCo.git")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def convert_aminer_source(
    source_root: Path,
    output_root: Path,
    source_url: str,
    source_commit: str,
    force: bool = False,
):
    source_root = Path(source_root)
    output_root = Path(output_root)
    missing = [str(source_root / name) for name in SOURCE_FILES if not (source_root / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing HeCo AMiner files: " + ", ".join(missing))
    outputs = [output_root / name for name in ("pa.npz", "pr.npz", "labels.npy", "pos.npz")]
    if not force and any(path.exists() for path in outputs):
        raise FileExistsError("AMiner output exists; pass --force to replace it")

    pa = _relation(source_root / "pa.txt", (PAPER_COUNT, AUTHOR_COUNT))
    pr = _relation(source_root / "pr.txt", (PAPER_COUNT, RESEARCH_COUNT))
    labels = np.load(source_root / "labels.npy", allow_pickle=False)
    positions = sp.load_npz(source_root / "pos.npz").tocsr()
    if labels.shape != (PAPER_COUNT,):
        raise ValueError(f"Expected {PAPER_COUNT} AMiner labels, got {labels.shape}")
    if set(np.unique(labels).tolist()) != {0, 1, 2, 3}:
        raise ValueError("AMiner labels must contain classes 0, 1, 2 and 3")
    if positions.shape != (PAPER_COUNT, PAPER_COUNT):
        raise ValueError(f"Unexpected AMiner position matrix shape: {positions.shape}")

    output_root.mkdir(parents=True, exist_ok=True)
    sp.save_npz(output_root / "pa.npz", pa, compressed=True)
    sp.save_npz(output_root / "pr.npz", pr, compressed=True)
    shutil.copyfile(source_root / "labels.npy", output_root / "labels.npy")
    shutil.copyfile(source_root / "pos.npz", output_root / "pos.npz")
    manifest = {
        "schema_version": 1,
        "dataset": "aminer",
        "source": {"url": source_url, "commit": source_commit},
        "source_files": {name: _sha256(source_root / name) for name in SOURCE_FILES},
        "outputs": {path.name: _sha256(path) for path in outputs},
        "stats": {
            "paper_nodes": PAPER_COUNT,
            "author_nodes": AUTHOR_COUNT,
            "research_nodes": RESEARCH_COUNT,
            "pa_edges": int(pa.nnz),
            "pr_edges": int(pr.nnz),
            "position_nonzeros": int(positions.nnz),
            "label_counts": {
                str(label): int((labels == label).sum()) for label in range(4)
            },
        },
    }
    (output_root / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _relation(path: Path, shape):
    edges = np.loadtxt(path, dtype=np.int64, ndmin=2)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Expected a two-column edge list: {path}")
    if edges.min() < 0 or edges[:, 0].max() >= shape[0] or edges[:, 1].max() >= shape[1]:
        raise ValueError(f"AMiner edge index is outside declared shape: {path}")
    matrix = sp.csr_matrix(
        (np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])), shape=shape
    )
    matrix.data[:] = 1
    matrix.eliminate_zeros()
    return matrix


def _sha256(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    manifest = convert_aminer_source(
        Path(args.source_root),
        Path(args.output_root),
        args.source_url,
        args.source_commit,
        args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
