"""Dataset preparation compatible with the HSeCo experiment data flow."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from .artifacts import CleanGraphArtifact


PREDICT_NTYPE = {
    "acm": "paper",
    "dblp": "author",
    "aminer": "paper",
    "imdb": "movie",
}

META_PATHS = {
    "acm": [["pa", "ap"], ["pf", "fp"]],
    "dblp": [["ap", "pa"], ["ap", "pc", "cp", "pa"], ["ap", "pt", "tp", "pa"]],
    "aminer": [["pa", "ap"], ["pr", "rp"]],
    "imdb": [["md", "dm"], ["ma", "am"]],
}

CANONICAL_ETYPES = {
    "acm": [
        ("paper", "pa", "author"),
        ("author", "ap", "paper"),
        ("paper", "pf", "field"),
        ("field", "fp", "paper"),
    ],
    "dblp": [
        ("author", "ap", "paper"),
        ("paper", "pa", "author"),
        ("paper", "pc", "conf"),
        ("conf", "cp", "paper"),
        ("paper", "pt", "term"),
        ("term", "tp", "paper"),
    ],
}


def build_clean_artifact(
    dataset: str,
    data_root: Path = Path("data"),
    version: str = "v1",
) -> CleanGraphArtifact:
    dataset = dataset.lower()
    loaders = {"acm": _load_acm, "dblp": _load_dblp}
    try:
        node_counts, adjs, features, labels, num_classes = loaders[dataset](Path(data_root))
    except KeyError as exc:
        raise ValueError("Native dataset preparation currently supports ACM and DBLP") from exc
    adjs = {name: _binary_csr(value) for name, value in adjs.items()}
    stats = graph_stats(node_counts, adjs, features, labels)
    return CleanGraphArtifact(
        dataset=dataset,
        version=f"{dataset}-{version}",
        predict_ntype=PREDICT_NTYPE[dataset],
        node_counts=node_counts,
        hete_adjs=adjs,
        features=features.detach().cpu().float(),
        labels=labels.detach().cpu().long(),
        num_classes=int(num_classes),
        meta_paths=META_PATHS[dataset],
        canonical_etypes=CANONICAL_ETYPES[dataset],
        stats=stats,
    )


def graph_stats(
    node_counts: Dict[str, int],
    adjs: Dict[str, sp.csr_matrix],
    features: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, object]:
    return {
        "node_counts": {name: int(value) for name, value in node_counts.items()},
        "edge_counts": {name: int(value.nnz) for name, value in adjs.items()},
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "num_labels": int(labels.numel()),
        "label_min": int(labels.min().item()),
        "label_max": int(labels.max().item()),
    }


def _load_acm(data_root: Path):
    try:
        from dgl.data.utils import _get_dgl_url, download, get_download_dir
        from scipy import io as sio
    except ImportError as exc:
        raise RuntimeError("Preparing ACM requires DGL and SciPy") from exc

    data_path = Path(get_download_dir()) / "ACM.mat"
    if not data_path.exists():
        download(_get_dgl_url("dataset/ACM.mat"), path=str(data_path))
    data = sio.loadmat(str(data_path))
    p_vs_f = data["PvsL"]
    p_vs_a = data["PvsA"]
    p_vs_t = data["PvsT"]
    p_vs_c = data["PvsC"]
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]
    selected = (p_vs_c[:, conf_ids].sum(1) != 0).A1.nonzero()[0]
    p_vs_f = p_vs_f[selected]
    p_vs_a = p_vs_a[selected]
    p_vs_t = p_vs_t[selected]
    p_vs_c = p_vs_c[selected]
    node_counts = {
        "paper": int(p_vs_f.shape[0]),
        "field": int(p_vs_f.shape[1]),
        "author": int(p_vs_a.shape[1]),
    }
    adjs = {"pa": p_vs_a, "ap": p_vs_a.T, "pf": p_vs_f, "fp": p_vs_f.T}
    features = torch.from_numpy(p_vs_t.toarray()).float()
    labels = np.zeros(len(selected), dtype=np.int64)
    pc_p, pc_c = p_vs_c.nonzero()
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    return node_counts, adjs, features, torch.from_numpy(labels), 3


def _load_dblp(data_root: Path):
    try:
        from torch_geometric.datasets import DBLP
    except ImportError as exc:
        raise RuntimeError("Preparing DBLP requires torch-geometric") from exc

    data = DBLP(str(data_root / "raw" / "dblp"))[0]
    features = data["author"].x.detach().cpu().float()
    labels = data["author"].y.detach().cpu().long()
    node_counts = {
        "author": int(data["author"].num_nodes),
        "paper": int(data["paper"].num_nodes),
        "term": int(data["term"].num_nodes),
        "conf": int(data["conference"].num_nodes),
    }
    ap = _edge_index_to_csr(
        data["author", "to", "paper"].edge_index,
        (node_counts["author"], node_counts["paper"]),
    )
    pt = _edge_index_to_csr(
        data["paper", "to", "term"].edge_index,
        (node_counts["paper"], node_counts["term"]),
    )
    pc = _edge_index_to_csr(
        data["paper", "to", "conference"].edge_index,
        (node_counts["paper"], node_counts["conf"]),
    )
    adjs = {"ap": ap, "pa": ap.T, "pc": pc, "cp": pc.T, "pt": pt, "tp": pt.T}
    return node_counts, adjs, features, labels, int(labels.max().item()) + 1


def _edge_index_to_csr(edge_index: torch.Tensor, shape: Tuple[int, int]) -> sp.csr_matrix:
    values = edge_index.detach().cpu().numpy()
    return sp.csr_matrix(
        (np.ones(values.shape[1], dtype=np.int8), (values[0], values[1])),
        shape=shape,
    )


def _binary_csr(value: sp.spmatrix) -> sp.csr_matrix:
    result = value.tocsr().astype(bool).astype(np.int8)
    result.eliminate_zeros()
    return result
