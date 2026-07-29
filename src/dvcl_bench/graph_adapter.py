"""Conversions between artifact sparse matrices and model graph formats."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import scipy.sparse as sp
import torch

CanonicalEType = Tuple[str, str, str]


def hete_adjs_to_dgl(
    hete_adjs: Mapping[str, sp.spmatrix],
    canonical_etypes: Iterable[CanonicalEType],
    node_counts: Mapping[str, int],
):
    try:
        import dgl
    except ImportError as exc:
        raise RuntimeError("DGL is required by the HSeCo and DVCL native adapters") from exc
    graph_data = {}
    for canonical in canonical_etypes:
        short_name = canonical[1]
        if short_name not in hete_adjs:
            raise KeyError(f"Missing adjacency for edge type: {short_name}")
        graph_data[canonical] = hete_adjs[short_name].nonzero()
    return dgl.heterograph(graph_data, num_nodes_dict=dict(node_counts))


def pyg_attack_to_adjs(data) -> Dict[str, sp.csr_matrix]:
    node_counts = {}
    for ntype in data.node_types:
        store = data[ntype]
        count = getattr(store, "num_nodes", None)
        if count is None and getattr(store, "x", None) is not None:
            count = store.x.shape[0]
        if count is None:
            raise ValueError(f"Cannot determine node count for {ntype}")
        node_counts[ntype] = int(count)

    result = {}
    for etype in data.edge_types:
        edge_index = data[etype].edge_index.detach().cpu().numpy()
        edge_weight = data[etype].get(
            "edge_weight", torch.ones(edge_index.shape[1])
        ).detach().cpu().numpy()
        shape = (node_counts[etype[0]], node_counts[etype[-1]])
        name = etype[0][0] + etype[-1][0]
        result[name] = sp.csr_matrix(
            (edge_weight, (edge_index[0], edge_index[1])), shape=shape
        )
    return result


def meta_path_adjacency(
    adjs: Mapping[str, sp.spmatrix],
    meta_path: Iterable[str],
) -> sp.csr_matrix:
    names = list(meta_path)
    if not names:
        raise ValueError("Meta-path cannot be empty")
    value = adjs[names[0]].tocsr()
    for name in names[1:]:
        value = value @ adjs[name].tocsr()
    value = value.astype(bool).astype(np.int8).tocsr()
    value.eliminate_zeros()
    return value


def sparse_to_edge_index(adj: sp.spmatrix, device=None) -> torch.Tensor:
    rows, cols = adj.nonzero()
    return torch.as_tensor(np.stack((rows, cols)), dtype=torch.long, device=device)
