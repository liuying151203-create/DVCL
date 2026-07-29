"""HSeCo-equivalent semantic purification components."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SemanticAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.last_attention = None

    def forward(self, values: Tensor) -> Tensor:
        attention = torch.softmax(self.project(values).mean(0), dim=0)
        self.last_attention = attention
        return (attention.expand((values.shape[0],) + attention.shape) * values).sum(1)


class SemanticHANLayer(nn.Module):
    def __init__(self, num_paths, input_dim, hidden_dim, heads, dropout) -> None:
        super().__init__()
        from dgl.nn.pytorch import GATConv

        self.gat_layers = nn.ModuleList([
            GATConv(input_dim, hidden_dim, heads, dropout, dropout, activation=F.elu)
            for _ in range(num_paths)
        ])
        self.semantic_attention = SemanticAttention(hidden_dim * heads)

    def forward(self, features: Tensor, graphs: Sequence) -> Tensor:
        import dgl

        values = [
            layer(dgl.add_self_loop(graph), features).flatten(1)
            for layer, graph in zip(self.gat_layers, graphs)
        ]
        return self.semantic_attention(torch.stack(values, dim=1))


class SemanticHAN(nn.Module):
    def __init__(self, num_paths, input_dim, hidden_dim, num_classes, heads, dropout) -> None:
        super().__init__()
        self.layer = SemanticHANLayer(num_paths, input_dim, hidden_dim, heads, dropout)
        self.predict = nn.Linear(hidden_dim * heads, num_classes)

    def encode(self, features: Tensor, graphs: Sequence) -> Tensor:
        return self.layer(features, graphs)

    def forward(self, features: Tensor, graphs: Sequence) -> Tensor:
        return self.predict(self.encode(features, graphs))

    def semantic_weights(self) -> Tensor:
        return self.layer.semantic_attention.last_attention.flatten()


class NodeLevelAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, heads, dropout) -> None:
        super().__init__()
        from dgl.nn.pytorch import GATConv

        self.gat = GATConv(
            input_dim, hidden_dim, heads,
            feat_drop=dropout, attn_drop=dropout, activation=F.elu,
        )
        self.classifier = nn.Linear(hidden_dim * heads, num_classes)

    def forward(self, features: Tensor, graph) -> Tensor:
        import dgl

        return self.classifier(self.gat(dgl.add_self_loop(graph), features).flatten(1))

    def contrastive_loss(self, features, views, positive, negative, mask, temperature=0.5):
        values = []
        for graph in views:
            anchor = self(features, graph)[mask]
            pos = self(features, positive)[mask]
            neg = self(features, negative)[mask]
            pos_sim = F.cosine_similarity(anchor, pos, dim=-1)
            neg_sim = F.cosine_similarity(anchor, neg, dim=-1)
            numerator = torch.exp(pos_sim / temperature)
            values.append((-torch.log(numerator / (numerator + torch.exp(neg_sim / temperature)))).mean())
        return torch.stack(values).mean()


def transition_edges(
    features: Tensor,
    adjs: Mapping[str, sp.spmatrix],
    meta_paths: Iterable[Iterable[str]],
):
    normalized = {}
    for name, value in adjs.items():
        value = value.tocsr()
        degree = np.asarray(value.sum(axis=1)).reshape(-1)
        degree = np.where(degree > 0, degree, 1)
        normalized[name] = sp.diags(1.0 / degree).dot(value)
    similarity = torch.mm(features.detach().cpu(), features.detach().cpu().t())
    result = []
    for path in meta_paths:
        names = list(path)
        adjacency = normalized[names[0]]
        for name in names[1:]:
            adjacency = adjacency.dot(normalized[name])
        coo = adjacency.tocoo()
        edges = torch.as_tensor(np.stack((coo.row, coo.col)), dtype=torch.long)
        weights = torch.as_tensor(coo.data, dtype=features.dtype)
        weights = weights * similarity[edges[0], edges[1]]
        degree = torch.zeros(features.shape[0], dtype=weights.dtype)
        degree.scatter_add_(0, edges[0], weights)
        inverse = degree.pow(-1)
        inverse[torch.isinf(inverse)] = 1.0
        weights = inverse[edges[0]] * weights * inverse[edges[1]]
        result.append((edges.to(features.device), weights.to(features.device), features.shape[0]))
    return result


def purified_graphs(transitions, thresholds):
    import dgl

    graphs = []
    for (edges, weights, num_nodes), threshold in zip(transitions, thresholds):
        mask = weights >= threshold
        matrix = _edge_matrix(edges[:, mask], weights[mask], num_nodes)
        graphs.append(dgl.from_scipy(matrix, eweight_name="weight").to(edges.device))
    return graphs


def semantic_graph(transitions, attention, threshold, apply_filter=True):
    indices = torch.cat([item[0] for item in transitions], dim=1)
    values = torch.cat([
        item[1] * attention[index] for index, item in enumerate(transitions)
    ])
    num_nodes = transitions[0][2]
    sparse = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()
    indices, values = sparse.indices(), sparse.values()
    if apply_filter:
        mask = values >= threshold
        indices, values = indices[:, mask], values[mask]
    return _edge_matrix(indices[[1, 0]], values, num_nodes)


def perturb_matrix(matrix: sp.spmatrix, noise_rate: float, rng: np.random.RandomState):
    matrix = matrix.tocsr()
    rows, cols = matrix.nonzero()
    count = int(noise_rate * len(matrix.data))
    if not count:
        return matrix
    selected = rng.choice(len(matrix.data), count, replace=False)
    new_cols = cols.copy()
    new_cols[selected] = rng.permutation(cols[selected])
    return sp.csr_matrix((matrix.data.copy(), (rows, new_cols)), shape=matrix.shape)


def _edge_matrix(edges: Tensor, weights: Tensor, num_nodes: int) -> sp.csr_matrix:
    values = weights.detach().cpu().numpy()
    coordinates = edges.detach().cpu().numpy()
    return sp.csr_matrix((values, (coordinates[0], coordinates[1])), shape=(num_nodes, num_nodes))
