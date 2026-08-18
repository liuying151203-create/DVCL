from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from .semantic import SemanticAttention


class RoHeConv(nn.Module):
    confidence_chunk_size = 65536

    def __init__(self, input_dim, output_dim, heads, dropout, top_t):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim * heads, bias=False)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.dropout = dropout
        self.top_t = top_t
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(
            self.projection.weight, gain=nn.init.calculate_gain("relu")
        )

    def forward(self, features, transition: sp.csr_matrix):
        import dgl
        from dgl import function as fn
        from dgl.nn.pytorch.softmax import edge_softmax

        dropped = F.dropout(features, self.dropout, self.training)
        projected = self.projection(dropped).view(-1, self.heads, self.output_dim)
        coo = transition.tocoo()
        rows = torch.as_tensor(coo.row, dtype=torch.long, device=features.device)
        columns = torch.as_tensor(coo.col, dtype=torch.long, device=features.device)
        prior = torch.as_tensor(coo.data, dtype=features.dtype, device=features.device)
        confidence = self._confidence(projected, rows, columns, prior)
        retained = self._top_t_mask(transition.indptr, rows, confidence)
        retained_rows = rows[retained]
        retained_columns = columns[retained]
        edge_scores = (
            projected[retained_rows] * projected[retained_columns]
        ).sum(-1)
        graph = dgl.graph(
            (retained_rows, retained_columns),
            num_nodes=features.shape[0],
            device=features.device,
        )
        graph.srcdata["projected"] = projected
        logits = edge_scores.unsqueeze(-1)
        graph.edata["attention"] = edge_softmax(graph, logits)
        graph.update_all(
            fn.u_mul_e("projected", "attention", "message"),
            fn.sum("message", "output"),
        )
        return F.elu(graph.dstdata["output"])

    def _confidence(self, projected, rows, columns, prior):
        confidence = torch.empty(
            rows.numel(), dtype=projected.dtype, device=projected.device
        )
        with torch.no_grad():
            for start in range(0, rows.numel(), self.confidence_chunk_size):
                end = min(start + self.confidence_chunk_size, rows.numel())
                edge_scores = (
                    projected[rows[start:end]] * projected[columns[start:end]]
                ).sum(-1)
                confidence[start:end] = edge_scores.sum(-1) * prior[start:end]
        return confidence

    def _top_t_mask(self, indptr, source_rows, confidence):
        retained = torch.zeros(confidence.shape[0], dtype=torch.bool, device=confidence.device)
        counts = np.diff(indptr)
        max_degree = int(counts.max(initial=0))
        count = min(self.top_t, max_degree)
        if not count:
            return retained
        starts = torch.as_tensor(
            indptr[:-1], dtype=torch.long, device=confidence.device
        )
        positions = torch.arange(
            confidence.numel(), device=confidence.device
        ) - starts[source_rows]
        padded = torch.full(
            (len(counts), max_degree),
            -torch.inf,
            dtype=confidence.dtype,
            device=confidence.device,
        )
        padded[source_rows, positions] = confidence
        values, selected_positions = torch.topk(padded, count, dim=1)
        selected = starts.unsqueeze(1) + selected_positions
        valid = values >= 0
        retained[selected[valid]] = True
        return retained


class RoHe(nn.Module):
    def __init__(
        self,
        num_paths: int,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        heads: int,
        dropout: float,
        top_t: Sequence[int],
    ) -> None:
        super().__init__()
        if len(top_t) != num_paths:
            raise ValueError("RoHe top_t must contain one value per meta-path")
        self.convs = nn.ModuleList([
            RoHeConv(input_dim, hidden_dim, heads, dropout, int(value))
            for value in top_t
        ])
        self.semantic_attention = SemanticAttention(hidden_dim * heads)
        self.classifier = nn.Linear(hidden_dim * heads, num_classes)

    def forward(self, features, transitions: Sequence[sp.csr_matrix]):
        values = [
            conv(features, transition).flatten(1)
            for conv, transition in zip(self.convs, transitions)
        ]
        embedding = self.semantic_attention(torch.stack(values, dim=1))
        return self.classifier(embedding)

    def semantic_weights(self):
        return self.semantic_attention.last_attention.flatten()
