from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from .semantic import SemanticAttention


class RoHeConv(nn.Module):
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
        edge_scores = (projected[rows] * projected[columns]).sum(-1)
        confidence = edge_scores.sum(-1) * prior
        retained = self._top_t_mask(coo.row, confidence)
        graph = dgl.graph(
            (rows[retained], columns[retained]),
            num_nodes=features.shape[0],
            device=features.device,
        )
        graph.srcdata["projected"] = projected
        logits = edge_scores[retained].unsqueeze(-1)
        graph.edata["attention"] = edge_softmax(graph, logits)
        graph.update_all(
            fn.u_mul_e("projected", "attention", "message"),
            fn.sum("message", "output"),
        )
        return F.elu(graph.dstdata["output"])

    def _top_t_mask(self, source_rows, confidence):
        retained = torch.zeros(confidence.shape[0], dtype=torch.bool, device=confidence.device)
        order = np.argsort(source_rows, kind="stable")
        sorted_rows = source_rows[order]
        boundaries = np.flatnonzero(np.r_[True, sorted_rows[1:] != sorted_rows[:-1], True])
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            indices = torch.as_tensor(order[start:end], dtype=torch.long, device=confidence.device)
            count = min(self.top_t, indices.numel())
            if count:
                chosen = indices[torch.topk(confidence[indices], count).indices]
                chosen = chosen[confidence[chosen] >= 0]
                retained[chosen] = True
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
