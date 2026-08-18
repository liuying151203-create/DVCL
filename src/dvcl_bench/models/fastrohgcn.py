from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class FastRoHGCN(nn.Module):
    def __init__(
        self,
        node_counts: Mapping[str, int],
        predict_ntype: str,
        target_input_dim: int,
        projection_dim: int,
        hidden_dim: int,
        layers: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        from dgl.nn.pytorch import GraphConv

        self.node_types = list(node_counts)
        self.node_counts = dict(node_counts)
        self.predict_ntype = predict_ntype
        self.target_projection = nn.Linear(target_input_dim, projection_dim)
        self.type_embeddings = nn.ModuleDict({
            node_type: nn.Embedding(count, projection_dim)
            for node_type, count in node_counts.items()
            if node_type != predict_ntype
        })
        dimensions = [projection_dim] + [hidden_dim] * layers + [num_classes]
        self.layers = nn.ModuleList([
            GraphConv(
                dimensions[index], dimensions[index + 1],
                norm="none", activation=F.elu if index + 2 < len(dimensions) else None,
                allow_zero_in_degree=True,
            )
            for index in range(len(dimensions) - 1)
        ])
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.target_projection.weight, gain=1.414)
        for embedding in self.type_embeddings.values():
            nn.init.xavier_normal_(embedding.weight, gain=1.414)

    def forward(self, target_features, graph, edge_weight):
        values = []
        for node_type in self.node_types:
            if node_type == self.predict_ntype:
                values.append(self.target_projection(target_features))
            else:
                indices = torch.arange(
                    self.node_counts[node_type], device=target_features.device
                )
                values.append(self.type_embeddings[node_type](indices))
        hidden = self.dropout(torch.cat(values, dim=0))
        for layer in self.layers:
            hidden = self.dropout(hidden)
            hidden = layer(graph, hidden, edge_weight=edge_weight)
        return hidden
