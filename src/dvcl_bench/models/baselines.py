"""Native heterogeneous graph baselines for the unified artifact protocol."""

from __future__ import annotations

from typing import Iterable, Mapping, Tuple

from torch import Tensor, nn
from torch.nn import functional as F


class HeteroSAGE(nn.Module):
    def __init__(
        self,
        canonical_etypes: Iterable[Tuple[str, str, str]],
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("HeteroSAGE requires at least two layers")
        from dgl.nn.pytorch import HeteroGraphConv, SAGEConv

        relations = sorted({canonical[1] for canonical in canonical_etypes})
        dimensions = [input_dim] + [hidden_dim] * (num_layers - 1) + [num_classes]
        self.layers = nn.ModuleList([
            HeteroGraphConv({
                relation: SAGEConv(
                    (dimensions[index], dimensions[index]),
                    dimensions[index + 1],
                    "mean",
                )
                for relation in relations
            }, aggregate="sum")
            for index in range(num_layers)
        ])
        self.dropout = dropout

    def forward(self, graph, features: Mapping[str, Tensor]):
        values = dict(features)
        for index, layer in enumerate(self.layers):
            values = layer(graph, values)
            if index + 1 < len(self.layers):
                values = {
                    node_type: F.dropout(F.relu(value), self.dropout, self.training)
                    for node_type, value in values.items()
                }
        return values
