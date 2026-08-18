"""Native heterogeneous graph baselines for the unified artifact protocol."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import torch
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


class WeightedRelationSAGE(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.neighbor = nn.Linear(input_dim, output_dim)
        self.root = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, source, target, edge_index, edge_weight):
        rows, columns = edge_index
        messages = self.neighbor(source)[rows] * edge_weight.unsqueeze(1)
        aggregated = target.new_zeros((target.shape[0], messages.shape[1]))
        aggregated.index_add_(0, columns, messages)
        degree = target.new_zeros(target.shape[0])
        degree.index_add_(0, columns, edge_weight)
        aggregated = aggregated / degree.clamp_min(1).unsqueeze(1)
        return aggregated + self.root(target)


class HeteroGuard(nn.Module):
    def __init__(
        self,
        canonical_etypes: Iterable[Tuple[str, str, str]],
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int,
        dropout: float,
        attention_threshold: float,
        gated_attention: bool,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("HeteroGuard requires at least two layers")
        self.canonical_etypes = list(canonical_etypes)
        dimensions = [input_dim] + [hidden_dim] * (num_layers - 1) + [num_classes]
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                self._key(canonical): WeightedRelationSAGE(
                    dimensions[index], dimensions[index + 1]
                )
                for canonical in self.canonical_etypes
            })
            for index in range(num_layers)
        ])
        self.gates = nn.ParameterList([
            nn.Parameter(torch.rand(())) for _ in range(num_layers)
        ])
        self.dropout = dropout
        self.attention_threshold = attention_threshold
        self.gated_attention = gated_attention
        self.last_attention_density: Dict[str, float] = {}

    def forward(self, features, edge_indices):
        values = dict(features)
        previous = None
        for layer_index, layer in enumerate(self.layers):
            attention = self._attention(values, edge_indices)
            if self.gated_attention and previous is not None:
                gate = torch.sigmoid(self.gates[layer_index])
                attention = {
                    canonical: gate * previous[canonical] + (1 - gate) * weight
                    for canonical, weight in attention.items()
                }
            outputs = {}
            for canonical in self.canonical_etypes:
                source_type, _, target_type = canonical
                output = layer[self._key(canonical)](
                    values[source_type], values[target_type],
                    edge_indices[canonical], attention[canonical],
                )
                outputs[target_type] = outputs.get(target_type, 0) + output
            if layer_index + 1 < len(self.layers):
                outputs = {
                    node_type: F.dropout(
                        F.relu(value), self.dropout, self.training
                    )
                    for node_type, value in outputs.items()
                }
            values = outputs
            previous = attention
        self.last_attention_density = {
            self._key(canonical): float((weight > 0).float().mean().detach())
            for canonical, weight in previous.items()
        }
        return values

    def _attention(self, features, edge_indices):
        result = {}
        for canonical in self.canonical_etypes:
            source_type, _, target_type = canonical
            rows, columns = edge_indices[canonical]
            source = features[source_type]
            target = features[target_type]
            neighborhood = target.new_zeros(target.shape)
            neighborhood.index_add_(0, columns, source[rows])
            degree = target.new_zeros(target.shape[0])
            degree.index_add_(0, columns, torch.ones_like(columns, dtype=target.dtype))
            neighborhood = neighborhood / degree.clamp_min(1).unsqueeze(1)
            weight = F.cosine_similarity(source[rows], neighborhood[columns])
            row_sum = source.new_zeros(source.shape[0])
            row_sum.index_add_(0, rows, weight)
            weight = torch.nan_to_num(weight / row_sum[rows].clamp_min(1e-12))
            weight = torch.where(
                weight > self.attention_threshold, weight, torch.zeros_like(weight)
            )
            result[canonical] = weight
        return result

    @staticmethod
    def _key(canonical):
        return "__".join(canonical)
