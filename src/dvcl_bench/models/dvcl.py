"""Equivalent implementation of the selected DVCL dual-view model."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GraphViewEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads, dropout) -> None:
        super().__init__()
        from dgl.nn.pytorch import GATConv
        self.gat = GATConv(
            input_dim, hidden_dim, heads,
            feat_drop=dropout, attn_drop=dropout, activation=F.elu,
        )

    def forward(self, features: Tensor, graph) -> Tensor:
        import dgl
        return self.gat(dgl.add_self_loop(graph), features).flatten(1)


class DualViewContrastiveDefense(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_classes, heads, dropout,
        feature_mask_rate=0.2, view_mode="both", fusion_mode="concat",
    ) -> None:
        super().__init__()
        if view_mode not in {"both", "both_nocl", "topo", "feat"}:
            raise ValueError(f"Unsupported view mode: {view_mode}")
        if fusion_mode not in {"concat", "gate", "gated_concat"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        self.view_mode = view_mode
        self.fusion_mode = fusion_mode
        self.feature_mask_rate = feature_mask_rate
        size = hidden_dim * heads
        self.topology_encoder = GraphViewEncoder(input_dim, hidden_dim, heads, dropout)
        self.feature_encoder = GraphViewEncoder(input_dim, hidden_dim, heads, dropout)
        self.gate = None
        if fusion_mode in {"gate", "gated_concat"}:
            self.gate = nn.Sequential(
                nn.Linear(size * 2, size), nn.ReLU(), nn.Linear(size, 1), nn.Sigmoid()
            )
        classifier_dim = (
            size * 2
            if view_mode in {"both", "both_nocl"} and fusion_mode in {"concat", "gated_concat"}
            else size
        )
        self.classifier = nn.Linear(classifier_dim, num_classes)
        self.last_gate_weight = None

    def forward(self, features, topology_graph, feature_graph):
        topology = None
        feature = None
        if self.view_mode in {"both", "both_nocl", "topo"}:
            topology = self.topology_encoder(features, topology_graph)
        if self.view_mode in {"both", "both_nocl", "feat"}:
            masked = F.dropout(features, p=self.feature_mask_rate, training=self.training)
            feature = self.feature_encoder(masked, feature_graph)
        return self.classify(topology, feature)

    def classify(self, topology, feature):
        self.last_gate_weight = None
        if self.view_mode in {"both", "both_nocl"}:
            if topology is None or feature is None:
                raise ValueError("Both embeddings are required")
            if self.fusion_mode in {"gate", "gated_concat"}:
                self.last_gate_weight = self.gate(torch.cat((topology, feature), dim=1))
                if self.fusion_mode == "gate":
                    representation = self.last_gate_weight * topology + (1 - self.last_gate_weight) * feature
                else:
                    representation = torch.cat((
                        self.last_gate_weight * topology,
                        (1 - self.last_gate_weight) * feature,
                    ), dim=1)
            else:
                representation = torch.cat((topology, feature), dim=1)
        elif self.view_mode == "topo":
            representation = topology
        else:
            representation = feature
        return self.classifier(representation), topology, feature


def build_feature_knn_graph(features: Tensor, k: int, mode: str = "directed"):
    import dgl
    if mode not in {"directed", "symmetric", "mutual"}:
        raise ValueError(f"Unsupported KNN mode: {mode}")
    num_nodes = features.shape[0]
    if num_nodes <= 1:
        return dgl.graph(([], []), num_nodes=num_nodes).to(features.device)
    k = min(k, num_nodes - 1)
    normalized = F.normalize(features, p=2, dim=1)
    similarity = torch.mm(normalized, normalized.t())
    similarity.fill_diagonal_(float("-inf"))
    neighbors = torch.topk(similarity, k=k, dim=1).indices
    source = torch.arange(num_nodes, device=features.device).repeat_interleave(k)
    target = neighbors.reshape(-1)
    if mode != "directed":
        adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=features.device)
        adjacency[source, target] = True
        adjacency = adjacency | adjacency.t() if mode == "symmetric" else adjacency & adjacency.t()
        source, target = adjacency.nonzero(as_tuple=True)
    return dgl.graph((source.cpu(), target.cpu()), num_nodes=num_nodes).to(features.device)


def cross_view_contrastive_loss(topology, feature, mask, temperature):
    topology = F.normalize(topology[mask.bool()], p=2, dim=1)
    feature = F.normalize(feature[mask.bool()], p=2, dim=1)
    if len(topology) <= 1:
        return topology.new_tensor(0.0)
    logits = torch.mm(topology, feature.t()) / temperature
    labels = torch.arange(len(topology), device=topology.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
