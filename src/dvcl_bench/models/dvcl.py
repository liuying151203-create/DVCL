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
        gate_hidden_dim=16, route_temperature=1.0,
    ) -> None:
        super().__init__()
        if view_mode not in {"both", "both_nocl", "topo", "feat"}:
            raise ValueError(f"Unsupported view mode: {view_mode}")
        if fusion_mode not in {"concat", "gate", "gated_concat", "reliability_gate"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        if fusion_mode == "reliability_gate" and view_mode not in {"both", "both_nocl"}:
            raise ValueError("reliability_gate requires both views")
        if gate_hidden_dim <= 0:
            raise ValueError("gate_hidden_dim must be positive")
        if route_temperature <= 0:
            raise ValueError("route_temperature must be positive")
        self.view_mode = view_mode
        self.fusion_mode = fusion_mode
        self.feature_mask_rate = feature_mask_rate
        self.route_temperature = route_temperature
        size = hidden_dim * heads
        self.topology_encoder = GraphViewEncoder(input_dim, hidden_dim, heads, dropout)
        self.feature_encoder = GraphViewEncoder(input_dim, hidden_dim, heads, dropout)
        self.gate = None
        if fusion_mode in {"gate", "gated_concat"}:
            self.gate = nn.Sequential(
                nn.Linear(size * 2, size), nn.ReLU(), nn.Linear(size, 1), nn.Sigmoid()
            )
        self.topology_classifier = None
        self.feature_classifier = None
        if fusion_mode == "reliability_gate":
            self.topology_classifier = nn.Linear(size, num_classes)
            self.feature_classifier = nn.Linear(size, num_classes)
            self.gate = nn.Sequential(
                nn.Linear(7, gate_hidden_dim), nn.ReLU(),
                nn.Linear(gate_hidden_dim, 1), nn.Sigmoid(),
            )
        classifier_dim = (
            size * 2
            if view_mode in {"both", "both_nocl"}
            and fusion_mode in {"concat", "gated_concat", "reliability_gate"}
            else size
        )
        self.classifier = nn.Linear(classifier_dim, num_classes)
        self.last_gate_weight = None
        self.last_gate_signals = None

    def forward(self, features, topology_graph, feature_graph):
        topology = None
        feature = None
        if self.view_mode in {"both", "both_nocl", "topo"}:
            topology = self.topology_encoder(features, topology_graph)
        if self.view_mode in {"both", "both_nocl", "feat"}:
            masked = F.dropout(features, p=self.feature_mask_rate, training=self.training)
            feature = self.feature_encoder(masked, feature_graph)
        return self.classify(topology, feature)

    def forward_with_topology_embedding(
        self, features, topology_embedding, feature_graph
    ):
        topology = None
        feature = None
        if self.view_mode in {"both", "both_nocl", "topo"}:
            topology = topology_embedding
        if self.view_mode in {"both", "both_nocl", "feat"}:
            masked = F.dropout(
                features, p=self.feature_mask_rate, training=self.training
            )
            feature = self.feature_encoder(masked, feature_graph)
        return self.classify(topology, feature)

    def classify(self, topology, feature):
        self.last_gate_weight = None
        self.last_gate_signals = None
        if self.view_mode in {"both", "both_nocl"}:
            if topology is None or feature is None:
                raise ValueError("Both embeddings are required")
            if self.fusion_mode == "reliability_gate":
                self.last_gate_weight, self.last_gate_signals = (
                    self._reliability_gate(topology, feature)
                )
                representation = torch.cat((
                    self.last_gate_weight * topology,
                    (1 - self.last_gate_weight) * feature,
                ), dim=1)
            elif self.fusion_mode in {"gate", "gated_concat"}:
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

    def reliability_losses(self, topology, feature, labels, mask):
        if self.fusion_mode != "reliability_gate":
            reference = topology if topology is not None else feature
            zero = reference.new_tensor(0.0)
            return {"auxiliary_loss": zero, "route_loss": zero}
        topology_logits = self.topology_classifier(topology)
        feature_logits = self.feature_classifier(feature)
        selected_labels = labels[mask]
        topology_losses = F.cross_entropy(
            topology_logits[mask], selected_labels, reduction="none"
        )
        feature_losses = F.cross_entropy(
            feature_logits[mask], selected_labels, reduction="none"
        )
        route_target = torch.softmax(
            torch.stack((-topology_losses, -feature_losses), dim=1)
            / self.route_temperature,
            dim=1,
        )[:, 0]
        gate_weight, _ = self._reliability_gate(topology, feature)
        return {
            "auxiliary_loss": 0.5 * (
                topology_losses.mean() + feature_losses.mean()
            ),
            "route_loss": F.binary_cross_entropy(
                gate_weight[mask].flatten(), route_target.detach()
            ),
        }

    def _reliability_gate(self, topology, feature):
        topology_logits = self.topology_classifier(topology)
        feature_logits = self.feature_classifier(feature)
        signals = reliability_signals(
            topology_logits, feature_logits, topology, feature
        ).detach()
        return self.gate(signals), signals

    def diagnostic_views(self, topology, feature):
        result = {}
        gate_weight = None
        if self.view_mode in {"both", "both_nocl"}:
            if topology is None or feature is None:
                raise ValueError("Both embeddings are required")
            if self.fusion_mode == "reliability_gate":
                gate_weight, gate_signals = self._reliability_gate(topology, feature)
                topology_representation = torch.cat((
                    gate_weight * topology, torch.zeros_like(feature),
                ), dim=1)
                feature_representation = torch.cat((
                    torch.zeros_like(topology), (1 - gate_weight) * feature,
                ), dim=1)
                result["topology_logits"] = self.topology_classifier(topology)
                result["feature_logits"] = self.feature_classifier(feature)
                result["gate_signals"] = gate_signals
                result["diagnostic_definition"] = (
                    "independent auxiliary view classifiers"
                )
            elif self.fusion_mode in {"gate", "gated_concat"}:
                gate_weight = self.gate(torch.cat((topology, feature), dim=1))
            if self.fusion_mode == "concat":
                zeros = torch.zeros_like(topology)
                topology_representation = torch.cat((topology, zeros), dim=1)
                feature_representation = torch.cat((zeros, feature), dim=1)
            elif self.fusion_mode == "gated_concat":
                zeros = torch.zeros_like(topology)
                topology_representation = torch.cat((
                    gate_weight * topology, zeros,
                ), dim=1)
                feature_representation = torch.cat((
                    zeros, (1 - gate_weight) * feature,
                ), dim=1)
            else:
                if self.fusion_mode == "gate":
                    topology_representation = gate_weight * topology
                    feature_representation = (1 - gate_weight) * feature
            if self.fusion_mode != "reliability_gate":
                result["topology_logits"] = self.classifier(topology_representation)
                result["feature_logits"] = self.classifier(feature_representation)
                result["diagnostic_definition"] = (
                    "same-checkpoint branch zero-ablation diagnostics"
                )
        elif self.view_mode == "topo":
            result["topology_logits"] = self.classifier(topology)
        else:
            result["feature_logits"] = self.classifier(feature)
        if gate_weight is not None:
            result["gate_weight"] = gate_weight
        return result


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


def perturb_topology_graph(graph, rate: float):
    import dgl
    if not 0 <= rate < 1:
        raise ValueError("Topology augmentation rate must be in [0, 1)")
    source, target = graph.edges()
    edge_count = int(source.numel())
    replacement_count = int(round(edge_count * rate))
    if replacement_count == 0:
        return graph
    keep = torch.randperm(edge_count, device=source.device)[replacement_count:]
    additions_source = torch.randint(
        graph.num_nodes(), (replacement_count,), device=source.device
    )
    additions_target = torch.randint(
        graph.num_nodes(), (replacement_count,), device=target.device
    )
    return dgl.graph(
        (
            torch.cat((source[keep], additions_source)),
            torch.cat((target[keep], additions_target)),
        ),
        num_nodes=graph.num_nodes(),
        device=graph.device,
    )


def reliability_signals(topology_logits, feature_logits, topology, feature):
    topology_probabilities = topology_logits.softmax(dim=1)
    feature_probabilities = feature_logits.softmax(dim=1)
    class_scale = torch.log(
        topology_logits.new_tensor(topology_logits.shape[1])
    ).clamp_min(1.0)
    topology_entropy = -(
        topology_probabilities * topology_probabilities.clamp_min(1e-12).log()
    ).sum(dim=1, keepdim=True) / class_scale
    feature_entropy = -(
        feature_probabilities * feature_probabilities.clamp_min(1e-12).log()
    ).sum(dim=1, keepdim=True) / class_scale
    topology_margin = _confidence_margin(topology_probabilities)
    feature_margin = _confidence_margin(feature_probabilities)
    midpoint = 0.5 * (topology_probabilities + feature_probabilities)
    disagreement = 0.5 * (
        F.kl_div(
            midpoint.clamp_min(1e-12).log(), topology_probabilities,
            reduction="none",
        ).sum(dim=1, keepdim=True)
        + F.kl_div(
            midpoint.clamp_min(1e-12).log(), feature_probabilities,
            reduction="none",
        ).sum(dim=1, keepdim=True)
    )
    consistency = F.cosine_similarity(topology, feature, dim=1, eps=1e-12).unsqueeze(1)
    norm_ratio = torch.log(
        topology.norm(dim=1, keepdim=True).clamp_min(1e-12)
        / feature.norm(dim=1, keepdim=True).clamp_min(1e-12)
    ).clamp(-5.0, 5.0)
    return torch.cat((
        topology_entropy, feature_entropy,
        topology_margin, feature_margin,
        disagreement, consistency, norm_ratio,
    ), dim=1)


def _confidence_margin(probabilities):
    if probabilities.shape[1] <= 1:
        return probabilities.new_zeros((probabilities.shape[0], 1))
    values = probabilities.topk(k=2, dim=1).values
    return values[:, :1] - values[:, 1:2]


def cross_view_contrastive_loss(topology, feature, mask, temperature):
    topology = F.normalize(topology[mask.bool()], p=2, dim=1)
    feature = F.normalize(feature[mask.bool()], p=2, dim=1)
    if len(topology) <= 1:
        return topology.new_tensor(0.0)
    logits = torch.mm(topology, feature.t()) / temperature
    labels = torch.arange(len(topology), device=topology.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
