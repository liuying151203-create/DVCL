"""Paper-derived HSeCo core.

This module is an independent implementation of the equations in:
Zhao et al., "Robust Heterogeneous GNNs via Semantic Attention and
Contrastive Learning", CIKM 2025, DOI: 10.1145/3746252.3761343.

It does not contain or derive from the authors' source code. Items omitted by
the paper (for example the concrete similarity function and augmentation
distribution) are explicit configuration choices and must be reported.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class HSeCoConfig:
    input_dim: int
    hidden_dim: int = 64
    heads: int = 4
    view_thresholds: Tuple[float, ...] = ()
    global_threshold: float = 0.0
    contrastive_weight: float = 1.0
    perturbation: str = "edge_dropout"
    perturbation_rate: float = 0.2
    add_self_loops: bool = True

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if self.heads <= 0 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if any(not 0.0 <= value <= 1.0 for value in self.view_thresholds):
            raise ValueError("view thresholds must be in [0, 1]")
        if not 0.0 <= self.global_threshold <= 1.0:
            raise ValueError("global_threshold must be in [0, 1]")
        if self.contrastive_weight < 0:
            raise ValueError("contrastive_weight must be non-negative")
        if self.perturbation not in {"edge_dropout", "feature_shuffle"}:
            raise ValueError("perturbation must be edge_dropout or feature_shuffle")
        if not 0.0 <= self.perturbation_rate < 1.0:
            raise ValueError("perturbation_rate must be in [0, 1)")


def _validate_edge_index(edge_index: Tensor, num_nodes: int) -> None:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must use torch.long")
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
        raise ValueError("edge_index contains an invalid node index")


def add_missing_self_loops(edge_index: Tensor, num_nodes: int) -> Tensor:
    _validate_edge_index(edge_index, num_nodes)
    loops = torch.arange(num_nodes, device=edge_index.device)
    loop_index = torch.stack((loops, loops))
    if edge_index.numel() == 0:
        return loop_index
    keys = edge_index[0] * num_nodes + edge_index[1]
    loop_keys = loops * num_nodes + loops
    missing = ~torch.isin(loop_keys, keys)
    return torch.cat((edge_index, loop_index[:, missing]), dim=1)


def segment_softmax(scores: Tensor, source: Tensor, num_nodes: int) -> Tensor:
    """Softmax over outgoing neighbors, corresponding to paper Eq. (3)."""
    if scores.ndim != 1 or source.ndim != 1 or scores.numel() != source.numel():
        raise ValueError("scores and source must be aligned one-dimensional tensors")
    maxima = scores.new_full((num_nodes,), -torch.inf)
    maxima.scatter_reduce_(0, source, scores, reduce="amax", include_self=True)
    exp_scores = torch.exp(scores - maxima[source])
    denominators = scores.new_zeros(num_nodes)
    denominators.index_add_(0, source, exp_scores)
    return exp_scores / denominators[source].clamp_min(torch.finfo(scores.dtype).eps)


def cosine_edge_weights(x: Tensor, edge_index: Tensor) -> Tensor:
    """Cosine instantiation of the unspecified similarity function φ in Eq. (2)."""
    _validate_edge_index(edge_index, x.shape[0])
    source, target = edge_index
    scores = F.cosine_similarity(x[source], x[target], dim=-1)
    return segment_softmax(scores, source, x.shape[0])


def prune_edges(edge_index: Tensor, weights: Tensor, threshold: float) -> Tuple[Tensor, Tensor]:
    if weights.ndim != 1 or weights.numel() != edge_index.shape[1]:
        raise ValueError("weights must align with edge_index")
    keep = weights >= threshold
    return edge_index[:, keep], weights[keep]


class SharedViewEncoder(nn.Module):
    """Shared multi-head weighted aggregator for paper Eq. (4)."""

    def __init__(self, hidden_dim: int, heads: int) -> None:
        super().__init__()
        head_dim = hidden_dim // heads
        self.heads = nn.ModuleList(
            nn.Linear(hidden_dim, head_dim, bias=False) for _ in range(heads)
        )

    def forward(self, x: Tensor, edge_index: Tensor, weights: Tensor) -> Tensor:
        source, target = edge_index
        outputs: List[Tensor] = []
        for projection in self.heads:
            values = projection(x)
            aggregated = values.new_zeros((x.shape[0], values.shape[1]))
            aggregated.index_add_(0, source, values[target] * weights.unsqueeze(-1))
            outputs.append(F.elu(aggregated))
        return torch.cat(outputs, dim=-1)


class SemanticAttention(nn.Module):
    """HAN-style semantic attention used for paper Eq. (5)."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Parameter(torch.empty(hidden_dim))
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.normal_(self.query, std=hidden_dim**-0.5)

    def forward(self, views: Sequence[Tensor]) -> Tensor:
        if not views:
            raise ValueError("at least one semantic view is required")
        scores = [
            (torch.tanh(self.projection(view)).mean(dim=0) * self.query).sum()
            for view in views
        ]
        return torch.softmax(torch.stack(scores), dim=0)


def fuse_view_edges(
    edge_sets: Sequence[Tensor],
    weight_sets: Sequence[Tensor],
    semantic_weights: Tensor,
    num_nodes: int,
) -> Tuple[Tensor, Tensor]:
    """Sparse union implementing the weighted fusion in paper Eq. (6)."""
    all_keys = torch.cat([edges[0] * num_nodes + edges[1] for edges in edge_sets])
    union_keys = torch.unique(all_keys, sorted=True)
    fused = semantic_weights.new_zeros(union_keys.numel())
    for index, (edges, weights) in enumerate(zip(edge_sets, weight_sets)):
        keys = edges[0] * num_nodes + edges[1]
        positions = torch.searchsorted(union_keys, keys)
        fused.index_add_(0, positions, semantic_weights[index] * weights)
    return torch.stack((union_keys // num_nodes, union_keys % num_nodes)), fused


def paper_contrastive_loss(global_h: Tensor, views: Sequence[Tensor], negative_h: Tensor) -> Tensor:
    """Numerically stable form of paper Eq. (7)-(8), without an added temperature."""
    anchor = global_h.flatten()
    negative = F.cosine_similarity(anchor, negative_h.flatten(), dim=0)
    losses = []
    for view in views:
        positive = F.cosine_similarity(anchor, view.flatten(), dim=0)
        losses.append(torch.logaddexp(positive, negative) - positive)
    return torch.stack(losses).mean()


class HSeCo(nn.Module):
    """Native core for semantic purification and structure-aware contrast."""

    def __init__(self, config: HSeCoConfig, num_classes: int) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.config = config
        self.feature_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.encoder = SharedViewEncoder(config.hidden_dim, config.heads)
        self.semantic_attention = SemanticAttention(config.hidden_dim)
        self.classifier = nn.Linear(config.hidden_dim, num_classes)

    def _thresholds(self, count: int) -> Tuple[float, ...]:
        if not self.config.view_thresholds:
            return (0.0,) * count
        if len(self.config.view_thresholds) != count:
            raise ValueError("one threshold is required for each meta-path view")
        return self.config.view_thresholds

    def _perturbed_embedding(
        self, x: Tensor, edge_index: Tensor, weights: Tensor
    ) -> Tensor:
        if weights.numel() == 0:
            return self.encoder(x, edge_index, weights)
        if self.config.perturbation == "feature_shuffle":
            order = torch.randperm(x.shape[0], device=x.device)
            return self.encoder(x[order], edge_index, weights)
        keep = torch.rand(weights.shape, device=weights.device) >= self.config.perturbation_rate
        if not torch.any(keep):
            keep[torch.argmax(weights)] = True
        return self.encoder(x, edge_index[:, keep], weights[keep])

    def forward(self, features: Tensor, meta_path_edges: Sequence[Tensor]) -> Dict[str, object]:
        if not meta_path_edges:
            raise ValueError("meta_path_edges must not be empty")
        x = self.feature_projection(features)
        num_nodes = x.shape[0]
        view_edges: List[Tensor] = []
        view_weights: List[Tensor] = []
        view_embeddings: List[Tensor] = []
        for edges, threshold in zip(meta_path_edges, self._thresholds(len(meta_path_edges))):
            _validate_edge_index(edges, num_nodes)
            if self.config.add_self_loops:
                edges = add_missing_self_loops(edges, num_nodes)
            weights = cosine_edge_weights(x, edges)
            edges, weights = prune_edges(edges, weights, threshold)
            view_edges.append(edges)
            view_weights.append(weights)
            view_embeddings.append(self.encoder(x, edges, weights))

        beta = self.semantic_attention(view_embeddings)
        global_edges, global_weights = fuse_view_edges(
            view_edges, view_weights, beta, num_nodes
        )
        global_edges, global_weights = prune_edges(
            global_edges, global_weights, self.config.global_threshold
        )
        global_h = self.encoder(x, global_edges, global_weights)
        negative_h = self._perturbed_embedding(x, global_edges, global_weights)
        contrastive = paper_contrastive_loss(global_h, view_embeddings, negative_h)
        return {
            "logits": self.classifier(global_h),
            "embedding": global_h,
            "view_embeddings": tuple(view_embeddings),
            "semantic_weights": beta,
            "global_edge_index": global_edges,
            "global_edge_weights": global_weights,
            "contrastive_loss": contrastive,
        }

    def loss(self, outputs: Dict[str, object], labels: Tensor, train_mask: Tensor) -> Tensor:
        logits = outputs["logits"]
        if not isinstance(logits, Tensor):
            raise TypeError("outputs['logits'] must be a tensor")
        classification = F.cross_entropy(logits[train_mask], labels[train_mask])
        contrastive = outputs["contrastive_loss"]
        if not isinstance(contrastive, Tensor):
            raise TypeError("outputs['contrastive_loss'] must be a tensor")
        return classification + self.config.contrastive_weight * contrastive
