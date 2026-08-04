"""Native HSeCo and DVCL training adapters over frozen artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
from torch import nn

from .artifacts import AttackArtifact, CleanGraphArtifact, SplitArtifact
from .environment import resolve_device
from .models.dvcl import (
    DualViewContrastiveDefense,
    build_feature_knn_graph,
    cross_view_contrastive_loss,
)
from .models.semantic import (
    NodeLevelAggregator,
    SemanticHAN,
    perturb_matrix,
    purified_graphs,
    semantic_graph,
    transition_edges,
)
from .models.baselines import HeteroSAGE
from .graph_adapter import hete_adjs_to_dgl, meta_path_adjacency
from .training import (
    DVCLTrainConfig,
    HANTrainConfig,
    HSeCoTrainConfig,
    HeteroSAGETrainConfig,
    LegacyEarlyStopping,
    TrainingResult,
    classification_metrics,
    save_checkpoint,
    set_random_seed,
)


def train_han(
    clean: CleanGraphArtifact,
    split: SplitArtifact,
    attack: Optional[AttackArtifact],
    config: HANTrainConfig,
    train_seed: int,
    epochs: int,
    patience: int,
    device: str,
    checkpoint_path: Path,
) -> TrainingResult:
    import dgl

    set_random_seed(train_seed)
    target = _device(device)
    features, labels, masks = _inputs(clean, split, target)
    selected_adjs = _selected_adjs(clean, attack)
    views = [
        dgl.from_scipy(meta_path_adjacency(selected_adjs, path)).to(target)
        for path in clean.meta_paths
    ]
    model = SemanticHAN(
        len(clean.meta_paths), features.shape[1], config.hidden_dim,
        clean.num_classes, config.heads, config.dropout,
    ).to(target)
    result = _train_supervised(
        model=model,
        forward=lambda: model(features, views),
        labels=labels,
        masks=masks,
        config=config,
        epochs=epochs,
        patience=patience,
        checkpoint_path=checkpoint_path,
    )
    result.diagnostics.update({
        "semantic_attention": model.semantic_weights().detach().cpu().tolist(),
        "meta_path_edges": [int(graph.num_edges()) for graph in views],
    })
    return result


def train_heterosage(
    clean: CleanGraphArtifact,
    split: SplitArtifact,
    attack: Optional[AttackArtifact],
    config: HeteroSAGETrainConfig,
    train_seed: int,
    epochs: int,
    patience: int,
    device: str,
    checkpoint_path: Path,
) -> TrainingResult:
    set_random_seed(train_seed)
    target = _device(device)
    features, labels, masks = _inputs(clean, split, target)
    graph = hete_adjs_to_dgl(
        _selected_adjs(clean, attack), clean.canonical_etypes, clean.node_counts
    ).to(target)
    node_features = {
        node_type: features.new_zeros((count, features.shape[1]))
        for node_type, count in clean.node_counts.items()
    }
    node_features[clean.predict_ntype] = features
    model = HeteroSAGE(
        clean.canonical_etypes,
        features.shape[1],
        config.hidden_dim,
        clean.num_classes,
        config.num_layers,
        config.dropout,
    ).to(target)
    result = _train_supervised(
        model=model,
        forward=lambda: model(graph, node_features)[clean.predict_ntype],
        labels=labels,
        masks=masks,
        config=config,
        epochs=epochs,
        patience=patience,
        checkpoint_path=checkpoint_path,
    )
    result.diagnostics.update({
        "num_layers": config.num_layers,
        "heterogeneous_edges": int(graph.num_edges()),
        "non_target_feature_fill": "zero",
    })
    return result


def train_hseco(
    clean: CleanGraphArtifact,
    split: SplitArtifact,
    attack: Optional[AttackArtifact],
    config: HSeCoTrainConfig,
    train_seed: int,
    epochs: int,
    patience: int,
    device: str,
    checkpoint_path: Path,
) -> TrainingResult:
    import dgl

    config = config.freeze_for_dataset(clean.dataset)
    set_random_seed(train_seed)
    rng = np.random.RandomState(train_seed)
    target = _device(device)
    features, labels, masks = _inputs(clean, split, target)
    transitions = transition_edges(features, _selected_adjs(clean, attack), clean.meta_paths)
    views = purified_graphs(transitions, config.thresholds)
    semantic = SemanticHAN(
        len(clean.meta_paths), features.shape[1], config.hidden_dim,
        clean.num_classes, config.semantic_heads, config.semantic_dropout,
    ).to(target)
    node_model = NodeLevelAggregator(
        features.shape[1], config.node_hidden_dim, clean.num_classes,
        config.node_heads, config.node_dropout,
    ).to(target)
    semantic_optimizer = torch.optim.Adam(
        semantic.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    node_optimizer = torch.optim.Adam(
        node_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()
    holder = node_model if config.legacy_checkpoint_semantics else nn.ModuleDict({
        "semantic": semantic, "node": node_model
    })
    stopper = LegacyEarlyStopping(patience)
    history = []
    last_graph = None
    last_attention = None
    stopped_epoch = 0

    for epoch in range(epochs):
        semantic.train()
        node_model.train()
        semantic_logits = semantic(features, views)
        attention = semantic.semantic_weights()
        matrix = semantic_graph(transitions, attention, config.global_threshold)
        topology_graph = dgl.from_scipy(matrix).to(target)
        negative_graph = dgl.from_scipy(
            perturb_matrix(matrix, config.negative_noise_rate, rng)
        ).to(target)
        logits = node_model(features, topology_graph)
        semantic_loss = loss_fn(semantic_logits[masks["train"]], labels[masks["train"]])
        classification_loss = loss_fn(logits[masks["train"]], labels[masks["train"]])
        contrastive_loss = node_model.contrastive_loss(
            features, views, topology_graph, negative_graph, masks["train"]
        )
        node_loss = classification_loss + config.contrastive_weight * contrastive_loss
        semantic_optimizer.zero_grad()
        node_optimizer.zero_grad()
        semantic_loss.backward()
        node_loss.backward()
        semantic_optimizer.step()
        node_optimizer.step()

        node_model.eval()
        with torch.no_grad():
            validation_logits = node_model(features, topology_graph)
            validation_loss = loss_fn(
                validation_logits[masks["val"]], labels[masks["val"]]
            )
            validation = classification_metrics(
                validation_logits[masks["val"]], labels[masks["val"]]
            )
        history.append({
            "epoch": epoch,
            "semantic_loss": float(semantic_loss.detach()),
            "classification_loss": float(classification_loss.detach()),
            "contrastive_loss": float(contrastive_loss.detach()),
            "val_loss": float(validation_loss),
            **{f"val_{key}": value for key, value in validation.items()},
        })
        last_graph = topology_graph
        last_attention = attention.detach().cpu()
        stopped_epoch = epoch
        if stopper.step(float(validation_loss), validation["accuracy"], holder, epoch):
            break

    stopper.restore(holder)
    if not config.legacy_checkpoint_semantics:
        semantic.eval()
        with torch.no_grad():
            semantic(features, views)
            last_attention = semantic.semantic_weights().detach().cpu()
        last_graph = dgl.from_scipy(
            semantic_graph(transitions, last_attention.to(target), config.global_threshold)
        ).to(target)
    node_model.eval()
    with torch.no_grad():
        test_logits = node_model(features, last_graph)
    metrics = classification_metrics(test_logits[masks["test"]], labels[masks["test"]])
    save_checkpoint(checkpoint_path, holder, config, stopper.best_epoch)
    return TrainingResult(
        metrics, history, stopper.best_epoch, stopped_epoch,
        {"semantic_attention": last_attention.tolist()},
    )


def train_dvcl(
    clean: CleanGraphArtifact,
    split: SplitArtifact,
    attack: Optional[AttackArtifact],
    config: DVCLTrainConfig,
    train_seed: int,
    epochs: int,
    patience: int,
    device: str,
    checkpoint_path: Path,
) -> TrainingResult:
    import dgl

    config = config.freeze_for_dataset(clean.dataset)
    set_random_seed(train_seed)
    target = _device(device)
    features, labels, masks = _inputs(clean, split, target)
    transitions = transition_edges(features, _selected_adjs(clean, attack), clean.meta_paths)
    views = purified_graphs(transitions, config.thresholds)
    feature_graph = build_feature_knn_graph(features, config.knn_k, config.knn_mode)
    semantic = SemanticHAN(
        len(clean.meta_paths), features.shape[1], config.semantic_hidden_dim,
        clean.num_classes, config.semantic_heads, config.dropout,
    ).to(target)
    model = DualViewContrastiveDefense(
        features.shape[1], config.hidden_dim, clean.num_classes, config.heads,
        config.dropout, config.feature_mask_rate, config.view_mode, config.fusion_mode,
    ).to(target)
    optimizer = torch.optim.Adam(
        list(semantic.parameters()) + list(model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    holder = model if config.legacy_checkpoint_semantics else nn.ModuleDict({
        "semantic": semantic, "dvcl": model
    })
    stopper = LegacyEarlyStopping(patience)
    history = []
    last_graph = None
    last_attention = None
    stopped_epoch = 0

    for epoch in range(epochs):
        semantic.train()
        model.train()
        semantic_logits = semantic(features, views)
        attention = semantic.semantic_weights()
        matrix = semantic_graph(transitions, attention, config.global_threshold)
        topology_graph = dgl.from_scipy(matrix).to(target)
        logits, topology, feature = model(features, topology_graph, feature_graph)
        semantic_loss = loss_fn(semantic_logits[masks["train"]], labels[masks["train"]])
        classification_loss = loss_fn(logits[masks["train"]], labels[masks["train"]])
        if config.view_mode == "both":
            contrastive_loss = cross_view_contrastive_loss(
                topology, feature, masks["train"], config.temperature
            )
        else:
            contrastive_loss = logits.new_tensor(0.0)
        total_loss = (
            config.lambda_han * semantic_loss
            + classification_loss
            + config.lambda_dvcl * contrastive_loss
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits, _, _ = model(features, topology_graph, feature_graph)
            validation_loss = loss_fn(
                validation_logits[masks["val"]], labels[masks["val"]]
            )
            validation = classification_metrics(
                validation_logits[masks["val"]], labels[masks["val"]]
            )
        history.append({
            "epoch": epoch,
            "semantic_loss": float(semantic_loss.detach()),
            "classification_loss": float(classification_loss.detach()),
            "contrastive_loss": float(contrastive_loss.detach()),
            "total_loss": float(total_loss.detach()),
            "val_loss": float(validation_loss),
            **{f"val_{key}": value for key, value in validation.items()},
        })
        last_graph = topology_graph
        last_attention = attention.detach().cpu()
        stopped_epoch = epoch
        if stopper.step(float(validation_loss), validation["accuracy"], holder, epoch):
            break

    stopper.restore(holder)
    if not config.legacy_checkpoint_semantics:
        semantic.eval()
        with torch.no_grad():
            semantic(features, views)
            last_attention = semantic.semantic_weights().detach().cpu()
        last_graph = dgl.from_scipy(
            semantic_graph(transitions, last_attention.to(target), config.global_threshold)
        ).to(target)
    model.eval()
    with torch.no_grad():
        test_logits, topology, feature = model(features, last_graph, feature_graph)
    metrics = classification_metrics(test_logits[masks["test"]], labels[masks["test"]])
    save_checkpoint(checkpoint_path, holder, config, stopper.best_epoch)
    diagnostics = {
        "semantic_attention": last_attention.tolist(),
        "feature_knn_edges": int(feature_graph.num_edges()),
        "view_mode": config.view_mode,
    }
    if model.last_gate_weight is not None:
        diagnostics["gate_mean"] = float(model.last_gate_weight.mean())
    return TrainingResult(metrics, history, stopper.best_epoch, stopped_epoch, diagnostics)


def _train_supervised(
    model,
    forward,
    labels,
    masks,
    config,
    epochs,
    patience,
    checkpoint_path,
):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()
    stopper = LegacyEarlyStopping(patience)
    history = []
    stopped_epoch = 0
    for epoch in range(epochs):
        model.train()
        logits = forward()
        classification_loss = loss_fn(logits[masks["train"]], labels[masks["train"]])
        optimizer.zero_grad()
        classification_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = forward()
            validation_loss = loss_fn(
                validation_logits[masks["val"]], labels[masks["val"]]
            )
            validation = classification_metrics(
                validation_logits[masks["val"]], labels[masks["val"]]
            )
        history.append({
            "epoch": epoch,
            "classification_loss": float(classification_loss.detach()),
            "val_loss": float(validation_loss),
            **{f"val_{key}": value for key, value in validation.items()},
        })
        stopped_epoch = epoch
        if stopper.step(float(validation_loss), validation["accuracy"], model, epoch):
            break

    stopper.restore(model)
    model.eval()
    with torch.no_grad():
        test_logits = forward()
    metrics = classification_metrics(test_logits[masks["test"]], labels[masks["test"]])
    save_checkpoint(checkpoint_path, model, config, stopper.best_epoch)
    return TrainingResult(metrics, history, stopper.best_epoch, stopped_epoch)


def _selected_adjs(
    clean: CleanGraphArtifact,
    attack: Optional[AttackArtifact],
) -> Mapping:
    if attack is None:
        return clean.hete_adjs
    if attack.dataset != clean.dataset or attack.clean_version != clean.version:
        raise ValueError("Attack artifact does not belong to the selected clean graph")
    return attack.perturbed_hete_adjs


def _inputs(clean: CleanGraphArtifact, split: SplitArtifact, device):
    if split.dataset != clean.dataset or len(split.train_mask) != len(clean.labels):
        raise ValueError("Split artifact does not belong to the selected clean graph")
    features = clean.features.to(device)
    labels = clean.labels.to(device)
    masks = {
        "train": split.train_mask.bool().to(device),
        "val": split.val_mask.bool().to(device),
        "test": split.test_mask.bool().to(device),
    }
    return features, labels, masks


def _device(requested: str):
    return resolve_device(requested)
