"""Native HSeCo and DVCL training adapters over frozen artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
from torch import nn

from .adaptive import (
    greedy_query_target_changes,
    record_query_count,
    records_at_budget,
)
from .artifacts import (
    AttackArtifact,
    CleanGraphArtifact,
    SplitArtifact,
    file_sha256,
    save_attack_artifact,
)
from .attacks import apply_target_change, build_attack_artifact, verify_attack
from .environment import resolve_device
from .manifest import save_json
from .models.dvcl import (
    DualViewContrastiveDefense,
    build_feature_knn_graph,
    cross_view_contrastive_loss,
    perturb_topology_graph,
)
from .models.semantic import (
    NodeLevelAggregator,
    SemanticHAN,
    feature_similarity,
    perturb_matrix,
    purified_graphs,
    semantic_graph,
    transition_edges,
)
from .models.baselines import HeteroGuard, HeteroSAGE
from .models.rohe import RoHe
from .models.fastrohgcn import FastRoHGCN
from .profiling import profile_inference
from .graph_adapter import hete_adjs_to_dgl, meta_path_adjacency, sparse_to_edge_index
from .training import (
    DVCLTrainConfig,
    HANTrainConfig,
    HSeCoTrainConfig,
    HeteroSAGETrainConfig,
    HeteroGuardTrainConfig,
    RoHeTrainConfig,
    FastRoHGCNTrainConfig,
    LegacyEarlyStopping,
    TrainingResult,
    classification_metrics,
    restore_checkpoint,
    save_checkpoint,
    set_random_seed,
)
from .view_diagnostics import target_view_diagnostics


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
    checkpoint_source: Optional[Path] = None,
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
        checkpoint_source=checkpoint_source,
    )
    result.diagnostics.update({
        "semantic_attention": model.semantic_weights().detach().cpu().tolist(),
        "meta_path_edges": [int(graph.num_edges()) for graph in views],
    })
    if _is_target_evasion(attack):
        model.eval()
        with torch.no_grad():
            clean_logits = model(features, views)
        _evaluate_model_target_evasion(
            result, clean, split, attack, labels, clean_logits,
            lambda adjs, record: model(features, [
                dgl.from_scipy(meta_path_adjacency(adjs, path)).to(target)
                for path in clean.meta_paths
            ]),
            checkpoint_path, "han", config, train_seed,
        )
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
    checkpoint_source: Optional[Path] = None,
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
        checkpoint_source=checkpoint_source,
    )
    result.diagnostics.update({
        "num_layers": config.num_layers,
        "heterogeneous_edges": int(graph.num_edges()),
        "non_target_feature_fill": "zero",
    })
    if _is_target_evasion(attack):
        model.eval()
        with torch.no_grad():
            clean_logits = model(graph, node_features)[clean.predict_ntype]

        def target_forward(adjs, record):
            attacked_graph = hete_adjs_to_dgl(
                adjs, clean.canonical_etypes, clean.node_counts
            ).to(target)
            return model(attacked_graph, node_features)[clean.predict_ntype]

        _evaluate_model_target_evasion(
            result, clean, split, attack, labels, clean_logits, target_forward,
            checkpoint_path, "heterosage", config, train_seed,
        )
    return result


def train_heteroguard(
    clean: CleanGraphArtifact,
    split: SplitArtifact,
    attack: Optional[AttackArtifact],
    config: HeteroGuardTrainConfig,
    train_seed: int,
    epochs: int,
    patience: int,
    device: str,
    checkpoint_path: Path,
    checkpoint_source: Optional[Path] = None,
) -> TrainingResult:
    set_random_seed(train_seed)
    target = _device(device)
    features, labels, masks = _inputs(clean, split, target)
    node_features = {
        node_type: features.new_zeros((count, features.shape[1]))
        for node_type, count in clean.node_counts.items()
    }
    node_features[clean.predict_ntype] = features

    def edge_indices(adjs):
        return {
            canonical: sparse_to_edge_index(adjs[canonical[1]], target)
            for canonical in clean.canonical_etypes
        }

    selected_edges = edge_indices(_selected_adjs(clean, attack))
    model = HeteroGuard(
        clean.canonical_etypes,
        features.shape[1],
        config.hidden_dim,
        clean.num_classes,
        config.num_layers,
        config.dropout,
        config.attention_threshold,
        config.gated_attention,
    ).to(target)
    result = _train_supervised(
        model=model,
        forward=lambda: model(node_features, selected_edges)[clean.predict_ntype],
        labels=labels,
        masks=masks,
        config=config,
        epochs=epochs,
        patience=patience,
        checkpoint_path=checkpoint_path,
        checkpoint_source=checkpoint_source,
    )
    result.diagnostics.update({
        "attention_threshold": config.attention_threshold,
        "gated_attention": config.gated_attention,
        "attention_density": model.last_attention_density,
        "non_target_feature_fill": "zero",
    })
    if _is_target_evasion(attack):
        model.eval()
        with torch.no_grad():
            clean_logits = model(node_features, selected_edges)[clean.predict_ntype]
        _evaluate_model_target_evasion(
            result, clean, split, attack, labels, clean_logits,
            lambda adjs, record: model(node_features, edge_indices(adjs))[clean.predict_ntype],
            checkpoint_path, "heteroguard", config, train_seed,
        )
    return result


def train_rohe(
    clean: CleanGraphArtifact,
    split: SplitArtifact,
    attack: Optional[AttackArtifact],
    config: RoHeTrainConfig,
    train_seed: int,
    epochs: int,
    patience: int,
    device: str,
    checkpoint_path: Path,
    checkpoint_source: Optional[Path] = None,
) -> TrainingResult:
    config = config.freeze_for_dataset(clean.dataset)
    set_random_seed(train_seed)
    target = _device(device)
    features, labels, masks = _inputs(clean, split, target)

    def transitions(adjs):
        normalized = {}
        for name, value in adjs.items():
            value = value.tocsr()
            degree = np.asarray(value.sum(axis=1)).reshape(-1)
            degree = np.where(degree > 0, degree, 1)
            normalized[name] = sp.diags(1.0 / degree).dot(value)
        result = []
        for path in clean.meta_paths:
            matrix = normalized[path[0]]
            for name in path[1:]:
                matrix = matrix.dot(normalized[name])
            result.append(matrix.tocsr())
        return result

    import scipy.sparse as sp

    selected_transitions = transitions(_selected_adjs(clean, attack))
    model = RoHe(
        len(clean.meta_paths), features.shape[1], config.hidden_dim,
        clean.num_classes, config.heads, config.dropout, config.top_t,
    ).to(target)
    result = _train_supervised(
        model=model,
        forward=lambda: model(features, selected_transitions),
        labels=labels,
        masks=masks,
        config=config,
        epochs=epochs,
        patience=patience,
        checkpoint_path=checkpoint_path,
        checkpoint_source=checkpoint_source,
    )
    result.diagnostics.update({
        "top_t": config.top_t,
        "semantic_attention": model.semantic_weights().detach().cpu().tolist(),
    })
    if _is_target_evasion(attack):
        model.eval()
        with torch.no_grad():
            clean_logits = model(features, selected_transitions)
        _evaluate_model_target_evasion(
            result, clean, split, attack, labels, clean_logits,
            lambda adjs, record: model(features, transitions(adjs)),
            checkpoint_path, "rohe", config, train_seed,
        )
    return result


def train_fastrohgcn(
    clean: CleanGraphArtifact,
    split: SplitArtifact,
    attack: Optional[AttackArtifact],
    config: FastRoHGCNTrainConfig,
    train_seed: int,
    epochs: int,
    patience: int,
    device: str,
    checkpoint_path: Path,
    checkpoint_source: Optional[Path] = None,
) -> TrainingResult:
    import dgl
    import scipy.sparse as sp

    set_random_seed(train_seed)
    target = _device(device)
    features, labels, masks = _inputs(clean, split, target)
    node_types = list(clean.node_counts)
    offsets = {}
    offset = 0
    for node_type in node_types:
        offsets[node_type] = offset
        offset += clean.node_counts[node_type]
    target_start = offsets[clean.predict_ntype]
    target_end = target_start + clean.node_counts[clean.predict_ntype]

    def preprocess(adjs):
        rows, columns, weights = [], [], []
        for source_type, relation, target_type in clean.canonical_etypes:
            adjacency = adjs[relation].tocsr().astype(np.float32)
            degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
            degree = np.where(degree > 0, degree, 1)
            normalized = sp.diags(1.0 / degree).dot(adjacency).tocoo()
            rows.extend((normalized.row + offsets[source_type]).tolist())
            columns.extend((normalized.col + offsets[target_type]).tolist())
            weights.extend(normalized.data.tolist())

        topology = None
        for path in clean.meta_paths:
            value = meta_path_adjacency(adjs, path).astype(np.float32)
            topology = value if topology is None else topology + value
        topology = topology.tocsr()
        virtual = set()
        for row in range(topology.shape[0]):
            start, end = topology.indptr[row], topology.indptr[row + 1]
            candidates = topology.indices[start:end]
            scores = topology.data[start:end]
            if candidates.size:
                order = np.argsort(scores)[-config.topk_similarity:]
                virtual.update((row, int(candidates[index])) for index in order)
        feature_graph = build_feature_knn_graph(
            features, config.topk_similarity, "symmetric"
        )
        feature_rows, feature_columns = feature_graph.edges()
        virtual.update(zip(
            feature_rows.detach().cpu().tolist(),
            feature_columns.detach().cpu().tolist(),
        ))
        for row, column in virtual:
            if row != column:
                rows.append(row + target_start)
                columns.append(column + target_start)
                weights.append(1.0)
        for node in range(target_start, target_end):
            rows.append(node)
            columns.append(node)
            weights.append(config.self_loop_weight)
        matrix = sp.csr_matrix((weights, (rows, columns)), shape=(offset, offset))
        matrix.sum_duplicates()
        norm = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).reshape(-1))
        norm = np.where(norm > 0, norm, 1)
        matrix = sp.diags(1.0 / norm).dot(matrix).tocoo()
        graph = dgl.graph(
            (torch.from_numpy(matrix.row), torch.from_numpy(matrix.col)),
            num_nodes=offset,
        ).to(target)
        edge_weight = torch.as_tensor(matrix.data, dtype=features.dtype, device=target)
        return graph, edge_weight

    selected_graph, selected_weight = preprocess(_selected_adjs(clean, attack))
    model = FastRoHGCN(
        clean.node_counts, clean.predict_ntype, features.shape[1],
        config.projection_dim, config.hidden_dim, config.layers,
        clean.num_classes, config.dropout,
    ).to(target)

    def forward(graph=selected_graph, edge_weight=selected_weight):
        return model(features, graph, edge_weight)[target_start:target_end]

    result = _train_supervised(
        model=model,
        forward=forward,
        labels=labels,
        masks=masks,
        config=config,
        epochs=epochs,
        patience=patience,
        checkpoint_path=checkpoint_path,
        checkpoint_source=checkpoint_source,
    )
    result.diagnostics.update({
        "implementation": "independent reproduction from official FastRo-HGCN",
        "official_revision": "ab938c28fb3d6c22a509f4d1f5050d810fb4e84a",
        "topk_similarity": config.topk_similarity,
        "self_loop_weight": config.self_loop_weight,
        "full_graph_edges": int(selected_graph.num_edges()),
    })
    if _is_target_evasion(attack):
        model.eval()
        with torch.no_grad():
            clean_logits = forward()

        def target_forward(adjs, record):
            graph, edge_weight = preprocess(adjs)
            return model(features, graph, edge_weight)[target_start:target_end]

        _evaluate_model_target_evasion(
            result, clean, split, attack, labels, clean_logits, target_forward,
            checkpoint_path, "fastrohgcn", config, train_seed,
        )
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
    checkpoint_source: Optional[Path] = None,
) -> TrainingResult:
    import dgl

    config = config.freeze_for_dataset(clean.dataset)
    set_random_seed(train_seed)
    rng = np.random.RandomState(train_seed)
    target = _device(device)
    features, labels, masks = _inputs(clean, split, target)
    similarity = feature_similarity(features)
    transitions = transition_edges(
        features, _selected_adjs(clean, attack), clean.meta_paths, similarity
    )
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
    if checkpoint_source is not None:
        if config.legacy_checkpoint_semantics:
            raise ValueError("Checkpoint reuse requires full HSeCo checkpoint semantics")
        best_epoch = restore_checkpoint(
            checkpoint_source, checkpoint_path, holder, config
        )
        semantic.eval()
        node_model.eval()
        with torch.no_grad():
            semantic(features, views)
            last_attention = semantic.semantic_weights().detach().cpu()
            last_graph = dgl.from_scipy(
                semantic_graph(
                    transitions, last_attention.to(target), config.global_threshold
                )
            ).to(target)
            test_logits = node_model(features, last_graph)
        metrics = classification_metrics(
            test_logits[masks["test"]], labels[masks["test"]]
        )
        profile_inference(
            (semantic, node_model), lambda: node_model(features, last_graph)
        )
        result = TrainingResult(
            metrics,
            [{"epoch": best_epoch, "checkpoint_reused": True}],
            best_epoch,
            best_epoch,
            {
                "semantic_attention": last_attention.tolist(),
                "checkpoint_reused": True,
                "checkpoint_source": str(Path(checkpoint_source).resolve()),
                "optimizer_steps": 0,
            },
        )
        if _is_target_evasion(attack):
            def target_forward(adjs, record):
                attacked_transitions = transition_edges(
                    features, adjs, clean.meta_paths, similarity
                )
                attacked_graph = dgl.from_scipy(semantic_graph(
                    attacked_transitions,
                    last_attention.to(target),
                    config.global_threshold,
                )).to(target)
                return node_model(features, attacked_graph)

            _evaluate_model_target_evasion(
                result, clean, split, attack, labels, test_logits, target_forward,
                checkpoint_path, "hseco", config, train_seed,
            )
        return result
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
    profile_inference(
        (semantic, node_model), lambda: node_model(features, last_graph)
    )
    result = TrainingResult(
        metrics, history, stopper.best_epoch, stopped_epoch,
        {"semantic_attention": last_attention.tolist()},
    )
    if _is_target_evasion(attack):
        def target_forward(adjs, record):
            attacked_transitions = transition_edges(
                features, adjs, clean.meta_paths, similarity
            )
            attacked_graph = dgl.from_scipy(semantic_graph(
                attacked_transitions, last_attention.to(target), config.global_threshold
            )).to(target)
            return node_model(features, attacked_graph)

        _evaluate_model_target_evasion(
            result, clean, split, attack, labels, test_logits, target_forward,
            checkpoint_path, "hseco", config, train_seed,
        )
    return result


def _validate_dvcl_strategy(config: DVCLTrainConfig) -> None:
    if config.topology_source not in {"graph", "han_semantic"}:
        raise ValueError(
            f"Unsupported DVCL topology source: {config.topology_source}"
        )
    if config.semantic_topology_filter not in {"hard", "none"}:
        raise ValueError(
            "Unsupported semantic topology filter: "
            f"{config.semantic_topology_filter}"
        )
    if config.topology_source != "han_semantic":
        return
    semantic_size = config.semantic_hidden_dim * config.semantic_heads
    topology_size = config.hidden_dim * config.heads
    if semantic_size != topology_size:
        raise ValueError(
            "han_semantic topology dimension mismatch: "
            f"semantic={semantic_size} dvcl={topology_size}"
        )
    if config.legacy_checkpoint_semantics:
        raise ValueError(
            "han_semantic requires full semantic and DVCL checkpoint semantics"
        )
    if config.structure_augment_rate > 0:
        raise ValueError(
            "Topology graph augmentation is not supported for han_semantic"
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
    checkpoint_source: Optional[Path] = None,
) -> TrainingResult:
    import dgl

    config = config.freeze_for_dataset(clean.dataset)
    set_random_seed(train_seed)
    target = _device(device)
    features, labels, masks = _inputs(clean, split, target)
    topology_enabled = config.view_mode in {"both", "both_nocl", "topo"}
    _validate_dvcl_strategy(config)
    use_han_semantic_topology = (
        topology_enabled and config.topology_source == "han_semantic"
    )
    apply_semantic_topology_filter = config.semantic_topology_filter == "hard"
    similarity = feature_similarity(features) if topology_enabled else None
    transitions = None
    views = None
    if topology_enabled:
        transitions = transition_edges(
            features, _selected_adjs(clean, attack), clean.meta_paths, similarity
        )
        views = purified_graphs(transitions, config.thresholds)
    feature_graph = build_feature_knn_graph(features, config.knn_k, config.knn_mode)
    semantic = SemanticHAN(
        len(clean.meta_paths), features.shape[1], config.semantic_hidden_dim,
        clean.num_classes, config.semantic_heads, config.dropout,
    ).to(target)
    model = DualViewContrastiveDefense(
        features.shape[1], config.hidden_dim, clean.num_classes, config.heads,
        config.dropout, config.feature_mask_rate, config.view_mode, config.fusion_mode,
        config.gate_hidden_dim, config.route_temperature,
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
    if checkpoint_source is not None:
        if config.legacy_checkpoint_semantics:
            raise ValueError("Checkpoint reuse requires full DVCL checkpoint semantics")
        best_epoch = restore_checkpoint(
            checkpoint_source, checkpoint_path, holder, config
        )
        semantic.eval()
        model.eval()
        with torch.no_grad():
            last_attention = None
            last_graph = None
            semantic_embedding = None
            if topology_enabled:
                semantic_embedding = semantic.encode(features, views)
                last_attention = semantic.semantic_weights().detach().cpu()
                if not use_han_semantic_topology:
                    last_graph = dgl.from_scipy(
                        semantic_graph(
                            transitions,
                            last_attention.to(target),
                            config.global_threshold,
                            apply_filter=apply_semantic_topology_filter,
                        )
                    ).to(target)
            if use_han_semantic_topology:
                test_logits, topology, feature = (
                    model.forward_with_topology_embedding(
                        features, semantic_embedding, feature_graph
                    )
                )
            else:
                test_logits, topology, feature = model(
                    features, last_graph, feature_graph
                )
            clean_view_state = _dvcl_view_state(
                model, test_logits, topology, feature
            )
        metrics = classification_metrics(
            test_logits[masks["test"]], labels[masks["test"]]
        )
        profile_inference(
            (semantic, model),
            lambda: (
                model.forward_with_topology_embedding(
                    features, semantic_embedding, feature_graph
                )[0]
                if use_han_semantic_topology else
                model(features, last_graph, feature_graph)[0]
            ),
        )
        diagnostics = {
            "semantic_attention": (
                last_attention.tolist() if last_attention is not None else []
            ),
            "feature_knn_edges": int(feature_graph.num_edges()),
            "view_mode": config.view_mode,
            "topology_branch_active": topology_enabled,
            "topology_source": config.topology_source,
            "semantic_topology_filter": config.semantic_topology_filter,
            "checkpoint_reused": True,
            "checkpoint_source": str(Path(checkpoint_source).resolve()),
            "optimizer_steps": 0,
        }
        diagnostics.update(_dvcl_gate_summary(model))
        result = TrainingResult(
            metrics,
            [{"epoch": best_epoch, "checkpoint_reused": True}],
            best_epoch,
            best_epoch,
            diagnostics,
        )
        if _is_target_evasion(attack):
            def target_forward(adjs, record):
                attacked_topology = None
                if topology_enabled:
                    attacked_transitions = transition_edges(
                        features, adjs, clean.meta_paths, similarity
                    )
                    if use_han_semantic_topology:
                        attacked_views = purified_graphs(
                            attacked_transitions, config.thresholds
                        )
                        attacked_topology = semantic.encode(
                            features, attacked_views
                        )
                    else:
                        attacked_graph = dgl.from_scipy(semantic_graph(
                            attacked_transitions,
                            last_attention.to(target),
                            config.global_threshold,
                            apply_filter=apply_semantic_topology_filter,
                        )).to(target)
                        attacked_topology = model.topology_encoder(
                            features, attacked_graph
                        )
                return model.classify(attacked_topology, feature)[0]

            def target_diagnostic_forward(adjs, record):
                attacked_topology = None
                if topology_enabled:
                    attacked_transitions = transition_edges(
                        features, adjs, clean.meta_paths, similarity
                    )
                    if use_han_semantic_topology:
                        attacked_views = purified_graphs(
                            attacked_transitions, config.thresholds
                        )
                        attacked_topology = semantic.encode(
                            features, attacked_views
                        )
                    else:
                        attacked_graph = dgl.from_scipy(semantic_graph(
                            attacked_transitions,
                            last_attention.to(target),
                            config.global_threshold,
                            apply_filter=apply_semantic_topology_filter,
                        )).to(target)
                        attacked_topology = model.topology_encoder(
                            features, attacked_graph
                        )
                attacked_logits = model.classify(attacked_topology, feature)[0]
                return _dvcl_target_view_state(
                    model, attacked_logits, attacked_topology, feature,
                    int(record["target"]),
                )

            _evaluate_model_target_evasion(
                result, clean, split, attack, labels, test_logits, target_forward,
                checkpoint_path, "dvcl", config, train_seed,
                clean_view_state=clean_view_state,
                diagnostic_forward_for_adjs=target_diagnostic_forward,
            )
        return result
    stopper = LegacyEarlyStopping(patience)
    history = []
    last_graph = None
    last_attention = None
    stopped_epoch = 0

    for epoch in range(epochs):
        semantic.train()
        model.train()
        attention = None
        topology_graph = None
        semantic_embedding = None
        if topology_enabled:
            semantic_embedding = semantic.encode(features, views)
            semantic_logits = semantic.predict(semantic_embedding)
            attention = semantic.semantic_weights()
            if not use_han_semantic_topology:
                matrix = semantic_graph(
                    transitions,
                    attention,
                    config.global_threshold,
                    apply_filter=apply_semantic_topology_filter,
                )
                topology_graph = dgl.from_scipy(matrix).to(target)
        if use_han_semantic_topology:
            logits, topology, feature = model.forward_with_topology_embedding(
                features, semantic_embedding, feature_graph
            )
        else:
            logits, topology, feature = model(
                features, topology_graph, feature_graph
            )
        if topology_enabled:
            semantic_loss = loss_fn(
                semantic_logits[masks["train"]], labels[masks["train"]]
            )
        else:
            semantic_loss = logits.new_tensor(0.0)
        classification_loss = loss_fn(logits[masks["train"]], labels[masks["train"]])
        if config.view_mode == "both":
            contrastive_loss = cross_view_contrastive_loss(
                topology, feature, masks["train"], config.temperature
            )
        else:
            contrastive_loss = logits.new_tensor(0.0)
        reliability = model.reliability_losses(
            topology, feature, labels, masks["train"]
        )
        augmented_classification_loss = logits.new_tensor(0.0)
        augmented_auxiliary_loss = logits.new_tensor(0.0)
        augmented_route_loss = logits.new_tensor(0.0)
        if config.structure_augment_rate > 0:
            augmented_graph = perturb_topology_graph(
                topology_graph, config.structure_augment_rate
            )
            augmented_topology = model.topology_encoder(
                features, augmented_graph
            )
            augmented_logits = model.classify(augmented_topology, feature)[0]
            augmented_classification_loss = loss_fn(
                augmented_logits[masks["train"]], labels[masks["train"]]
            )
            augmented_reliability = model.reliability_losses(
                augmented_topology, feature, labels, masks["train"]
            )
            augmented_auxiliary_loss = augmented_reliability["auxiliary_loss"]
            augmented_route_loss = augmented_reliability["route_loss"]
        augmented_loss = (
            augmented_classification_loss
            + config.beta_aux * augmented_auxiliary_loss
            + config.lambda_route * augmented_route_loss
        )
        total_loss = (
            config.lambda_han * semantic_loss
            + classification_loss
            + config.lambda_dvcl * contrastive_loss
            + config.beta_aux * reliability["auxiliary_loss"]
            + config.lambda_route * reliability["route_loss"]
            + config.lambda_aug * augmented_loss
        )
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        semantic.eval()
        model.eval()
        with torch.no_grad():
            if use_han_semantic_topology:
                validation_topology = semantic.encode(features, views)
                validation_logits, _, _ = (
                    model.forward_with_topology_embedding(
                        features, validation_topology, feature_graph
                    )
                )
            else:
                validation_logits, _, _ = model(
                    features, topology_graph, feature_graph
                )
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
            "auxiliary_loss": float(reliability["auxiliary_loss"].detach()),
            "route_loss": float(reliability["route_loss"].detach()),
            "augmented_classification_loss": float(
                augmented_classification_loss.detach()
            ),
            "augmented_auxiliary_loss": float(
                augmented_auxiliary_loss.detach()
            ),
            "augmented_route_loss": float(augmented_route_loss.detach()),
            "augmented_loss": float(augmented_loss.detach()),
            "total_loss": float(total_loss.detach()),
            "val_loss": float(validation_loss),
            **{f"val_{key}": value for key, value in validation.items()},
        })
        if topology_enabled:
            last_graph = topology_graph
            last_attention = attention.detach().cpu()
        stopped_epoch = epoch
        if stopper.step(float(validation_loss), validation["accuracy"], holder, epoch):
            break

    stopper.restore(holder)
    semantic_embedding = None
    if not config.legacy_checkpoint_semantics and topology_enabled:
        semantic.eval()
        with torch.no_grad():
            semantic_embedding = semantic.encode(features, views)
            last_attention = semantic.semantic_weights().detach().cpu()
        if not use_han_semantic_topology:
            last_graph = dgl.from_scipy(
                semantic_graph(
                    transitions,
                    last_attention.to(target),
                    config.global_threshold,
                    apply_filter=apply_semantic_topology_filter,
                )
            ).to(target)
    model.eval()
    with torch.no_grad():
        if use_han_semantic_topology:
            test_logits, topology, feature = (
                model.forward_with_topology_embedding(
                    features, semantic_embedding, feature_graph
                )
            )
        else:
            test_logits, topology, feature = model(
                features, last_graph, feature_graph
            )
        clean_view_state = _dvcl_view_state(model, test_logits, topology, feature)
    metrics = classification_metrics(test_logits[masks["test"]], labels[masks["test"]])
    save_checkpoint(checkpoint_path, holder, config, stopper.best_epoch)
    profile_inference(
        (semantic, model),
        lambda: (
            model.forward_with_topology_embedding(
                features, semantic_embedding, feature_graph
            )[0]
            if use_han_semantic_topology else
            model(features, last_graph, feature_graph)[0]
        ),
    )
    diagnostics = {
        "semantic_attention": (
            last_attention.tolist() if last_attention is not None else []
        ),
        "feature_knn_edges": int(feature_graph.num_edges()),
        "view_mode": config.view_mode,
        "topology_branch_active": topology_enabled,
        "topology_source": config.topology_source,
        "semantic_topology_filter": config.semantic_topology_filter,
    }
    diagnostics.update(_dvcl_gate_summary(model))
    target_forward = None
    if _is_target_evasion(attack):
        def target_forward(adjs, record):
            topology = None
            if topology_enabled:
                attacked_transitions = transition_edges(
                    features, adjs, clean.meta_paths, similarity
                )
                if use_han_semantic_topology:
                    attacked_views = purified_graphs(
                        attacked_transitions, config.thresholds
                    )
                    topology = semantic.encode(features, attacked_views)
                else:
                    attacked_graph = dgl.from_scipy(semantic_graph(
                        attacked_transitions,
                        last_attention.to(target),
                        config.global_threshold,
                        apply_filter=apply_semantic_topology_filter,
                    )).to(target)
                    topology = model.topology_encoder(features, attacked_graph)
            return model.classify(topology, feature)[0]

        def target_diagnostic_forward(adjs, record):
            attacked_topology = None
            if topology_enabled:
                attacked_transitions = transition_edges(
                    features, adjs, clean.meta_paths, similarity
                )
                if use_han_semantic_topology:
                    attacked_views = purified_graphs(
                        attacked_transitions, config.thresholds
                    )
                    attacked_topology = semantic.encode(
                        features, attacked_views
                    )
                else:
                    attacked_graph = dgl.from_scipy(semantic_graph(
                        attacked_transitions,
                        last_attention.to(target),
                        config.global_threshold,
                        apply_filter=apply_semantic_topology_filter,
                    )).to(target)
                    attacked_topology = model.topology_encoder(
                        features, attacked_graph
                    )
            attacked_logits = model.classify(attacked_topology, feature)[0]
            return _dvcl_target_view_state(
                model, attacked_logits, attacked_topology, feature,
                int(record["target"]),
            )

    result = TrainingResult(metrics, history, stopper.best_epoch, stopped_epoch, diagnostics)
    if _is_target_evasion(attack):
        _evaluate_model_target_evasion(
            result, clean, split, attack, labels, test_logits, target_forward,
            checkpoint_path, "dvcl", config, train_seed,
            clean_view_state=clean_view_state,
            diagnostic_forward_for_adjs=target_diagnostic_forward,
        )
    return result


def _train_supervised(
    model,
    forward,
    labels,
    masks,
    config,
    epochs,
    patience,
    checkpoint_path,
    checkpoint_source=None,
):
    if checkpoint_source is not None:
        best_epoch = restore_checkpoint(
            checkpoint_source, checkpoint_path, model, config
        )
        model.eval()
        with torch.no_grad():
            test_logits = forward()
        metrics = classification_metrics(
            test_logits[masks["test"]], labels[masks["test"]]
        )
        profile_inference(model, forward)
        return TrainingResult(
            metrics,
            [{"epoch": best_epoch, "checkpoint_reused": True}],
            best_epoch,
            best_epoch,
            {
                "checkpoint_reused": True,
                "checkpoint_source": str(Path(checkpoint_source).resolve()),
                "optimizer_steps": 0,
            },
        )
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
    profile_inference(model, forward)
    return TrainingResult(metrics, history, stopper.best_epoch, stopped_epoch)


def _selected_adjs(
    clean: CleanGraphArtifact,
    attack: Optional[AttackArtifact],
) -> Mapping:
    if attack is None:
        return clean.hete_adjs
    if attack.dataset != clean.dataset or attack.clean_version != clean.version:
        raise ValueError("Attack artifact does not belong to the selected clean graph")
    if _is_target_evasion(attack):
        return clean.hete_adjs
    return attack.perturbed_hete_adjs


def _is_target_evasion(attack: Optional[AttackArtifact]) -> bool:
    return bool(
        attack is not None
        and attack.threat_model == "evasion"
        and attack.scope == "target"
    )


def _is_adaptive_request(attack: Optional[AttackArtifact]) -> bool:
    return bool(
        _is_target_evasion(attack)
        and attack.adaptive
        and attack.attack_name in {"adaptive_query", "dvcl_adaptive_query"}
        and attack.provenance.get("request_only") is True
        and attack.target_nodes is not None
    )


def _evaluate_model_target_evasion(
    result: TrainingResult,
    clean: CleanGraphArtifact,
    split: SplitArtifact,
    attack: AttackArtifact,
    labels: torch.Tensor,
    clean_logits: torch.Tensor,
    forward_for_adjs,
    checkpoint_path: Path,
    victim_model: str,
    victim_config,
    train_seed: int,
    clean_view_state=None,
    diagnostic_forward_for_adjs=None,
) -> None:
    effective_attack = attack
    if _is_adaptive_request(attack):
        max_additions = int(attack.provenance.get("candidate_additions", 16))
        max_deletions = int(attack.provenance.get("candidate_deletions", 16))
        records, adaptive_diagnostics = greedy_query_target_changes(
            clean=clean,
            targets=attack.target_nodes.tolist(),
            labels=labels,
            clean_logits=clean_logits,
            forward_for_adjs=forward_for_adjs,
            budget=int(attack.attack_rate),
            seed=attack.seed,
            max_additions=max_additions,
            max_deletions=max_deletions,
        )
        checkpoint_sha256 = file_sha256(checkpoint_path)
        search_budget = int(attack.attack_rate)
        budgets = [
            budget for budget in (1, 3, 5) if budget <= search_budget
        ]
        if search_budget not in budgets:
            budgets.append(search_budget)
        budget_evaluations = {}
        full_test_clean_metrics = dict(result.metrics)
        for budget in sorted(set(budgets)):
            budget_records = records_at_budget(records, budget)
            budget_attack = _materialize_adaptive_attack(
                clean, split, attack, budget_records, budget, search_budget,
                adaptive_diagnostics, checkpoint_path, checkpoint_sha256,
                victim_model, victim_config, train_seed,
            )
            attack_path, verification_path = _adaptive_attack_paths(
                checkpoint_path, budget, search_budget
            )
            save_attack_artifact(budget_attack, attack_path)
            report = verify_attack(clean, split, budget_attack)
            save_json(report, verification_path)
            if not report["ok"]:
                raise RuntimeError(
                    "Generated adaptive attack failed verification: "
                    + "; ".join(report["issues"])
                )
            metrics, evaluation = _target_evasion_values(
                clean, budget_attack, labels, clean_logits, forward_for_adjs,
                full_test_clean_metrics,
                clean_view_state=clean_view_state,
                diagnostic_forward_for_adjs=diagnostic_forward_for_adjs,
            )
            change_stats = _target_change_stats(budget_records, budget)
            budget_evaluations[str(budget)] = {
                "metrics": metrics,
                "diagnostics": evaluation,
                "adaptive_attack": {
                    **change_stats,
                    "queries": record_query_count(budget_records),
                    "candidate_additions": max_additions,
                    "candidate_deletions": max_deletions,
                    "candidate_pool_sha256": adaptive_diagnostics[
                        "candidate_pool_sha256"
                    ],
                    "path": str(attack_path),
                    "sha256": file_sha256(attack_path),
                },
            }
            if budget == search_budget:
                effective_attack = budget_attack
                result.metrics = metrics
                result.diagnostics.update(evaluation)
                result.diagnostics["adaptive_attack"] = {
                    **adaptive_diagnostics,
                    "path": str(attack_path),
                    "sha256": file_sha256(attack_path),
                    "victim_model": victim_model,
                    "victim_checkpoint_sha256": checkpoint_sha256,
                }
        result.diagnostics["budget_evaluations"] = budget_evaluations
        result.diagnostics["search_budget"] = search_budget
        return
    _evaluate_target_evasion(
        result, clean, effective_attack, labels, clean_logits, forward_for_adjs,
        clean_view_state=clean_view_state,
        diagnostic_forward_for_adjs=diagnostic_forward_for_adjs,
    )


def _materialize_adaptive_attack(
    clean, split, request, records, budget, search_budget, adaptive_diagnostics,
    checkpoint_path, checkpoint_sha256, victim_model, victim_config, train_seed,
):
    query_count = record_query_count(records)
    value = build_attack_artifact(
        clean=clean,
        split=split,
        attack_name=request.attack_name,
        attack_rate=budget,
        seed=request.seed,
        perturbed=clean.hete_adjs,
        target_nodes=request.target_nodes,
        source=str(Path(checkpoint_path).resolve()),
        source_sha256=checkpoint_sha256,
        provenance={
            **request.provenance,
            "request_only": False,
            "generator": "dvcl_bench.adaptive.greedy_query_target_changes",
            "victim_model": victim_model,
            "victim_train_seed": int(train_seed),
            "victim_model_config": asdict(victim_config),
            "victim_checkpoint": str(Path(checkpoint_path).resolve()),
            "victim_checkpoint_sha256": checkpoint_sha256,
            "search_budget": int(search_budget),
            "evaluation_budget": int(budget),
            "queries_for_evaluation_budget": query_count,
            **adaptive_diagnostics,
        },
        threat_model="evasion",
        scope="target",
        adaptive=True,
        target_changes=records,
    )
    value.stats["_target"] = _target_change_stats(records, budget)
    value.provenance["queries_for_evaluation_budget"] = query_count
    return value


def _adaptive_attack_paths(checkpoint_path, budget, search_budget):
    checkpoint_path = Path(checkpoint_path)
    if budget == search_budget:
        return (
            checkpoint_path.with_name("adaptive_attack.pt"),
            checkpoint_path.with_name("adaptive_attack_verification.json"),
        )
    root = checkpoint_path.parent / "adaptive_attacks" / f"rate_{budget}"
    return root / "attack.pt", root / "verification.json"


def _target_change_stats(records, budget):
    counts = [
        len(record.get("deleted", [])) + len(record.get("added", []))
        for record in records
    ]
    return {
        "targets": len(records),
        "budget_per_target": int(budget),
        "total_changes": int(sum(counts)),
        "min_changes": int(min(counts, default=0)),
        "max_changes": int(max(counts, default=0)),
        "mean_changes": float(np.mean(counts)) if counts else 0.0,
        "budget_utilization": (
            float(sum(counts) / (len(counts) * int(budget)))
            if counts and int(budget) else 0.0
        ),
    }


def _evaluate_target_evasion(
    result: TrainingResult,
    clean: CleanGraphArtifact,
    attack: AttackArtifact,
    labels: torch.Tensor,
    clean_logits: torch.Tensor,
    forward_for_adjs,
    clean_view_state=None,
    diagnostic_forward_for_adjs=None,
) -> None:
    metrics, diagnostics = _target_evasion_values(
        clean, attack, labels, clean_logits, forward_for_adjs,
        dict(result.metrics),
        clean_view_state=clean_view_state,
        diagnostic_forward_for_adjs=diagnostic_forward_for_adjs,
    )
    result.metrics = metrics
    result.diagnostics.update(diagnostics)


def _target_evasion_values(
    clean: CleanGraphArtifact,
    attack: AttackArtifact,
    labels: torch.Tensor,
    clean_logits: torch.Tensor,
    forward_for_adjs,
    full_test_clean_metrics,
    clean_view_state=None,
    diagnostic_forward_for_adjs=None,
):
    if attack.target_nodes is None or not attack.target_changes:
        raise ValueError("Target evasion requires target nodes and per-target changes")
    targets = attack.target_nodes.long().to(labels.device)
    clean_target_metrics = classification_metrics(
        clean_logits[targets], labels[targets]
    )
    attacked_logits = []
    attacked_labels = []
    attacked_view_states = []
    with torch.no_grad():
        for record in attack.target_changes:
            target = int(record["target"])
            attacked_adjs = apply_target_change(clean, record)
            if diagnostic_forward_for_adjs is None:
                logits = forward_for_adjs(attacked_adjs, record)
                target_logits = logits[target]
            else:
                view_state = diagnostic_forward_for_adjs(attacked_adjs, record)
                target_logits = view_state["fused_logits"]
                attacked_view_states.append(view_state)
            attacked_logits.append(target_logits.unsqueeze(0))
            attacked_labels.append(labels[target].unsqueeze(0))
    target_logits = torch.cat(attacked_logits, dim=0)
    target_labels = torch.cat(attacked_labels, dim=0)
    clean_target_logits = clean_logits[targets]
    clean_prediction = clean_target_logits.argmax(dim=1)
    attacked_prediction = target_logits.argmax(dim=1)
    clean_correct = clean_prediction.eq(target_labels)
    attack_success = clean_correct & attacked_prediction.ne(target_labels)
    metrics = classification_metrics(target_logits, target_labels)
    diagnostics = {
        "evaluation_scope": "target",
        "threat_model": "evasion",
        "target_count": len(attack.target_changes),
        "clean_target_metrics": clean_target_metrics,
        "full_test_clean_metrics": full_test_clean_metrics,
        "target_accuracy_drop": clean_target_metrics["accuracy"] - metrics["accuracy"],
        "clean_correct_target_count": int(clean_correct.sum()),
        "attack_success_count": int(attack_success.sum()),
        "attack_success_rate": (
            float(attack_success.sum() / clean_correct.sum())
            if bool(clean_correct.any()) else 0.0
        ),
    }
    if clean_view_state is not None and attacked_view_states:
        diagnostics["view_diagnostics"] = target_view_diagnostics(
            clean_view_state, attacked_view_states, labels
        )
    return metrics, diagnostics


def _dvcl_view_state(model, fused_logits, topology, feature):
    result = {"fused_logits": fused_logits}
    if topology is not None:
        result["topology_embedding"] = topology
    if feature is not None:
        result["feature_embedding"] = feature
    result.update(model.diagnostic_views(topology, feature))
    return result


def _dvcl_gate_summary(model):
    if model.last_gate_weight is None:
        return {}
    weights = model.last_gate_weight.detach().flatten()
    return {
        "gate_mean": float(weights.mean()),
        "gate_std": float(weights.std(unbiased=False)),
        "gate_min": float(weights.min()),
        "gate_max": float(weights.max()),
        "gate_topology_fraction": float((weights >= 0.9).float().mean()),
        "gate_feature_fraction": float((weights <= 0.1).float().mean()),
    }


def _dvcl_target_view_state(model, fused_logits, topology, feature, target):
    values = _dvcl_view_state(model, fused_logits, topology, feature)
    return {
        "target": int(target),
        **{
            key: value[target].detach()
            for key, value in values.items()
            if isinstance(value, torch.Tensor)
        },
    }


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
