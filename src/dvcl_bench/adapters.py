"""Native HSeCo and DVCL training adapters over frozen artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
from torch import nn

from .adaptive import greedy_query_target_changes
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
)
from .models.semantic import (
    NodeLevelAggregator,
    SemanticHAN,
    perturb_matrix,
    purified_graphs,
    semantic_graph,
    transition_edges,
)
from .models.baselines import HeteroGuard, HeteroSAGE
from .models.rohe import RoHe
from .models.fastrohgcn import FastRoHGCN
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
    if _is_target_evasion(attack):
        model.eval()
        with torch.no_grad():
            clean_logits = model(features, views)
        _evaluate_target_evasion(
            result, clean, attack, labels, clean_logits,
            lambda adjs, record: model(features, [
                dgl.from_scipy(meta_path_adjacency(adjs, path)).to(target)
                for path in clean.meta_paths
            ]),
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
    if _is_target_evasion(attack):
        model.eval()
        with torch.no_grad():
            clean_logits = model(graph, node_features)[clean.predict_ntype]

        def target_forward(adjs, record):
            attacked_graph = hete_adjs_to_dgl(
                adjs, clean.canonical_etypes, clean.node_counts
            ).to(target)
            return model(attacked_graph, node_features)[clean.predict_ntype]

        _evaluate_target_evasion(
            result, clean, attack, labels, clean_logits, target_forward
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
        _evaluate_target_evasion(
            result, clean, attack, labels, clean_logits,
            lambda adjs, record: model(node_features, edge_indices(adjs))[clean.predict_ntype],
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
    )
    result.diagnostics.update({
        "top_t": config.top_t,
        "semantic_attention": model.semantic_weights().detach().cpu().tolist(),
    })
    if _is_target_evasion(attack):
        model.eval()
        with torch.no_grad():
            clean_logits = model(features, selected_transitions)
        _evaluate_target_evasion(
            result, clean, attack, labels, clean_logits,
            lambda adjs, record: model(features, transitions(adjs)),
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

        _evaluate_target_evasion(
            result, clean, attack, labels, clean_logits, target_forward
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
    result = TrainingResult(
        metrics, history, stopper.best_epoch, stopped_epoch,
        {"semantic_attention": last_attention.tolist()},
    )
    if _is_target_evasion(attack):
        def target_forward(adjs, record):
            attacked_transitions = transition_edges(features, adjs, clean.meta_paths)
            attacked_graph = dgl.from_scipy(semantic_graph(
                attacked_transitions, last_attention.to(target), config.global_threshold
            )).to(target)
            return node_model(features, attacked_graph)

        _evaluate_target_evasion(
            result, clean, attack, labels, test_logits, target_forward
        )
    return result


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
    effective_attack = attack
    target_forward = None
    if _is_target_evasion(attack):
        def target_forward(adjs, record):
            attacked_transitions = transition_edges(features, adjs, clean.meta_paths)
            attacked_graph = dgl.from_scipy(semantic_graph(
                attacked_transitions, last_attention.to(target), config.global_threshold
            )).to(target)
            topology = None
            if config.view_mode in {"both", "both_nocl", "topo"}:
                topology = model.topology_encoder(features, attacked_graph)
            return model.classify(topology, feature)[0]

    if _is_dvcl_adaptive_request(attack):
        semantic.eval()
        model.eval()
        max_additions = int(attack.provenance.get("candidate_additions", 16))
        max_deletions = int(attack.provenance.get("candidate_deletions", 16))
        records, adaptive_diagnostics = greedy_query_target_changes(
            clean=clean,
            targets=attack.target_nodes.tolist(),
            labels=labels,
            clean_logits=test_logits,
            forward_for_adjs=target_forward,
            budget=int(attack.attack_rate),
            seed=attack.seed,
            max_additions=max_additions,
            max_deletions=max_deletions,
        )
        checkpoint_sha256 = file_sha256(checkpoint_path)
        effective_attack = build_attack_artifact(
            clean=clean,
            split=split,
            attack_name=attack.attack_name,
            attack_rate=attack.attack_rate,
            seed=attack.seed,
            perturbed=clean.hete_adjs,
            target_nodes=attack.target_nodes,
            source=str(Path(checkpoint_path).resolve()),
            source_sha256=checkpoint_sha256,
            provenance={
                **attack.provenance,
                "request_only": False,
                "generator": "dvcl_bench.adaptive.greedy_query_target_changes",
                "victim_model": "dvcl",
                "victim_train_seed": int(train_seed),
                "victim_model_config": asdict(config),
                "victim_checkpoint": str(Path(checkpoint_path).resolve()),
                "victim_checkpoint_sha256": checkpoint_sha256,
                **adaptive_diagnostics,
            },
            threat_model="evasion",
            scope="target",
            adaptive=True,
            target_changes=records,
        )
        effective_attack.stats["_target"] = _target_change_stats(
            records, effective_attack.attack_rate
        )
        adaptive_path = Path(checkpoint_path).with_name("adaptive_attack.pt")
        save_attack_artifact(effective_attack, adaptive_path)
        adaptive_report = verify_attack(clean, split, effective_attack)
        save_json(
            adaptive_report,
            Path(checkpoint_path).with_name("adaptive_attack_verification.json"),
        )
        if not adaptive_report["ok"]:
            raise RuntimeError(
                "Generated adaptive attack failed verification: "
                + "; ".join(adaptive_report["issues"])
            )
        diagnostics["adaptive_attack"] = {
            **adaptive_diagnostics,
            "path": str(adaptive_path),
            "sha256": file_sha256(adaptive_path),
            "victim_checkpoint_sha256": checkpoint_sha256,
        }

    result = TrainingResult(metrics, history, stopper.best_epoch, stopped_epoch, diagnostics)
    if _is_target_evasion(effective_attack):

        _evaluate_target_evasion(
            result, clean, effective_attack, labels, test_logits, target_forward
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
    if _is_target_evasion(attack):
        return clean.hete_adjs
    return attack.perturbed_hete_adjs


def _is_target_evasion(attack: Optional[AttackArtifact]) -> bool:
    return bool(
        attack is not None
        and attack.threat_model == "evasion"
        and attack.scope == "target"
    )


def _is_dvcl_adaptive_request(attack: Optional[AttackArtifact]) -> bool:
    return bool(
        _is_target_evasion(attack)
        and attack.adaptive
        and attack.attack_name == "dvcl_adaptive_query"
        and attack.provenance.get("request_only") is True
        and attack.target_nodes is not None
    )


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
    }


def _evaluate_target_evasion(
    result: TrainingResult,
    clean: CleanGraphArtifact,
    attack: AttackArtifact,
    labels: torch.Tensor,
    clean_logits: torch.Tensor,
    forward_for_adjs,
) -> None:
    if attack.target_nodes is None or not attack.target_changes:
        raise ValueError("Target evasion requires target nodes and per-target changes")
    targets = attack.target_nodes.long().to(labels.device)
    clean_target_metrics = classification_metrics(
        clean_logits[targets], labels[targets]
    )
    attacked_logits = []
    attacked_labels = []
    with torch.no_grad():
        for record in attack.target_changes:
            target = int(record["target"])
            logits = forward_for_adjs(apply_target_change(clean, record), record)
            attacked_logits.append(logits[target].unsqueeze(0))
            attacked_labels.append(labels[target].unsqueeze(0))
    target_logits = torch.cat(attacked_logits, dim=0)
    target_labels = torch.cat(attacked_labels, dim=0)
    full_test_clean_metrics = dict(result.metrics)
    result.metrics = classification_metrics(target_logits, target_labels)
    result.diagnostics.update({
        "evaluation_scope": "target",
        "threat_model": "evasion",
        "target_count": len(attack.target_changes),
        "clean_target_metrics": clean_target_metrics,
        "full_test_clean_metrics": full_test_clean_metrics,
        "target_accuracy_drop": clean_target_metrics["accuracy"] - result.metrics["accuracy"],
    })


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
