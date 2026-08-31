from __future__ import annotations

import copy
import io
import types
from contextlib import redirect_stdout
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from .adapters import (
    _device,
    _evaluate_model_target_evasion,
    _inputs,
    _is_target_evasion,
    _selected_adjs,
    _train_supervised,
)
from .artifacts import AttackArtifact, CleanGraphArtifact, SplitArtifact
from .graph_adapter import hete_adjs_to_dgl
from .profiling import profile_inference
from .integrations.openhgnn import (
    OPENHGNN_MODEL_HASHES,
    OPENHGNN_REVISION,
    OPENHGNN_VERSION,
    require_openhgnn_model,
)
from .training import (
    HGTTrainConfig,
    HeCoTrainConfig,
    LegacyEarlyStopping,
    MAGNNTrainConfig,
    SimpleHGNTrainConfig,
    TrainingResult,
    classification_metrics,
    restore_checkpoint,
    save_checkpoint,
    set_random_seed,
)


OPENHGNN_MODELS = {
    "hgt": HGTTrainConfig,
    "magnn": MAGNNTrainConfig,
    "heco": HeCoTrainConfig,
    "simplehgn": SimpleHGNTrainConfig,
}


def build_openhgnn_config(name: str, values: Dict[str, Any]):
    try:
        config_type = OPENHGNN_MODELS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported OpenHGNN model: {name}") from exc
    allowed = {item.name for item in fields(config_type)}
    unknown = set(values) - allowed - {"variant"}
    if unknown:
        raise ValueError(f"Unknown {name} configuration fields: {sorted(unknown)}")
    return config_type(**{key: value for key, value in values.items() if key in allowed})


class OpenHGNNClassifier(nn.Module):
    def __init__(self, input_feature, encoder, predict_ntype: str):
        super().__init__()
        self.input_feature = input_feature
        self.encoder = encoder
        self.predict_ntype = predict_ntype

    def forward(self, graph):
        return self.encoder(graph, self.input_feature())[self.predict_ntype]


class HeCoPipeline(nn.Module):
    def __init__(self, input_feature, encoder, classifier):
        super().__init__()
        self.input_feature = input_feature
        self.encoder = encoder
        self.classifier = classifier

    def contrastive_loss(self, graph, positive):
        with redirect_stdout(io.StringIO()):
            return self.encoder(graph, self.input_feature(), positive)

    def embeddings(self, graph):
        with redirect_stdout(io.StringIO()):
            return self.encoder.get_embeds(graph, self.input_feature())

    def forward(self, graph):
        return self.classifier(self.embeddings(graph))


def train_openhgnn(
    clean: CleanGraphArtifact,
    split: SplitArtifact,
    attack: Optional[AttackArtifact],
    config,
    train_seed: int,
    epochs: int,
    patience: int,
    device: str,
    checkpoint_path: Path,
    model_name: str,
    checkpoint_source: Optional[Path] = None,
) -> TrainingResult:
    set_random_seed(train_seed)
    target = _device(device)
    features, labels, masks = _inputs(clean, split, target)
    selected_adjs = _selected_adjs(clean, attack)
    if model_name == "heco":
        return _train_heco(
            clean, split, attack, config, features, labels, masks, selected_adjs,
            epochs, patience, target, checkpoint_path, train_seed,
            checkpoint_source,
        )

    graph = _graph(clean, selected_adjs, target)
    input_feature = _input_feature(clean, features, config.hidden_dim, target)
    encoder, context = _build_supervised_encoder(
        model_name, clean, selected_adjs, graph, config, target, train_seed
    )
    model = OpenHGNNClassifier(input_feature, encoder, clean.predict_ntype).to(target)
    result = _train_supervised(
        model=model,
        forward=lambda: model(graph),
        labels=labels,
        masks=masks,
        config=config,
        epochs=epochs,
        patience=patience,
        checkpoint_path=checkpoint_path,
        checkpoint_source=checkpoint_source,
    )
    result.diagnostics.update(_diagnostics(model_name))
    result.diagnostics.update({
        "heterogeneous_edges": int(graph.num_edges()),
        "non_target_feature_fill": "trainable OpenHGNN embedding",
    })
    if model_name == "magnn":
        result.diagnostics["metapath_instances"] = {
            key: int(value.shape[0]) for key, value in context.items()
        }
        result.diagnostics["adapter_fixes"] = [
            "propagate hidden-layer outputs",
            "preserve inter-metapath attention gradients and device",
        ]
        result.diagnostics["instances_per_node"] = config.instances_per_node
    if _is_target_evasion(attack):
        model.eval()
        with torch.no_grad():
            clean_logits = model(graph)

        def target_forward(adjs, record):
            attacked_graph = _graph(clean, adjs, target)
            if model_name != "magnn":
                return model(attacked_graph)
            attacked_target = int(record["target"])
            replacement = _replace_magnn_target_instances(
                clean, adjs, context, attacked_target,
                config.instances_per_node, train_seed,
            )
            previous = model.encoder.metapath_idx_dict
            model.encoder.metapath_idx_dict = replacement
            try:
                return model(attacked_graph)
            finally:
                model.encoder.metapath_idx_dict = previous

        _evaluate_model_target_evasion(
            result, clean, split, attack, labels, clean_logits, target_forward,
            checkpoint_path, model_name, config, train_seed,
        )
    return result


def _build_supervised_encoder(name, clean, adjs, graph, config, device, train_seed):
    model_type = require_openhgnn_model(name)
    if name == "hgt":
        if config.hidden_dim % config.num_heads:
            raise ValueError("HGT hidden_dim must be divisible by num_heads")
        return model_type(
            config.hidden_dim, clean.num_classes, config.num_heads,
            len(graph.etypes), graph.ntypes, config.num_layers,
            config.dropout, config.norm,
        ).to(device), {}
    if name == "simplehgn":
        if config.hidden_dim % config.num_heads:
            raise ValueError("SimpleHGN hidden_dim must be divisible by num_heads")
        heads = [config.num_heads] * config.num_layers + [1]
        return model_type(
            config.edge_dim, len(graph.etypes), [config.hidden_dim],
            config.hidden_dim // config.num_heads, clean.num_classes,
            config.num_layers, heads, config.dropout, config.slope,
            True, config.beta, graph.ntypes,
        ).to(device), {}
    if name == "magnn":
        if config.hidden_dim % config.num_heads:
            raise ValueError("MAGNN hidden_dim must be divisible by num_heads")
        metapaths = _magnn_metapaths(clean)
        edge_types = [
            f"{source}-{target}" for source, _, target in clean.canonical_etypes
        ]
        instances = _magnn_instances(
            clean, adjs, config.instances_per_node, train_seed
        )
        model = model_type(
            ntypes=graph.ntypes,
            h_feats=config.hidden_dim // config.num_heads,
            inter_attn_feats=config.inter_attention_dim,
            num_heads=config.num_heads,
            num_classes=clean.num_classes,
            num_layers=config.num_layers,
            metapath_list=metapaths,
            edge_type_list=edge_types,
            dropout_rate=config.dropout,
            metapath_idx_dict=instances,
            encoder_type=config.encoder_type,
        ).to(device)
        _patch_magnn(model)
        return model, instances
    raise ValueError(f"Unsupported supervised OpenHGNN model: {name}")


def _train_heco(
    clean, split, attack, config, features, labels, masks, selected_adjs,
    epochs, patience, device, checkpoint_path, train_seed, checkpoint_source,
):
    from openhgnn.models.HeCo import LogReg

    graph = _graph(clean, selected_adjs, device)
    input_feature = _input_feature(clean, features, config.hidden_dim, device)
    model_type = require_openhgnn_model("heco")
    meta_paths = _heco_metapaths(clean)
    schema = [
        canonical for canonical in clean.canonical_etypes
        if canonical[2] == clean.predict_ntype
    ]
    sample_rate = {
        source: config.schema_sample_size for source, _, _ in schema
    }
    encoder = model_type(
        meta_paths, schema, clean.predict_ntype, config.hidden_dim,
        config.feature_dropout, config.attention_dropout, sample_rate,
        config.temperature, config.contrastive_balance,
    ).to(device)
    pipeline = HeCoPipeline(
        input_feature, encoder, LogReg(config.hidden_dim, clean.num_classes).to(device)
    ).to(device)
    if checkpoint_source is not None:
        best_epoch = restore_checkpoint(
            checkpoint_source, checkpoint_path, pipeline, config
        )
        pipeline.eval()
        with torch.no_grad():
            clean_logits = pipeline(graph)
        metrics = classification_metrics(
            clean_logits[masks["test"]], labels[masks["test"]]
        )
        profile_inference(pipeline, lambda: pipeline(graph))
        result = TrainingResult(
            metrics,
            [{"epoch": best_epoch, "checkpoint_reused": True}],
            best_epoch,
            best_epoch,
            {
                **_diagnostics("heco"),
                "checkpoint_reused": True,
                "checkpoint_source": str(Path(checkpoint_source).resolve()),
                "optimizer_steps": 0,
            },
        )
        if _is_target_evasion(attack):
            def target_forward(adjs, record):
                attacked_graph = _graph(clean, adjs, device)
                return pipeline(attacked_graph)

            _evaluate_model_target_evasion(
                result, clean, split, attack, labels, clean_logits,
                target_forward, checkpoint_path, "heco", config, train_seed,
            )
        return result
    positive = _heco_positive(selected_adjs, clean.meta_paths, config.positive_topk, device)
    optimizer = torch.optim.Adam(
        list(pipeline.input_feature.parameters()) + list(pipeline.encoder.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history = []
    best_loss = None
    best_state = None
    pretrain_best_epoch = -1
    pretrain_stopped_epoch = -1
    counter = 0
    for epoch in range(epochs):
        pipeline.train()
        loss = pipeline.contrastive_loss(graph, positive)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        history.append(_history_row("pretrain", epoch, contrastive_loss=value))
        pretrain_stopped_epoch = epoch
        if best_loss is None or value < best_loss:
            best_loss = value
            best_state = copy.deepcopy(pipeline.state_dict())
            pretrain_best_epoch = epoch
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    if best_state is None:
        raise RuntimeError("HeCo pretraining did not produce a checkpoint")
    pipeline.load_state_dict(best_state)
    pipeline.eval()
    embeddings = pipeline.embeddings(graph)
    classifier_optimizer = torch.optim.Adam(
        pipeline.classifier.parameters(),
        lr=config.evaluation_learning_rate,
        weight_decay=config.evaluation_weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    stopper = LegacyEarlyStopping(patience)
    stopped_epoch = -1
    for epoch in range(epochs):
        pipeline.classifier.train()
        logits = pipeline.classifier(embeddings)
        classification_loss = loss_fn(logits[masks["train"]], labels[masks["train"]])
        classifier_optimizer.zero_grad()
        classification_loss.backward()
        classifier_optimizer.step()
        pipeline.classifier.eval()
        with torch.no_grad():
            validation_logits = pipeline.classifier(embeddings)
            validation_loss = loss_fn(
                validation_logits[masks["val"]], labels[masks["val"]]
            )
            validation = classification_metrics(
                validation_logits[masks["val"]], labels[masks["val"]]
            )
        history.append(_history_row(
            "linear_eval", epoch,
            classification_loss=float(classification_loss.detach()),
            val_loss=float(validation_loss),
            validation=validation,
        ))
        stopped_epoch = epoch
        if stopper.step(
            float(validation_loss), validation["accuracy"], pipeline.classifier, epoch
        ):
            break
    stopper.restore(pipeline.classifier)
    pipeline.eval()
    with torch.no_grad():
        clean_logits = pipeline.classifier(embeddings)
    metrics = classification_metrics(clean_logits[masks["test"]], labels[masks["test"]])
    save_checkpoint(checkpoint_path, pipeline, config, stopper.best_epoch)
    profile_inference(pipeline, lambda: pipeline(graph))
    result = TrainingResult(metrics, history, stopper.best_epoch, stopped_epoch)
    result.diagnostics.update(_diagnostics("heco"))
    result.diagnostics.update({
        "pretrain_best_epoch": pretrain_best_epoch,
        "pretrain_stopped_epoch": pretrain_stopped_epoch,
        "positive_topk": config.positive_topk,
        "schema_sample_size": config.schema_sample_size,
        "evaluation_protocol": "single seeded linear classifier with validation early stopping",
    })
    if _is_target_evasion(attack):
        def target_forward(adjs, record):
            attacked_graph = _graph(clean, adjs, device)
            return pipeline.classifier(pipeline.embeddings(attacked_graph))

        _evaluate_model_target_evasion(
            result, clean, split, attack, labels, clean_logits, target_forward,
            checkpoint_path, "heco", config, train_seed,
        )
    return result


def _input_feature(clean, features, hidden_dim, device):
    from openhgnn.layers.HeteroLinear import HeteroFeature

    return HeteroFeature(
        {clean.predict_ntype: features}, clean.node_counts, hidden_dim
    ).to(device)


def _graph(clean, adjs, device):
    return hete_adjs_to_dgl(adjs, clean.canonical_etypes, clean.node_counts).to(device)


def _magnn_metapaths(clean):
    canonical = {relation: (source, target) for source, relation, target in clean.canonical_etypes}
    values = []
    for path in clean.meta_paths:
        source, target = canonical[path[0]]
        nodes = [source, target]
        for relation in path[1:]:
            next_source, next_target = canonical[relation]
            if nodes[-1] != next_source:
                raise ValueError(f"Disconnected MAGNN meta-path: {path}")
            nodes.append(next_target)
        values.append("-".join(nodes))
    return values


def _magnn_instances(clean, adjs, instances_per_node, seed):
    result = {}
    for name, path in zip(_magnn_metapaths(clean), clean.meta_paths):
        first = adjs[path[0]].tocsr()
        rows, columns = first.nonzero()
        instances = np.column_stack((rows, columns)).astype(np.int64, copy=False)
        for relation in path[1:]:
            adjacency = adjs[relation].tocsr()
            if instances.shape[0] == 0:
                instances = np.empty((0, instances.shape[1] + 1), dtype=np.int64)
                continue
            last = instances[:, -1]
            counts = np.diff(adjacency.indptr)[last]
            expanded = np.repeat(instances, counts, axis=0)
            destinations = np.concatenate([
                adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]]
                for node, count in zip(last, counts) if count
            ]) if counts.sum() else np.empty(0, dtype=np.int64)
            instances = np.column_stack((expanded, destinations))
        result[name] = _sample_magnn_instances(
            instances, instances_per_node, seed
        )
    return result


def _sample_magnn_instances(instances, limit, seed):
    if limit <= 0 or instances.shape[0] == 0:
        return instances
    order = np.argsort(instances[:, 0], kind="stable")
    ordered = instances[order]
    boundaries = np.flatnonzero(np.diff(ordered[:, 0])) + 1
    groups = np.split(ordered, boundaries)
    selected = []
    for group in groups:
        if len(group) <= limit:
            selected.append(group)
            continue
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, int(group[0, 0])])
        )
        endpoints, inverse, counts = np.unique(
            group[:, -1], return_inverse=True, return_counts=True
        )
        del endpoints
        probabilities = counts[inverse].astype(np.float64) ** -0.25
        probabilities /= probabilities.sum()
        selected.append(group[
            rng.choice(len(group), limit, replace=False, p=probabilities)
        ])
    return np.concatenate(selected, axis=0)


def _replace_magnn_target_instances(
    clean, adjs, clean_instances, target, limit, seed,
):
    replacement = {}
    for name, path in zip(_magnn_metapaths(clean), clean.meta_paths):
        first = adjs[path[0]].tocsr()
        neighbors = first.indices[first.indptr[target]:first.indptr[target + 1]]
        if len(neighbors):
            instances = np.column_stack((
                np.full(len(neighbors), target, dtype=np.int64),
                neighbors.astype(np.int64, copy=False),
            ))
        else:
            instances = np.empty((0, 2), dtype=np.int64)
        for relation in path[1:]:
            adjacency = adjs[relation].tocsr()
            if instances.shape[0] == 0:
                instances = np.empty((0, instances.shape[1] + 1), dtype=np.int64)
                continue
            last = instances[:, -1]
            counts = np.diff(adjacency.indptr)[last]
            expanded = np.repeat(instances, counts, axis=0)
            destinations = np.concatenate([
                adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]]
                for node, count in zip(last, counts) if count
            ]) if counts.sum() else np.empty(0, dtype=np.int64)
            instances = np.column_stack((expanded, destinations))
        sampled = _sample_magnn_instances(instances, limit, seed)
        retained = clean_instances[name][clean_instances[name][:, 0] != target]
        replacement[name] = np.concatenate((retained, sampled), axis=0)
    return replacement


def _patch_magnn(model):
    model.forward = types.MethodType(_magnn_forward, model)
    for layer in model.layers:
        layer.inter_metapath_trans = types.MethodType(
            _magnn_inter_metapath_trans, layer
        )


def _magnn_forward(model, graph, feat_dict=None):
    features = feat_dict
    for layer in model.layers[:-1]:
        features, _ = layer(features, model.metapath_idx_dict)
        features = {key: model.activation(value) for key, value in features.items()}
    output, _ = model.layers[-1](features, model.metapath_idx_dict)
    return output


def _magnn_inter_metapath_trans(layer, feat_dict, feat_intra, metapath_list):
    scores = {}
    for metapath in metapath_list:
        node_type = metapath.split("-")[0]
        summary = torch.tanh(
            layer.inter_linear[node_type](feat_intra[metapath])
        ).mean(dim=0)
        scores[metapath] = layer.inter_attn_vec[node_type](summary).reshape(())
    result = {}
    for node_type in layer.ntypes:
        selected = [
            metapath for metapath in metapath_list
            if metapath.split("-")[0] == node_type
        ]
        if not selected:
            result[node_type] = feat_dict[node_type]
            continue
        attention = F.softmax(
            torch.stack([scores[value] for value in selected]), dim=0
        )
        result[node_type] = torch.stack([
            attention[index] * feat_intra[value]
            for index, value in enumerate(selected)
        ]).sum(dim=0)
    return result


def _heco_metapaths(clean):
    canonical = {relation: value for value in clean.canonical_etypes for relation in [value[1]]}
    return {
        f"mp{index}": [canonical[relation] for relation in path]
        for index, path in enumerate(clean.meta_paths)
    }


def _heco_positive(adjs, meta_paths, topk, device):
    combined = None
    for path in meta_paths:
        value = adjs[path[0]].astype(np.float32)
        for relation in path[1:]:
            value = value @ adjs[relation].astype(np.float32)
        combined = value.tocsr() if combined is None else combined + value
    combined = combined.tolil()
    combined.setdiag(np.asarray(combined.diagonal()).reshape(-1) + 1.0)
    combined = combined.tocsr()
    rows = []
    columns = []
    for row in range(combined.shape[0]):
        start, end = combined.indptr[row], combined.indptr[row + 1]
        candidates = combined.indices[start:end]
        scores = combined.data[start:end]
        if len(candidates) > topk:
            selected = np.argpartition(scores, -topk)[-topk:]
            candidates = candidates[selected]
        rows.extend([row] * len(candidates))
        columns.extend(candidates.tolist())
    indices = torch.as_tensor([rows, columns], dtype=torch.long, device=device)
    values = torch.ones(len(rows), dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        indices, values, combined.shape, device=device
    ).coalesce()


def _history_row(
    stage, epoch, contrastive_loss=None, classification_loss=None,
    val_loss=None, validation=None,
):
    validation = validation or {}
    return {
        "stage": stage,
        "epoch": epoch,
        "contrastive_loss": "" if contrastive_loss is None else contrastive_loss,
        "classification_loss": "" if classification_loss is None else classification_loss,
        "val_loss": "" if val_loss is None else val_loss,
        "val_accuracy": validation.get("accuracy", ""),
        "val_micro_f1": validation.get("micro_f1", ""),
        "val_macro_f1": validation.get("macro_f1", ""),
    }


def _diagnostics(name):
    return {
        "implementation": "official OpenHGNN model with DVCL protocol adapter",
        "openhgnn_version": OPENHGNN_VERSION,
        "openhgnn_revision": OPENHGNN_REVISION,
        "official_model_sha256": OPENHGNN_MODEL_HASHES[name],
    }
