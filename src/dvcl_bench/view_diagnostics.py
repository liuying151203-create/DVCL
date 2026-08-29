"""Target-level diagnostics for DVCL topology and feature views."""

from __future__ import annotations

import statistics

import torch
from torch.nn import functional as F


def target_view_diagnostics(clean_state, attacked_states, labels):
    targets = torch.as_tensor(
        [int(state["target"]) for state in attacked_states],
        dtype=torch.long,
        device=labels.device,
    )
    target_labels = labels[targets]
    output = {
        "definition": clean_state.get(
            "diagnostic_definition",
            "same-checkpoint branch zero-ablation diagnostics",
        ),
        "clean": {},
        "attacked": {},
        "drift": {},
        "gate": {},
        "per_target": [],
    }
    clean_values = _target_values(clean_state, targets)
    attacked_values = _stack_attacked(attacked_states, labels.device)
    for name in ("fused", "topology", "feature"):
        clean_logits = clean_values.get(f"{name}_logits")
        attacked_logits = attacked_values.get(f"{name}_logits")
        if clean_logits is None or attacked_logits is None:
            continue
        output["clean"].update(_prediction_summary(name, clean_logits, target_labels))
        output["attacked"].update(
            _prediction_summary(name, attacked_logits, target_labels)
        )
    if "topology_logits" in clean_values and "feature_logits" in clean_values:
        output["clean"]["view_disagreement_rate"] = _disagreement_rate(
            clean_values["topology_logits"], clean_values["feature_logits"]
        )
        output["attacked"]["view_disagreement_rate"] = _disagreement_rate(
            attacked_values["topology_logits"], attacked_values["feature_logits"]
        )
    for name in ("topology", "feature"):
        clean_embedding = clean_values.get(f"{name}_embedding")
        attacked_embedding = attacked_values.get(f"{name}_embedding")
        if clean_embedding is None or attacked_embedding is None:
            continue
        l2 = torch.linalg.vector_norm(attacked_embedding - clean_embedding, dim=1)
        cosine = 1 - F.cosine_similarity(
            attacked_embedding, clean_embedding, dim=1, eps=1e-12
        )
        output["drift"].update(_distribution(f"{name}_l2", l2))
        output["drift"].update(_distribution(f"{name}_cosine", cosine))
    clean_gate = clean_values.get("gate_weight")
    attacked_gate = attacked_values.get("gate_weight")
    if clean_gate is not None and attacked_gate is not None:
        clean_gate = clean_gate.flatten()
        attacked_gate = attacked_gate.flatten()
        output["gate"].update(_distribution("clean", clean_gate))
        output["gate"].update(_distribution("attacked", attacked_gate))
        output["gate"].update(_distribution("delta", attacked_gate - clean_gate))
    output["per_target"] = _per_target_rows(
        targets, target_labels, clean_values, attacked_values
    )
    return output


def _target_values(state, targets):
    result = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            result[key] = value[targets]
    return result


def _stack_attacked(states, device):
    result = {}
    keys = set.intersection(*(
        {key for key, value in state.items() if isinstance(value, torch.Tensor)}
        for state in states
    )) if states else set()
    for key in keys:
        result[key] = torch.stack([
            state[key].detach().to(device) for state in states
        ])
    return result


def _prediction_summary(prefix, logits, labels):
    predictions = logits.argmax(dim=1)
    margins = true_class_margin(logits, labels)
    return {
        f"{prefix}_target_micro_f1": float(predictions.eq(labels).float().mean()),
        **_distribution(f"{prefix}_margin", margins),
    }


def true_class_margin(logits, labels):
    true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    other = logits.clone()
    other.scatter_(1, labels.unsqueeze(1), float("-inf"))
    return true_logits - other.max(dim=1).values


def _distribution(prefix, values):
    samples = [float(value) for value in values.detach().cpu().flatten()]
    return {
        f"{prefix}_mean": statistics.fmean(samples) if samples else 0.0,
        f"{prefix}_std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def _disagreement_rate(left, right):
    return float(left.argmax(dim=1).ne(right.argmax(dim=1)).float().mean())


def _per_target_rows(targets, labels, clean, attacked):
    rows = []
    clean_fused = clean["fused_logits"]
    attacked_fused = attacked["fused_logits"]
    clean_prediction = clean_fused.argmax(dim=1)
    attacked_prediction = attacked_fused.argmax(dim=1)
    clean_margin = true_class_margin(clean_fused, labels)
    attacked_margin = true_class_margin(attacked_fused, labels)
    for index, target in enumerate(targets.detach().cpu().tolist()):
        row = {
            "target": int(target),
            "label": int(labels[index]),
            "clean_correct": bool(clean_prediction[index] == labels[index]),
            "attacked_correct": bool(attacked_prediction[index] == labels[index]),
            "attack_success": bool(
                clean_prediction[index] == labels[index]
                and attacked_prediction[index] != labels[index]
            ),
            "clean_fused_margin": float(clean_margin[index]),
            "attacked_fused_margin": float(attacked_margin[index]),
        }
        for name in ("topology", "feature"):
            logits_key = f"{name}_logits"
            embedding_key = f"{name}_embedding"
            if logits_key in clean:
                row[f"clean_{name}_prediction"] = int(
                    clean[logits_key][index].argmax()
                )
                row[f"attacked_{name}_prediction"] = int(
                    attacked[logits_key][index].argmax()
                )
            if embedding_key in clean:
                row[f"{name}_l2_drift"] = float(torch.linalg.vector_norm(
                    attacked[embedding_key][index] - clean[embedding_key][index]
                ))
                row[f"{name}_cosine_drift"] = float(1 - F.cosine_similarity(
                    attacked[embedding_key][index].unsqueeze(0),
                    clean[embedding_key][index].unsqueeze(0),
                    dim=1,
                    eps=1e-12,
                )[0])
        if "topology_logits" in clean and "feature_logits" in clean:
            row["clean_view_disagreement"] = bool(
                clean["topology_logits"][index].argmax()
                != clean["feature_logits"][index].argmax()
            )
            row["attacked_view_disagreement"] = bool(
                attacked["topology_logits"][index].argmax()
                != attacked["feature_logits"][index].argmax()
            )
        if "gate_weight" in clean:
            row["clean_gate_weight"] = float(clean["gate_weight"][index])
            row["attacked_gate_weight"] = float(attacked["gate_weight"][index])
        rows.append(row)
    return rows
