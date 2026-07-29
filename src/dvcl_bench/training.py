"""Shared full-batch training utilities and model-specific configurations."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score


@dataclass
class HSeCoTrainConfig:
    hidden_dim: int = 64
    semantic_heads: int = 8
    semantic_dropout: float = 0.3
    node_hidden_dim: int = 128
    node_heads: int = 8
    node_dropout: float = 0.6
    learning_rate: float = 0.005
    weight_decay: float = 0.001
    thresholds: Optional[List[float]] = None
    global_threshold: float = 0.002
    contrastive_weight: Optional[float] = None
    negative_noise_rate: float = 0.01
    legacy_checkpoint_semantics: bool = False

    def freeze_for_dataset(self, dataset: str) -> "HSeCoTrainConfig":
        value = copy.deepcopy(self)
        defaults = {
            "acm": ([0.0005, 0.0006], 0.2),
            "dblp": ([0.0005, 0.0006, 0.0008], 0.6),
            "aminer": ([0.0005, 0.0006], 0.2),
        }
        thresholds, weight = defaults[dataset]
        value.thresholds = value.thresholds or thresholds
        value.contrastive_weight = weight if value.contrastive_weight is None else value.contrastive_weight
        return value


@dataclass
class DVCLTrainConfig:
    hidden_dim: int = 128
    heads: int = 4
    dropout: float = 0.3
    feature_mask_rate: float = 0.2
    knn_k: int = 20
    knn_mode: str = "directed"
    view_mode: str = "both"
    fusion_mode: str = "concat"
    temperature: float = 0.5
    lambda_han: float = 1.0
    lambda_dvcl: float = 1.0
    learning_rate: float = 0.005
    weight_decay: float = 0.001
    thresholds: Optional[List[float]] = None
    global_threshold: float = 0.002
    semantic_hidden_dim: int = 64
    semantic_heads: int = 8
    legacy_checkpoint_semantics: bool = False

    def freeze_for_dataset(self, dataset: str) -> "DVCLTrainConfig":
        value = copy.deepcopy(self)
        defaults = {
            "acm": [0.0005, 0.0006],
            "dblp": [0.0005, 0.0006, 0.0008],
            "aminer": [0.0005, 0.0006],
        }
        value.thresholds = value.thresholds or defaults[dataset]
        return value


@dataclass
class TrainingResult:
    metrics: Dict[str, float]
    history: List[Dict[str, float]]
    best_epoch: int
    stopped_epoch: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class LegacyEarlyStopping:
    """In-memory equivalent of the original loss-and-accuracy stopping rule."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.best_accuracy = None
        self.best_epoch = -1
        self.state = None

    def step(self, loss: float, accuracy: float, model, epoch: int) -> bool:
        if self.best_loss is None:
            self._save(model, epoch)
            self.best_loss = loss
            self.best_accuracy = accuracy
        elif loss > self.best_loss and accuracy < self.best_accuracy:
            self.counter += 1
            return self.counter >= self.patience
        else:
            if loss <= self.best_loss and accuracy >= self.best_accuracy:
                self._save(model, epoch)
            self.best_loss = min(loss, self.best_loss)
            self.best_accuracy = max(accuracy, self.best_accuracy)
            self.counter = 0
        return False

    def restore(self, model) -> None:
        if self.state is None:
            raise RuntimeError("No early-stopping checkpoint was recorded")
        model.load_state_dict(self.state)

    def _save(self, model, epoch: int) -> None:
        self.state = copy.deepcopy(model.state_dict())
        self.best_epoch = epoch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def classification_metrics(logits, labels) -> Dict[str, float]:
    prediction = logits.argmax(dim=1).detach().cpu().numpy()
    truth = labels.detach().cpu().numpy()
    return {
        "accuracy": float((prediction == truth).mean()),
        "micro_f1": float(f1_score(truth, prediction, average="micro")),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
    }


def save_checkpoint(path: Path, model, config, best_epoch: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "best_epoch": best_epoch,
    }, Path(path))
