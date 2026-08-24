from dataclasses import dataclass, field
from typing import Any, Dict, Mapping


SUPPORTED_DATASETS = {"acm", "dblp", "aminer", "imdb"}
SUPPORTED_ATTACKS = {
    "clean", "rnd", "prbcd", "heteprbcd", "hg_baseline",
    "dvcl_adaptive_query",
}
LEGACY_MODELS = {"hseco", "dvcl"}
NATIVE_MODELS = {
    "hseco", "dvcl", "han", "heterosage", "heteroguard", "rohe", "fastrohgcn"
}
OPENHGNN_MODELS = {"hgt", "magnn", "heco", "simplehgn"}


@dataclass(frozen=True)
class SeedSpec:
    split: int
    attack: int
    train: int

    def __post_init__(self) -> None:
        for name, value in (("split", self.split), ("attack", self.attack), ("train", self.train)):
            if value < 0:
                raise ValueError(f"{name}_seed must be non-negative")


@dataclass(frozen=True)
class AttackSpec:
    name: str = "clean"
    rate: float = 0.0
    threat_model: str = "poisoning"
    scope: str = "global"
    adaptive: bool = False
    variant: str = "default"

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_ATTACKS:
            raise ValueError(f"Unsupported attack: {self.name}")
        if self.name == "clean" and self.rate != 0:
            raise ValueError("clean attack must use rate=0")
        if self.name != "clean" and self.rate <= 0:
            raise ValueError("non-clean attack must use a positive rate")
        if self.threat_model not in {"poisoning", "evasion"}:
            raise ValueError(f"Unsupported threat model: {self.threat_model}")
        if self.scope not in {"global", "target"}:
            raise ValueError(f"Unsupported attack scope: {self.scope}")
        if self.scope == "target" and self.threat_model != "evasion":
            raise ValueError("target attacks must use evasion semantics")
        if self.name == "hg_baseline" and (
            self.threat_model != "evasion" or self.scope != "target"
        ):
            raise ValueError("hg_baseline requires target evasion semantics")
        if self.name == "dvcl_adaptive_query" and (
            self.threat_model != "evasion"
            or self.scope != "target"
            or not self.adaptive
        ):
            raise ValueError(
                "dvcl_adaptive_query requires adaptive target evasion semantics"
            )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str = "legacy"
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend not in {"legacy", "native", "openhgnn"}:
            raise ValueError(f"Unsupported model backend: {self.backend}")
        if self.backend == "legacy" and self.name not in LEGACY_MODELS:
            raise ValueError(f"Legacy backend supports only: {sorted(LEGACY_MODELS)}")
        if self.backend == "native" and self.name not in NATIVE_MODELS:
            raise ValueError(f"Native backend supports only: {sorted(NATIVE_MODELS)}")
        if self.backend == "openhgnn" and self.name not in OPENHGNN_MODELS:
            raise ValueError(f"OpenHGNN backend supports only: {sorted(OPENHGNN_MODELS)}")


@dataclass(frozen=True)
class ExperimentSpec:
    protocol: str
    dataset: str
    split_name: str
    seeds: SeedSpec
    attack: AttackSpec
    model: ModelSpec
    device: str = "cuda:0"
    epochs: int = 200
    patience: int = 100
    extra_args: tuple = ()

    def __post_init__(self) -> None:
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentSpec":
        seeds = raw.get("seeds", {})
        attack = raw.get("attack", {})
        model = raw.get("model", {})
        return cls(
            protocol=str(raw.get("protocol", "dvcl_main")),
            dataset=str(raw["dataset"]).lower(),
            split_name=str(raw.get("split_name", f"seed_{seeds.get('split', 1)}")),
            seeds=SeedSpec(
                split=int(seeds.get("split", 1)),
                attack=int(seeds.get("attack", 1)),
                train=int(seeds.get("train", 1)),
            ),
            attack=AttackSpec(
                name=str(attack.get("name", "clean")).lower(),
                rate=float(attack.get("rate", 0)),
                threat_model=str(attack.get("threat_model", "poisoning")),
                scope=str(attack.get("scope", "global")),
                adaptive=bool(attack.get("adaptive", False)),
                variant=str(attack.get("variant", "default")),
            ),
            model=ModelSpec(
                name=str(model["name"]).lower(),
                backend=str(model.get("backend", "legacy")).lower(),
                config=dict(model.get("config", {})),
            ),
            device=str(raw.get("device", "cuda:0")),
            epochs=int(raw.get("epochs", 200)),
            patience=int(raw.get("patience", 100)),
            extra_args=tuple(str(value) for value in raw.get("extra_args", ())),
        )
