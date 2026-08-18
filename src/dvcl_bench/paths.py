import os
from dataclasses import dataclass
from pathlib import Path


def format_rate(rate: float) -> str:
    return str(int(rate)) if float(rate).is_integer() else str(rate).replace(".", "p")


@dataclass(frozen=True)
class ExperimentLayout:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def legacy_hseco(self) -> Path:
        override = os.environ.get("DVCL_PRIVATE_HSECO_ROOT")
        if override:
            return Path(override).expanduser().resolve()
        return self.root.parent / "private" / "hseco_reference"

    def clean_path(self, dataset: str) -> Path:
        return self.data / "processed" / dataset / "clean.pt"

    def split_path(self, dataset: str, split_name: str) -> Path:
        return self.data / "splits" / dataset / f"{split_name}.pt"

    def attack_path(self, dataset: str, attack: str, rate: float, attack_seed: int) -> Path:
        return (
            self.data
            / "attacks"
            / dataset
            / attack
            / f"rate_{format_rate(rate)}"
            / f"seed_{attack_seed}"
            / "attack.pt"
        )

    def run_dir(self, spec) -> Path:
        variant = str(spec.model.config.get("variant", "default"))
        attack_name = spec.attack.name
        if spec.attack.variant != "default":
            attack_name = f"{attack_name}_{spec.attack.variant}"
        return (
            self.outputs
            / "runs"
            / spec.protocol
            / spec.dataset
            / spec.model.name
            / variant
            / attack_name
            / f"rate_{format_rate(spec.attack.rate)}"
            / f"split_seed_{spec.seeds.split}"
            / f"attack_seed_{spec.seeds.attack}"
            / f"train_seed_{spec.seeds.train}"
        )
