import importlib.util
import json
from pathlib import Path

from scripts.freeze_adaptive_attack_protocols import (
    confirmation_config,
    formal_config,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_suite_for_adaptive_freeze", ROOT / "scripts" / "run_suite.py"
)
RUN_SUITE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_SUITE)


def selection_file(tmp_path: Path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"selected_candidate_size": 64}), encoding="utf-8")
    return path


def test_confirmation_protocol_has_48_searches_and_144_evaluations(tmp_path: Path):
    config = confirmation_config(64, selection_file(tmp_path))
    commands = list(RUN_SUITE.commands(config, "python", ROOT))
    assert len(commands) == 48
    assert config["evaluation_budgets"] == [1, 3, 5]
    assert {command[command.index("--model") + 1] for command in commands} == {
        "han", "heteroguard", "hseco", "dvcl"
    }
    assert all("cand_64" in command for command in commands)


def test_formal_protocol_has_99_searches_and_297_evaluations(tmp_path: Path):
    config = formal_config(64, selection_file(tmp_path))
    commands = list(RUN_SUITE.commands(config, "python", ROOT))
    assert len(commands) == 99
    assert config["evaluation_budgets"] == [1, 3, 5]
    assert all("--checkpoint-source" in command for command in commands)
    assert {command[command.index("--dataset") + 1] for command in commands} == {
        "acm", "dblp", "aminer"
    }
    assert config["seeds"]["pairs"] == [
        {"attack": 1, "train": 1},
        {"attack": 2, "train": 2},
        {"attack": 3, "train": 3},
    ]
