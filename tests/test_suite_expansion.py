import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_suite", ROOT / "scripts" / "run_suite.py")
RUN_SUITE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_SUITE)


def test_main_and_ablation_suite_sizes():
    main = RUN_SUITE.load_config(ROOT / "configs" / "protocols" / "dvcl_main.yaml")
    ablation = RUN_SUITE.load_config(
        ROOT / "configs" / "suites" / "dvcl_component_ablation.yaml"
    )
    assert len(list(RUN_SUITE.commands(main, "python", ROOT))) == 220
    assert len(list(RUN_SUITE.commands(ablation, "python", ROOT))) == 140


def test_baseline_suite_size():
    baseline = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "baseline_main.yaml"
    )
    assert len(list(RUN_SUITE.commands(baseline, "python", ROOT))) == 220


def test_corrected_dblp_poisoning_suite_size():
    config = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "dblp_poisoning_main_v1.yaml"
    )
    assert len(list(RUN_SUITE.commands(config, "python", ROOT))) == 220


def test_model_selection_partitions_formal_suite():
    config = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "dblp_poisoning_main_v1.yaml"
    )
    selected = RUN_SUITE.select_models(config, ["han"])
    commands = list(RUN_SUITE.commands(selected, "python", ROOT))
    assert len(commands) == 55
    assert all(command[command.index("--model") + 1] == "han" for command in commands)


def test_attack_path_pattern_is_expanded():
    config = {
        "protocol": "pilot",
        "datasets": ["dblp"],
        "models": [{"name": "han", "backend": "native"}],
        "attacks": [{
            "name": "heteprbcd",
            "rates": [5],
            "path_pattern": "outputs/pilots/{dataset}/{attack}/rate_{rate}/attack.pt",
        }],
        "seeds": {"split": [1], "attack": [1], "train": [1]},
    }
    command = next(RUN_SUITE.commands(config, "python", ROOT))
    index = command.index("--attack-path")
    assert command[index + 1] == "outputs/pilots/dblp/heteprbcd/rate_5/attack.pt"
