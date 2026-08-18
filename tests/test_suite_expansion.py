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


def test_corrected_acm_poisoning_suite_sizes():
    main = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "acm_poisoning_main_v1.yaml"
    )
    ablation = RUN_SUITE.load_config(
        ROOT / "configs" / "suites" / "acm_poisoning_ablation_v1.yaml"
    )
    assert len(list(RUN_SUITE.commands(main, "python", ROOT))) == 220
    assert len(list(RUN_SUITE.commands(ablation, "python", ROOT))) == 140


def test_hg_baseline_target_evasion_suite_size():
    config = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "hg_baseline_target_evasion_v1.yaml"
    )
    commands = list(RUN_SUITE.commands(config, "python", ROOT))
    assert len(commands) == 210
    assert all(command[command.index("--threat-model") + 1] == "evasion" for command in commands)
    assert all(command[command.index("--scope") + 1] == "target" for command in commands)


def test_robust_baseline_and_rnd_suite_sizes():
    robust = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "robust_baselines_poisoning_v1.yaml"
    )
    rnd = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "rnd_poisoning_v1.yaml"
    )
    assert len(list(RUN_SUITE.commands(robust, "python", ROOT))) == 330
    assert len(list(RUN_SUITE.commands(rnd, "python", ROOT))) == 350


def test_attack_factorial_suite_has_isolated_variants():
    config = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "attack_factorial_v1.yaml"
    )
    commands = list(RUN_SUITE.commands(config, "python", ROOT))
    assert len(commands) == 240
    variants = {command[command.index("--attack-variant") + 1] for command in commands}
    assert variants == {"unconstrained", "unbiased"}


def test_aminer_suite_sizes():
    poisoning = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "aminer_poisoning_main_v1.yaml"
    )
    target = RUN_SUITE.load_config(
        ROOT
        / "configs"
        / "protocols"
        / "aminer_hg_baseline_target_evasion_v1.yaml"
    )
    rnd = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "aminer_rnd_poisoning_v1.yaml"
    )
    assert len(list(RUN_SUITE.commands(poisoning, "python", ROOT))) == 385
    assert len(list(RUN_SUITE.commands(target, "python", ROOT))) == 105
    assert len(list(RUN_SUITE.commands(rnd, "python", ROOT))) == 175


def test_model_selection_partitions_formal_suite():
    config = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "dblp_poisoning_main_v1.yaml"
    )
    selected = RUN_SUITE.select_models(config, ["han"])
    commands = list(RUN_SUITE.commands(selected, "python", ROOT))
    assert len(commands) == 55
    assert all(command[command.index("--model") + 1] == "han" for command in commands)


def test_variant_selection_partitions_ablation_suite():
    config = RUN_SUITE.load_config(
        ROOT / "configs" / "suites" / "acm_poisoning_ablation_v1.yaml"
    )
    selected = RUN_SUITE.select_variants(config, ["no_cl"])
    commands = list(RUN_SUITE.commands(selected, "python", ROOT))
    assert len(commands) == 35
    assert all('"variant":"no_cl"' in command[-1] for command in commands)


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


def test_target_evasion_and_adaptive_flags_are_expanded():
    config = {
        "protocol": "target",
        "datasets": ["acm"],
        "models": [{"name": "han", "backend": "native"}],
        "attacks": [{
            "name": "hg_baseline",
            "rates": [3],
            "threat_model": "evasion",
            "scope": "target",
            "adaptive": True,
            "path_pattern": "data/{dataset}/{model}/{attack}_{rate}.pt",
        }],
        "seeds": {"split": [1], "attack": [1], "train": [1]},
    }
    command = next(RUN_SUITE.commands(config, "python", ROOT))
    assert command[command.index("--threat-model") + 1] == "evasion"
    assert command[command.index("--scope") + 1] == "target"
    assert command[command.index("--attack-path") + 1] == "data/acm/han/hg_baseline_3.pt"
    assert "--adaptive" in command
