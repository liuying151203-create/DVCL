import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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

    dblp_ablation = RUN_SUITE.load_config(
        ROOT / "configs" / "suites" / "dblp_poisoning_ablation_v1.yaml"
    )
    assert len(list(RUN_SUITE.commands(dblp_ablation, "python", ROOT))) == 140


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
    assert len(commands) == 330
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
    assert len(list(RUN_SUITE.commands(rnd, "python", ROOT))) == 550


def test_openhgnn_poisoning_suite_size():
    config = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "openhgnn_baselines_poisoning_v1.yaml"
    )
    commands = list(RUN_SUITE.commands(config, "python", ROOT))
    assert len(commands) == 440
    assert all(command[command.index("--backend") + 1] == "openhgnn" for command in commands)


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
    assert len(list(RUN_SUITE.commands(poisoning, "python", ROOT))) == 605
    assert len(list(RUN_SUITE.commands(target, "python", ROOT))) == 165
    assert len(list(RUN_SUITE.commands(rnd, "python", ROOT))) == 275


def test_attack_seed_recheck_suite_size():
    config = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "acm_dblp_attack_seed_recheck_v1.yaml"
    )
    commands = list(RUN_SUITE.commands(config, "python", ROOT))
    assert len(commands) == 720
    assert {
        command[command.index("--attack-seed") + 1] for command in commands
    } == {"1", "2", "3"}


def test_dvcl_adaptive_suite_size_and_semantics():
    config = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "dvcl_adaptive_target_evasion_v1.yaml"
    )
    commands = list(RUN_SUITE.commands(config, "python", ROOT))
    assert len(commands) == 30
    assert all(command[command.index("--model") + 1] == "dvcl" for command in commands)
    assert all(command[command.index("--threat-model") + 1] == "evasion" for command in commands)
    assert all(command[command.index("--scope") + 1] == "target" for command in commands)
    assert all("--adaptive" in command for command in commands)


def test_model_selection_partitions_formal_suite():
    config = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "dblp_poisoning_main_v1.yaml"
    )
    selected = RUN_SUITE.select_models(config, ["han"])
    commands = list(RUN_SUITE.commands(selected, "python", ROOT))
    assert len(commands) == 55
    assert all(command[command.index("--model") + 1] == "han" for command in commands)


def test_dimension_filters_select_one_adaptive_pilot_unit():
    config = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "adaptive_attack_strength_pilot_v1.yaml"
    )
    config = RUN_SUITE.select_models(config, ["han"])
    config = RUN_SUITE.select_datasets(config, ["acm"])
    config = RUN_SUITE.select_attacks(config, ["cand_16"], [1])
    config = RUN_SUITE.select_seeds(config, split=[1], attack=[1], train=[1])
    commands = list(RUN_SUITE.commands(config, "python", ROOT))
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--dataset") + 1] == "acm"
    assert command[command.index("--attack-variant") + 1] == "cand_16"
    assert command[command.index("--rate") + 1] == "1"
    assert command[command.index("--attack-seed") + 1] == "1"
    assert command[command.index("--train-seed") + 1] == "1"


def test_adaptive_strength_screen_has_twelve_units():
    config = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "adaptive_attack_strength_screen_v1.yaml"
    )
    commands = list(RUN_SUITE.commands(config, "python", ROOT))
    assert len(commands) == 12
    assert {
        command[command.index("--attack-variant") + 1] for command in commands
    } == {"cand_16", "cand_64", "cand_128"}


def test_variant_selection_partitions_ablation_suite():
    config = RUN_SUITE.load_config(
        ROOT / "configs" / "suites" / "acm_poisoning_ablation_v1.yaml"
    )
    selected = RUN_SUITE.select_variants(config, ["no_cl"])
    commands = list(RUN_SUITE.commands(selected, "python", ROOT))
    assert len(commands) == 35
    assert all('"variant":"no_cl"' in command[-1] for command in commands)


def test_seed_filter_partitions_legacy_ablation_suite():
    config = RUN_SUITE.load_config(
        ROOT / "configs" / "suites" / "dblp_poisoning_ablation_v1.yaml"
    )
    config = RUN_SUITE.select_variants(config, ["no_cl"])
    selected = RUN_SUITE.select_seeds(config, train=[2, 4])
    commands = list(RUN_SUITE.commands(selected, "python", ROOT))
    assert len(commands) == 14
    assert {
        command[command.index("--train-seed") + 1] for command in commands
    } == {"2", "4"}


def test_seed_filter_rejects_unknown_legacy_ablation_seed():
    config = RUN_SUITE.load_config(
        ROOT / "configs" / "suites" / "dblp_poisoning_ablation_v1.yaml"
    )
    with pytest.raises(ValueError, match="Unknown selected train seeds: \\[6\\]"):
        RUN_SUITE.select_seeds(config, train=[6])


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


def test_attack_path_pattern_supports_all_seed_dimensions():
    config = {
        "protocol": "adaptive",
        "datasets": ["acm"],
        "models": [{"name": "dvcl", "backend": "native"}],
        "attacks": [{
            "name": "query",
            "rates": [5],
            "path_pattern": (
                "data/{dataset}/{model}/split_{split_seed}/attack_{attack_seed}/"
                "train_{train_seed}.pt"
            ),
        }],
        "seeds": {"split": [1], "attack": [2], "train": [3]},
    }
    command = next(RUN_SUITE.commands(config, "python", ROOT))
    assert command[command.index("--attack-path") + 1] == (
        "data/acm/dvcl/split_1/attack_2/train_3.pt"
    )


def test_aminer_relation_pilot_expands_twelve_isolated_runs():
    config = RUN_SUITE.load_config(
        ROOT / "configs/protocols/aminer_poisoning_relation_pilot_v1.yaml"
    )
    commands = list(RUN_SUITE.commands(config, "python", ROOT))
    assert len(commands) == 12
    variants = {
        command[command.index("--attack-variant") + 1] for command in commands
    }
    assert variants == {"pa", "pr", "joint"}
    for command in commands:
        variant = command[command.index("--attack-variant") + 1]
        attack = command[command.index("--attack") + 1]
        path = command[command.index("--attack-path") + 1]
        assert f"/f1_relations/{variant}/{attack}/rate_15/seed_1/attack.pt" in path


def test_checkpoint_pattern_is_expanded_per_model_and_train_seed():
    config = {
        "protocol": "adaptive",
        "datasets": ["acm"],
        "models": [{"name": "han", "backend": "native"}],
        "attacks": [{
            "name": "adaptive_query", "rates": [3],
            "threat_model": "evasion", "scope": "target", "adaptive": True,
        }],
        "seeds": {"split": [1], "attack": [2], "train": [3]},
        "checkpoint_pattern": (
            "outputs/clean/{dataset}/{model}/split_{split_seed}/train_{train_seed}.pt"
        ),
    }
    command = next(RUN_SUITE.commands(config, "python", ROOT))
    assert command[command.index("--checkpoint-source") + 1] == (
        "outputs/clean/acm/han/split_1/train_3.pt"
    )


def test_adaptive_clean_and_strength_pilot_suite_sizes():
    clean = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "adaptive_clean_checkpoints_v1.yaml"
    )
    pilot = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "adaptive_attack_strength_pilot_v1.yaml"
    )
    assert len(list(RUN_SUITE.commands(clean, "python", ROOT))) == 165
    commands = list(RUN_SUITE.commands(pilot, "python", ROOT))
    assert len(commands) == 432
    assert all("--checkpoint-source" in command for command in commands)
    assert all("--adaptive" in command for command in commands)


def test_formal_adaptive_suite_uses_three_paired_seed_repeats():
    config = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "adaptive_target_evasion_v1.yaml"
    )
    commands = list(RUN_SUITE.commands(config, "python", ROOT))
    assert len(commands) == 99
    assert {
        (
            command[command.index("--attack-seed") + 1],
            command[command.index("--train-seed") + 1],
        )
        for command in commands
    } == {("1", "1"), ("2", "2"), ("3", "3")}


def test_dvcl_view_diagnosis_pilot_sizes_and_variants():
    clean = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "dvcl_view_diagnosis_clean_pilot_v1.yaml"
    )
    evaluation = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" / "dvcl_view_diagnosis_pilot_v1.yaml"
    )
    clean_commands = list(RUN_SUITE.commands(clean, "python", ROOT))
    evaluation_commands = list(RUN_SUITE.commands(evaluation, "python", ROOT))
    assert len(clean_commands) == 15
    assert len(evaluation_commands) == 60
    variants = {"topo", "feat", "concat", "gate", "gated_concat"}
    assert {
        json.loads(command[command.index("--model-config-json") + 1])["variant"]
        for command in evaluation_commands
    } == variants
    assert sum("--adaptive" in command for command in evaluation_commands) == 15
    assert all("--checkpoint-source" in command for command in evaluation_commands)
    for command in evaluation_commands:
        model_config = json.loads(
            command[command.index("--model-config-json") + 1]
        )
        checkpoint = command[command.index("--checkpoint-source") + 1]
        assert f"/dvcl/{model_config['variant']}/clean/" in checkpoint


def test_dvcl_reliability_gate_pilot_has_fifteen_physical_runs():
    clean = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" /
        "dvcl_reliability_gate_clean_pilot_v1.yaml"
    )
    evaluation = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" /
        "dvcl_reliability_gate_pilot_v1.yaml"
    )
    clean_commands = list(RUN_SUITE.commands(clean, "python", ROOT))
    evaluation_commands = list(RUN_SUITE.commands(evaluation, "python", ROOT))
    assert len(clean_commands) == 3
    assert len(evaluation_commands) == 12
    assert sum("--adaptive" in command for command in evaluation_commands) == 3
    assert all("--checkpoint-source" in command for command in evaluation_commands)
    assert all(
        json.loads(command[command.index("--model-config-json") + 1])[
            "fusion_mode"
        ] == "reliability_gate"
        for command in clean_commands + evaluation_commands
    )


def test_dvcl_reliability_gate_aug_pilot_changes_training_only():
    clean = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" /
        "dvcl_reliability_gate_aug_clean_pilot_v1.yaml"
    )
    evaluation = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" /
        "dvcl_reliability_gate_aug_pilot_v1.yaml"
    )
    clean_commands = list(RUN_SUITE.commands(clean, "python", ROOT))
    evaluation_commands = list(RUN_SUITE.commands(evaluation, "python", ROOT))
    assert len(clean_commands) == 3
    assert len(evaluation_commands) == 12
    for command in clean_commands + evaluation_commands:
        model_config = json.loads(
            command[command.index("--model-config-json") + 1]
        )
        assert model_config["fusion_mode"] == "reliability_gate"
        assert model_config["structure_augment_rate"] == 0.1
        assert model_config["lambda_aug"] == 1.0


def test_seed_filters_preserve_configured_pairs():
    config = {
        "datasets": ["acm"],
        "models": [{"name": "han"}],
        "attacks": [{"name": "clean"}],
        "seeds": {
            "split": [1],
            "pairs": [
                {"attack": 1, "train": 1},
                {"attack": 2, "train": 2},
            ],
        },
    }
    selected = RUN_SUITE.select_seeds(config, attack=[2], train=[2])
    commands = list(RUN_SUITE.commands(selected, "python", ROOT))
    assert len(commands) == 1
    assert commands[0][commands[0].index("--attack-seed") + 1] == "2"
    assert commands[0][commands[0].index("--train-seed") + 1] == "2"


def test_dvcl_hyperparameter_sensitivity_has_120_one_factor_runs():
    config = RUN_SUITE.load_config(
        ROOT / "configs" / "protocols" /
        "dvcl_hyperparameter_sensitivity_v1.yaml"
    )
    commands = list(RUN_SUITE.commands(config, "python", ROOT))
    assert len(commands) == 120
    model_configs = [
        json.loads(command[command.index("--model-config-json") + 1])
        for command in commands
    ]
    assert {value["variant"] for value in model_configs} == {
        variant["name"] for variant in config["variants"]
    }
    assert all(value["topology_source"] == "graph" for value in model_configs)
    assert all(
        value["semantic_topology_filter"] == "hard"
        for value in model_configs
    )


def test_model_efficiency_has_99_profiled_physical_runs():
    config = RUN_SUITE.load_config(
        ROOT / "configs/protocols/model_efficiency_v1.yaml"
    )
    commands = list(RUN_SUITE.commands(config, "python", ROOT))
    assert len(commands) == 99
    assert {command[command.index("--dataset") + 1] for command in commands} == {
        "acm", "dblp", "aminer",
    }
    assert {command[command.index("--model") + 1] for command in commands} == {
        "han", "heterosage", "rohe", "heteroguard", "fastrohgcn",
        "hgt", "magnn", "heco", "simplehgn", "hseco", "dvcl",
    }
    assert all("--profile-efficiency" in command for command in commands)
    assert all(
        command[command.index("--profile-inference-warmup") + 1] == "10"
        for command in commands
    )
    assert all(
        command[command.index("--profile-inference-repetitions") + 1] == "50"
        for command in commands
    )


def test_cuda_oom_is_retried_without_retrying_other_failures():
    responses = iter([
        SimpleNamespace(returncode=RUN_SUITE.CUDA_OOM_EXIT_CODE),
        SimpleNamespace(returncode=0),
    ])
    calls = []
    sleeps = []

    def runner(command, cwd, check):
        calls.append((command, cwd, check))
        return next(responses)

    result = RUN_SUITE.run_with_oom_retries(
        ["python", "experiment.py"], ROOT, retries=2, delay=3,
        runner=runner, sleeper=sleeps.append,
    )
    assert result.returncode == 0
    assert len(calls) == 2
    assert sleeps == [3]

    non_oom_calls = []

    def fail_once(command, cwd, check):
        non_oom_calls.append((command, cwd, check))
        return SimpleNamespace(returncode=1)

    result = RUN_SUITE.run_with_oom_retries(
        ["python", "experiment.py"], ROOT, retries=2,
        runner=fail_once,
        sleeper=sleeps.append,
    )
    assert result.returncode == 1
    assert len(non_oom_calls) == 1
