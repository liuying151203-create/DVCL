import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_adaptive_pool", ROOT / "scripts" / "run_adaptive_pool.py"
)
POOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POOL)


def command(model, train_seed=1, attack_seed=1, device="cuda:0"):
    return [
        "python", "run_experiment.py", "--model", model,
        "--dataset", "acm", "--train-seed", str(train_seed),
        "--attack-seed", str(attack_seed), "--device", device,
    ]


def test_device_replacement_and_slow_model_priority():
    value = POOL.replace_device(command("han"), "cuda:3")
    assert value[value.index("--device") + 1] == "cuda:3"
    ordered = POOL.prioritize_commands([
        command("han"), command("hseco"), command("dvcl")
    ])
    assert [POOL.options(item)["--model"] for item in ordered] == [
        "dvcl", "hseco", "han"
    ]
    assert POOL.options(value)["--device"] == "cuda:3"


def test_required_audits_must_be_complete(tmp_path: Path):
    confirmation = tmp_path / "confirmation.json"
    inputs = tmp_path / "inputs.json"
    confirmation.write_text(json.dumps({"ok": True}), encoding="utf-8")
    inputs.write_text(json.dumps({
        "summary": {"total": 180, "passed": 180, "failed": 0}
    }), encoding="utf-8")
    POOL.validate_audits(confirmation, inputs)
    confirmation.write_text(json.dumps({"ok": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="Confirmation audit"):
        POOL.validate_audits(confirmation, inputs)


def test_formal_suite_expands_to_495_prioritized_runs():
    run_suite = POOL.load_run_suite()
    config = run_suite.load_config(
        ROOT / "configs" / "protocols" / "adaptive_target_evasion_v1.yaml"
    )
    commands = POOL.prioritize_commands(
        list(run_suite.commands(config, "python", ROOT))
    )
    assert len(commands) == 495
    assert POOL.options(commands[0])["--model"] == "dvcl"
