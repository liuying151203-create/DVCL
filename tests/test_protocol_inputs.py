import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_protocol_inputs", ROOT / "scripts" / "check_protocol_inputs.py"
)
CHECK_INPUTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK_INPUTS)


def test_main_protocol_has_expected_unique_inputs():
    config = CHECK_INPUTS.load_config(ROOT / "configs" / "protocols" / "dvcl_main.yaml")
    requirements = list(CHECK_INPUTS.protocol_requirements(config, ROOT))
    assert len(requirements) == 24
    assert sum(value["kind"] == "clean" for value in requirements) == 2
    assert sum(value["kind"] == "split" for value in requirements) == 2
    assert sum(value["kind"] == "attack" for value in requirements) == 20
    assert any(
        value["kind"] == "attack"
        and value["dataset"] == "dblp"
        and value["attack"] == "heteprbcd"
        and value["rate"] == 25
        for value in requirements
    )
