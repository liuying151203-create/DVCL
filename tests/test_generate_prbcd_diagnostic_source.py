import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "generate_prbcd_diagnostic_source.py"
)
SPEC = importlib.util.spec_from_file_location(
    "generate_prbcd_diagnostic_source", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeData:
    edge_types = [
        ("paper", "pa", "author"),
        ("author", "ap", "paper"),
        ("paper", "pr", "research"),
        ("research", "rp", "paper"),
    ]


class FakeGenerator:
    @staticmethod
    def find_reverse_edge_type(edge_types, edge_type):
        return next(
            (
                candidate for candidate in edge_types
                if candidate[0] == edge_type[2] and candidate[2] == edge_type[0]
            ),
            None,
        )


def test_resolve_relation_budget_selects_forward_relations():
    default_budget = [("paper", "pa", "author")]
    default_symmetric = {
        ("paper", "pa", "author"): ("author", "ap", "paper")
    }
    budget, symmetric = MODULE._resolve_relation_budget(
        FakeGenerator, FakeData(), "joint", default_budget, default_symmetric
    )
    assert budget == [
        ("paper", "pa", "author"),
        ("paper", "pr", "research"),
    ]
    assert set(symmetric.values()) == {
        ("author", "ap", "paper"),
        ("research", "rp", "paper"),
    }


def test_resolve_relation_budget_preserves_default():
    default_budget = [("paper", "pa", "author")]
    default_symmetric = {default_budget[0]: ("author", "ap", "paper")}
    budget, symmetric = MODULE._resolve_relation_budget(
        FakeGenerator, FakeData(), "default", default_budget, default_symmetric
    )
    assert budget is default_budget
    assert symmetric is default_symmetric
