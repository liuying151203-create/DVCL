import importlib.util
from pathlib import Path

import numpy as np
import scipy.sparse as sp


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_aminer_source.py"
SPEC = importlib.util.spec_from_file_location("prepare_aminer_source", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_relation_builds_binary_csr(tmp_path):
    source = tmp_path / "edges.txt"
    source.write_text("0 1\n0 1\n1 2\n", encoding="utf-8")
    relation = MODULE._relation(source, (2, 3))
    assert relation.shape == (2, 3)
    assert relation.nnz == 2
    assert np.array_equal(relation.toarray(), np.array([[0, 1, 0], [0, 0, 1]]))


def test_relation_rejects_out_of_bounds_index(tmp_path):
    source = tmp_path / "edges.txt"
    source.write_text("2 0\n", encoding="utf-8")
    try:
        MODULE._relation(source, (2, 1))
    except ValueError as exc:
        assert "outside declared shape" in str(exc)
    else:
        raise AssertionError("out-of-bounds AMiner edges must fail")
