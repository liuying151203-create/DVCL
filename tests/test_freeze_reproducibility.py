import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "freeze_reproducibility.py"
SPEC = importlib.util.spec_from_file_location("freeze_reproducibility", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tree_sha256_depends_on_relative_path_and_content(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("same", encoding="utf-8")
    second.write_text("same", encoding="utf-8")
    assert MODULE.tree_sha256([first]) != MODULE.tree_sha256([second])
    initial = MODULE.tree_sha256([first, second])
    second.write_text("changed", encoding="utf-8")
    assert MODULE.tree_sha256([first, second]) != initial


def test_files_under_is_sorted_and_excludes_pycache(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    (tmp_path / "z.txt").write_text("z", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"ignored")
    assert [path.name for path in MODULE.files_under([tmp_path])] == ["a.txt", "z.txt"]
