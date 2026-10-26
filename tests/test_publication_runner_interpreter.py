"""Ensure publication runs reuse the interpreter selected by the user."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_publication_final.py"
    spec = importlib.util.spec_from_file_location("qcenn_publication_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_venv_interpreter_is_selected(monkeypatch, tmp_path):
    runner = _runner()
    venv = tmp_path / ".venv-qcenn"
    python = venv / "bin" / "python"
    monkeypatch.setattr(runner.sys, "prefix", str(venv))
    monkeypatch.setattr(runner.sys, "base_prefix", "/usr")
    monkeypatch.setattr(runner.sys, "executable", str(python))
    assert runner.select_environment(None) == (venv, python)


def test_explicit_custom_venv_is_allowed(tmp_path):
    runner = _runner()
    venv = tmp_path / ".venv-qcenn"
    result, executable = runner.select_environment(venv)
    assert result == venv
    assert executable == venv / "bin" / "python"


def test_non312_python_is_compatibility_checked_not_rejected(monkeypatch, tmp_path):
    runner = _runner()
    venv = tmp_path / ".venv-qcenn"
    executable = venv / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(runner, "interpreter_version", lambda _: (3, 14))
    verified = []
    monkeypatch.setattr(
        runner, "verify_dependencies",
        lambda binary, strict_lock: verified.append((binary, strict_lock)),
    )
    selected, mode = runner.python_environment(
        venv, executable=executable, skip_install=True
    )
    assert selected == executable
    assert mode == "compatible-unpinned-python"
    assert verified == [(executable, False)]


def test_exact_lock_only_on_python312(monkeypatch, tmp_path):
    runner = _runner()
    executable = tmp_path / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(runner, "interpreter_version", lambda _: (3, 12))
    choices = []
    monkeypatch.setattr(
        runner, "verify_dependencies",
        lambda binary, strict_lock: choices.append(strict_lock),
    )
    _, mode = runner.python_environment(
        tmp_path, executable=executable, skip_install=True
    )
    assert mode == "exact-python312-lock"
    assert choices == [True]


def test_missing_custom_python_is_not_silently_replaced(tmp_path):
    runner = _runner()
    with pytest.raises(RuntimeError, match="No Python interpreter"):
        runner.python_environment(
            tmp_path / ".venv-qcenn",
            executable=tmp_path / ".venv-qcenn/bin/python",
            skip_install=True,
        )


def test_publication_runner_does_not_reject_dirty_git_tree(monkeypatch):
    runner = _runner()
    observed = []

    def fake_git(*args):
        observed.append(args)
        if args == ("rev-parse", "HEAD"):
            return "1234567890abcdef"
        if args == ("status", "--porcelain", "--untracked-files=normal"):
            return " M manuscript/QCeNN_MI.tex"
        raise AssertionError(f"Unexpected git query: {args}")

    monkeypatch.setattr(runner, "git", fake_git)
    assert runner.source_revision() == "1234567890abcdef"
    assert ("status", "--porcelain", "--untracked-files=normal") not in observed


def test_source_fingerprint_snapshot_is_retained_for_dirty_checkout():
    runner = _runner()
    from qcenn_mi.source_provenance import source_fingerprint

    assert runner.source_fingerprint() == source_fingerprint(runner.ROOT)
