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



def test_safe_resume_permits_only_orchestration_changes(tmp_path, monkeypatch):
    import shutil

    from qcenn_mi.source_provenance import freeze_source, source_fingerprint

    root = tmp_path / "checkout"
    (root / "src/qcenn_mi").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / ".github/workflows").mkdir(parents=True)
    (root / ".github/workflows/native-python-publication.yml").write_text("before")
    (root / "README.md").write_text("before")
    (root / "scripts/run_publication_final.py").write_text("# old runner")
    original_source = Path(__file__).resolve().parents[1] / (
        "src/qcenn_mi/source_provenance.py"
    )
    shutil.copy(original_source, root / "src/qcenn_mi/source_provenance.py")
    out = tmp_path / "partial"
    (out / "suite").mkdir(parents=True)
    index = freeze_source(root, out / "suite")
    runner = _runner()
    monkeypatch.setattr(runner, "ROOT", root)

    # Same bytes permit perfect replay.
    assert runner.verify_resumable_source(
        out, {"source_sha256": index["source_sha256"]},
        source_fingerprint(root),
    ) == []

    # Only runner metadata change is allowed; previous scientific inputs remain fixed.
    (root / "scripts/run_publication_final.py").write_text("# fixed runner")
    assert runner.verify_resumable_source(
        out, {"source_sha256": index["source_sha256"]},
        source_fingerprint(root),
    ) == ["scripts/run_publication_final.py"]

    # Workflow-only changes are allowed, but separately disclosed in metadata.
    (root / ".github/workflows/native-python-publication.yml").write_text("after")
    assert runner.verify_resumable_source(
        out, {"source_sha256": index["source_sha256"]},
        source_fingerprint(root),
    ) == [
        ".github/workflows/native-python-publication.yml",
        "scripts/run_publication_final.py",
    ]

    # New or modified scientific code must invalidate a partial-run replay.
    (root / "src/qcenn_mi/new_scientific_method.py").write_text("parameter = 1")
    with pytest.raises(RuntimeError, match="scientific campaign"):
        runner.verify_resumable_source(
            out, {"source_sha256": index["source_sha256"]},
            source_fingerprint(root),
        )


def test_resume_rejects_modified_source_snapshot(tmp_path, monkeypatch):
    import shutil

    from qcenn_mi.source_provenance import freeze_source, source_fingerprint

    root = tmp_path / "checkout"
    (root / "src/qcenn_mi").mkdir(parents=True)
    shutil.copy(
        Path(__file__).resolve().parents[1] / "src/qcenn_mi/source_provenance.py",
        root / "src/qcenn_mi/source_provenance.py",
    )
    out = tmp_path / "partial"
    (out / "suite").mkdir(parents=True)
    index = freeze_source(root, out / "suite")
    (out / "suite/source-snapshot.zip").write_bytes(b"invalid snapshot")
    runner = _runner()
    monkeypatch.setattr(runner, "ROOT", root)
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        runner.verify_resumable_source(
            out, {"source_sha256": index["source_sha256"]},
            source_fingerprint(root),
        )


def test_completed_stage_requires_valid_checksum_manifest(tmp_path):
    import hashlib
    import json

    runner = _runner()
    output = tmp_path / "oldrun"
    folder = output / "temporal-full"
    folder.mkdir(parents=True)
    (folder / "result.json").write_text('{"status":"passed"}')
    assert not runner.completed_stage(output, "phase-bias")
    with pytest.raises(RuntimeError, match="manifest absent"):
        runner.completed_stage(output, "temporal-full")
    expected = hashlib.sha256((folder / "result.json").read_bytes()).hexdigest()
    (folder / "MANIFEST.json").write_text(
        json.dumps({"sha256": {"result.json": expected}})
    )
    assert runner.completed_stage(output, "temporal-full")
    (folder / "result.json").write_text('{"status":"changed"}')
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        runner.completed_stage(output, "temporal-full")
