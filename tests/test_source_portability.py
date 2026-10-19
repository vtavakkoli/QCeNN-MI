"""Protect frozen evidence across Windows Git checkouts and Docker builds."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docker/normalize_source.py"
PLAN = json.loads((ROOT / "publication-evidence/historical-reconciliation.json").read_text())
RECORDS = [PLAN[key] for key in ("public_archive", "manuscript_matching_run")]


def normalizer():
    spec = importlib.util.spec_from_file_location("normalize_source", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize_source


def test_reported_windows_checksum_is_exactly_crlf_conversion():
    record = PLAN["public_archive"]
    original = (ROOT / record["file"]).read_bytes()
    assert hashlib.sha256(original).hexdigest() == record["sha256"]
    assert hashlib.sha256(original.replace(b"\n", b"\r\n")).hexdigest() == (
        "2f6ef9ee6028049149f483f6273742cf869ee8729b514ff4500294ccea4b805b"
    )


def test_existing_crlf_checkout_restores_original_bytes_without_touching_binaries(tmp_path):
    shutil.copytree(ROOT / "publication-evidence", tmp_path / "publication-evidence")
    originals = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    for relative, data in originals.items():
        if relative.suffix in {".json", ".log"} or relative.name == "ORIGINAL_LICENSE":
            (tmp_path / relative).write_bytes(data.replace(b"\n", b"\r\n"))
    (tmp_path / "src").mkdir()
    (tmp_path / "src/module.py").write_bytes(b"value = 1\r\n")
    (tmp_path / "LICENSE").write_bytes(b"license text\r\n")
    # Even a binary containing CRLF must remain byte-identical.
    binary = tmp_path / "publication-evidence/sentinel.qpy"
    binary.write_bytes(b"QISKIT\x00\r\n\xff")
    output = tmp_path / "artifacts"
    output.mkdir()
    (output / "result.json").write_bytes(b'{"generated": true}\r\n')

    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "normalized" in result.stdout
    for relative, data in originals.items():
        assert (tmp_path / relative).read_bytes() == data
    for record in RECORDS:
        assert (
            hashlib.sha256((tmp_path / record["file"]).read_bytes()).hexdigest()
            == (record["sha256"])
        )
    assert (tmp_path / "src/module.py").read_bytes() == b"value = 1\n"
    assert (tmp_path / "LICENSE").read_bytes() == b"license text\n"
    assert binary.read_bytes() == b"QISKIT\x00\r\n\xff"
    assert (output / "result.json").read_bytes() == b'{"generated": true}\r\n'
    assert normalizer()(tmp_path) == []


@pytest.mark.parametrize("record", RECORDS, ids=["public-archive", "manuscript-run"])
def test_newline_normalization_does_not_accept_other_evidence_changes(tmp_path, record):
    path = tmp_path / record["file"]
    path.parent.mkdir(parents=True)
    original = (ROOT / record["file"]).read_bytes()
    # A valid JSON document with changed raw bytes must still fail the frozen hash.
    path.write_bytes(original.replace(b"\n", b"\r\n") + b" \r\n")
    normalizer()(tmp_path)
    json.loads(path.read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]


@pytest.mark.skipif(shutil.which("git") is None, reason="Git checkout tested in Windows CI")
def test_git_autocrlf_checkout_preserves_frozen_checksums(tmp_path):
    source, checkout = tmp_path / "source", tmp_path / "checkout"
    source.mkdir()
    shutil.copyfile(ROOT / ".gitattributes", source / ".gitattributes")
    for record in RECORDS:
        path = source / record["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / record["file"], path)
    binary = source / "circuit.qpy.gz"
    binary.write_bytes(b"\x1f\x8b\r\n\x00\xff")

    def git(*args):
        subprocess.run(["git", *args], check=True, capture_output=True)

    git("init", str(source))
    git("-C", str(source), "-c", "core.autocrlf=false", "add", ".")
    git(
        "-C",
        str(source),
        "-c",
        "user.name=Portability Test",
        "-c",
        "user.email=portability@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "Frozen evidence fixture",
    )
    git("-c", "core.autocrlf=true", "clone", "--no-hardlinks", str(source), str(checkout))
    for record in RECORDS:
        assert (
            hashlib.sha256((checkout / record["file"]).read_bytes()).hexdigest()
            == (record["sha256"])
        )
    assert (checkout / "circuit.qpy.gz").read_bytes() == binary.read_bytes()
