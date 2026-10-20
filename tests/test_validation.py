import json
import os
import sys
import zipfile

import numpy as np
import pytest

from qcenn_mi.compact import build_compact_inverse
from qcenn_mi.precision_benchmark import audit_operator, controlled_fixture
from qcenn_mi.validation import (
    create_bundle,
    execute_stage,
    freeze_source,
    provenance,
    source_fingerprint,
    stage_commands,
)


def test_suite_contains_every_existing_campaign_and_1000_precision(tmp_path):
    commands = dict(stage_commands(tmp_path))
    assert set(commands) == {
        "runtime-smoke",
        "temporal-refresh",
        "noise-verification",
        "baselines",
        "resources",
        "operator-audit",
        "recurrent-quantum",
        "recurrent-scaling",
        "dissipative-comparison",
        "compile",
        "lint",
        "unit-tests",
        "assumption-a1",
        "a2-a4",
        "scaling",
        "coherent",
        "coherent-large",
        "compact",
        "precision-1000",
        "theory",
        "time-varying",
        "hardware",
    }
    assert "qcenn_mi.precision_benchmark" in commands["precision-1000"]


def test_stage_failure_is_retained_with_logs_and_metrics(tmp_path):
    row = execute_stage(
        "fails",
        [sys.executable, "-c", "print('evidence'); raise SystemExit(3)"],
        tmp_path,
        env=os.environ.copy(),
    )
    assert row["returncode"] == 3
    assert not row["passed"]
    assert "evidence" in (tmp_path / row["log"]).read_text()
    assert row["wall_seconds"] > 0
    assert json.loads((tmp_path / "logs/fails.metrics.json").read_text())["returncode"] == 3


def test_fingerprint_changes_when_the_solver_changes(tmp_path):
    (tmp_path / "src").mkdir()
    source = tmp_path / "src/module.py"
    source.write_text("a = 1\n")
    initial = source_fingerprint(tmp_path)
    source.write_text("a = 2\n")
    assert source_fingerprint(tmp_path) != initial


def test_frozen_source_contains_every_fingerprinted_byte(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/module.py").write_text("a = 1\n")
    output = tmp_path / "run"
    output.mkdir()
    inventory = freeze_source(tmp_path, output)
    with zipfile.ZipFile(output / "source-snapshot.zip") as archive:
        assert archive.read("QCeNN-MI/src/module.py") == b"a = 1\n"
    assert inventory["source_sha256"] == source_fingerprint(tmp_path)
    (tmp_path / "src/module.py").write_text("a = 2\n")
    assert inventory["source_sha256"] != source_fingerprint(tmp_path)


def test_pinned_build_rejects_changed_source_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("QCENN_EXPECTED_SOURCE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="Source bytes differ"):
        provenance(tmp_path)


def test_checkout_policy_is_fingerprinted_and_included_in_source_bundle(tmp_path):
    attributes = tmp_path / ".gitattributes"
    attributes.write_bytes(b"* text=auto eol=lf\n")
    initial = source_fingerprint(tmp_path)
    attributes.write_bytes(attributes.read_bytes() + b"*.gz binary\n")
    assert source_fingerprint(tmp_path) != initial
    output = tmp_path / "artifacts"
    output.mkdir()
    with zipfile.ZipFile(create_bundle(tmp_path, output)) as archive:
        assert archive.read("QCeNN-MI/.gitattributes") == attributes.read_bytes()


def test_bundle_contains_source_and_evidence_without_archiving_itself(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.py").write_text("x = 1\n")
    output = tmp_path / "artifacts"
    output.mkdir()
    (output / "validation.json").write_text('{"passed": true}\n')
    bundle = create_bundle(tmp_path, output)
    with zipfile.ZipFile(bundle) as archive:
        assert "QCeNN-MI/src/a.py" in archive.namelist()
        assert "QCeNN-MI/evidence/validation.json" in archive.namelist()
        assert not any(name.endswith(".zip") for name in archive.namelist())
    assert (output / "PACKAGE_SHA256").is_file()


@pytest.mark.parametrize(
    "family", ["controlled_spd", "controlled_nonsymmetric", "controlled_indefinite"]
)
def test_theory_operator_and_normalized_state_bounds(family):
    matrix = controlled_fixture(4, family, 10)
    operation = build_compact_inverse(matrix, degree=15)
    audit = audit_operator(operation)
    rhs = np.arange(1.0, 5.0)
    result = operation.simulate(rhs)
    reference = np.linalg.solve(matrix, rhs)
    relative = np.linalg.norm(result["solution"] - reference) / np.linalg.norm(reference)
    state_error = np.linalg.norm(
        result["solution"] / np.linalg.norm(result["solution"])
        - reference / np.linalg.norm(reference)
    )
    assert audit["operator_relative_residual_2"] <= operation.residual_bound + 1e-12
    assert relative <= operation.residual_bound + 1e-12
    assert state_error <= 2 * operation.residual_bound + 1e-12
    assert operation.normalization <= 1 / np.min(operation.encoding.singular_values) + 1e-12


def test_generated_build_metadata_does_not_change_source_or_enter_the_bundle(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.py").write_text("x = 1\n")
    initial = source_fingerprint(tmp_path)
    metadata = tmp_path / "src/package.egg-info"
    metadata.mkdir()
    (metadata / "PKG-INFO").write_text("generated install metadata\n")
    assert source_fingerprint(tmp_path) == initial
    output = tmp_path / "evidence"
    output.mkdir()
    bundle = create_bundle(tmp_path, output)
    with zipfile.ZipFile(bundle) as archive:
        assert not any(".egg-info/" in name for name in archive.namelist())
