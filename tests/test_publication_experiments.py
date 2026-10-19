import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import linalg, optimize

from qcenn_mi.baseline_benchmark import classical_solve
from qcenn_mi.compact import build_compact_inverse
from qcenn_mi.evidence_report import csv_text, verify
from qcenn_mi.experiment_io import configuration
from qcenn_mi.noise_verification_benchmark import statistics
from qcenn_mi.operator_audit import audit_case, matrix_fixture
from qcenn_mi.temporal_refresh_benchmark import campaign, sufficient_age
from qcenn_mi.time_varying_benchmark import tracking_bound


@pytest.mark.parametrize("size", [2, 4, 10])
@pytest.mark.parametrize("family", ["nonnormal", "signed_indefinite", "repeated_singular_values"])
def test_unfused_native_operator_matches_every_sector_and_unseen_rhs(size, family):
    result = audit_case(size, family)
    assert result["passed"]
    assert len([s for s in result["samples"] if s["unseen_rhs"]]) == 3
    assert result["left_operator_residual"] < 2e-8
    assert result["right_operator_residual"] < 2e-8


@pytest.mark.parametrize("size", [3, 5, 7])
def test_random_nonnormal_two_sided_operator_bound(size):
    from qcenn_mi.time_varying_benchmark import simulated_operator

    matrix = matrix_fixture(size, "nonnormal", 100 + size)
    assert np.linalg.norm(matrix @ matrix.T - matrix.T @ matrix) > 0.1
    op = build_compact_inverse(matrix, tolerance=1e-10)
    inverse = simulated_operator(op)
    assert np.linalg.norm(matrix @ inverse - np.eye(size), 2) <= op.residual_bound + 1e-10
    assert np.linalg.norm(inverse @ matrix - np.eye(size), 2) <= op.residual_bound + 1e-10


def test_coherent_runtime_cannot_call_reference_solvers_or_feedback(monkeypatch):
    matrix = matrix_fixture(3, "nonnormal", 17)

    def forbidden(*args, **kwargs):
        raise AssertionError("Reference solve or residual optimization entered coherent runtime")

    for module, names in [
        (np.linalg, ["solve", "inv", "lstsq"]),
        (linalg, ["solve", "inv", "lstsq", "solve_triangular"]),
        (optimize, ["minimize", "minimize_scalar"]),
    ]:
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    from qcenn_mi import tracking

    monkeypatch.setattr(tracking, "polynomial_correct", forbidden)
    monkeypatch.setattr(np.polynomial.chebyshev, "chebval", forbidden)
    op = build_compact_inverse(matrix, tolerance=1e-8)
    operator = op.operator_circuit()
    assert operator.num_qubits > 0
    for rhs in [np.ones(3), np.array([-1.0, 0.3, 2.0])]:
        assert op.simulate(rhs)["relative_residual"] < 1e-8


def test_dimension_aware_rank_threshold_is_distinct_from_degree_budget():
    size = 3
    threshold = np.finfo(float).eps * size
    with pytest.raises(ValueError, match="singular"):
        build_compact_inverse(np.diag([1, 1, threshold / 2]))
    with pytest.raises(ValueError, match="budget"):
        build_compact_inverse(np.diag([1, 1, threshold * 2]))


def test_refresh_sufficient_interval_includes_nonzero_acquisition_error():
    delta, gap, lipschitz, noise, target = 1e-10, 0.1, 2.0, 1e-5, 1e-3
    age = sufficient_age(target, delta, gap, lipschitz, noise)
    assert age > 0
    assert tracking_bound(delta, gap, lipschitz * age, noise) == pytest.approx(target)
    assert sufficient_age(1e-6, delta, gap, lipschitz, noise) is None


def test_refresh_study_has_positive_and_failed_accuracy_regimes(tmp_path):
    result = campaign(
        tmp_path,
        sizes=[3],
        conditions=[10],
        families=["fast_nonsymmetric"],
        ages=[0, 1e-8, 1e-4, 0.1],
        update_times=[0],
        noise_to_gap_ratios=[0, 1e-5],
    )
    assert result["passed"]
    assert result["positive_accuracy_cases"] > 0
    assert result["intentional_accuracy_failures"] > 0
    assert all(x["sufficient_condition_validated"] for x in result["refresh_intervals"])


def test_echo_statistics_do_not_confuse_success_and_good_output():
    result = statistics({"0000": 50, "0001": 10, "1000": 40}, 2, 4)
    assert result["success_frequency"] == pytest.approx(0.6)
    assert result["observed_echo_return"] == pytest.approx(5 / 6)
    assert result["effective_good_output_probability"] == pytest.approx(0.5)
    assert result["echo_return_95ci"][0] < 5 / 6 < result["echo_return_95ci"][1]
    assert not result["quality_gate"]


@pytest.mark.parametrize("method", ["scipy_solve", "lu", "qr", "svd"])
def test_classical_baselines_solve_the_same_nonnormal_system(method):
    matrix = matrix_fixture(7, "nonnormal", 19)
    rhs = np.arange(1.0, 8)
    result = classical_solve(matrix, rhs, method)
    assert np.linalg.norm(matrix @ result - rhs) < 1e-12


def test_configuration_rejects_unknown_fields(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"schema_version": 1, "study": {"unexpected": 2}}))
    with pytest.raises(ValueError, match="Unknown"):
        configuration(path, "study", {"sizes": [2]})


def test_forged_table_fails_even_if_its_checksum_is_updated(tmp_path):
    (tmp_path / "hardware").mkdir()
    data = {
        "cases": [
            {
                "size": 2,
                "required_qubits": 5,
                "native_action_depth": 60,
                "native_action_operations": {"cx": 20},
                "native_ideal_residual": 1e-12,
            }
        ]
    }
    (tmp_path / "hardware/result.json").write_text(json.dumps(data))
    (tmp_path / "tables").mkdir()
    table = tmp_path / "tables/native-resources.csv"
    table.write_text(csv_text(tmp_path, "native-resources"))
    manifest = {
        "schema_version": 1,
        "files": {"tables/native-resources.csv": hashlib.sha256(table.read_bytes()).hexdigest()},
        "tables": [{"id": "native-resources", "file": "tables/native-resources.csv"}],
    }
    (tmp_path / "publication-evidence").mkdir()
    path = tmp_path / "publication-evidence/manifest.json"
    path.write_text(json.dumps(manifest))
    assert verify(tmp_path)
    table.write_text(table.read_text().replace(",60,", ",999,"))
    manifest["files"]["tables/native-resources.csv"] = hashlib.sha256(
        table.read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="metrics disagree"):
        verify(tmp_path)


def test_both_historical_runs_and_manuscript_metrics_remain_distinct():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "publication-evidence/historical-reconciliation.json").read_text())
    for key in ["public_archive", "manuscript_matching_run"]:
        record = plan[key]
        assert hashlib.sha256((root / record["file"]).read_bytes()).hexdigest() == record["sha256"]
    clean = json.loads((root / plan["manuscript_matching_run"]["file"]).read_text())
    assert [r["native_action_depth"] for r in clean["cases"]] == [66, 138, 1032, 41033]
    profile = clean["cases"][-1]["noise"][0]
    assert profile["accepted"] == 35
    assert profile["observed_echo_return"] == pytest.approx(10 / 35)
    assert not plan["investigation"]["historical_files_overwritten"]


@pytest.mark.parametrize("successes,total", [(0, 100), (100, 100), (1, 3)])
def test_wilson_interval_contains_its_observed_proportion(successes, total):
    from qcenn_mi.hardware import wilson

    lo, hi = wilson(successes, total)
    assert 0 <= lo <= successes / total <= hi <= 1


def test_historical_circuit_archive_matches_the_reported_native_depths():
    import gzip
    import io

    from qiskit import qpy

    root = (
        Path(__file__).resolve().parents[1]
        / "publication-evidence/historical-runs/e6769dbe-clean-benchmark"
    )
    records = json.loads((root / "CIRCUITS.json").read_text())
    expected = {2: 66, 4: 138, 10: 1032, 100: 41033}
    for record in records:
        raw = gzip.decompress((root / record["file"]).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == record["raw_qpy_sha256"]
        action, verifier = qpy.load(io.BytesIO(raw))
        size = int(Path(record["file"]).name.split("-")[0])
        assert action.depth() == expected[size]
        assert verifier.num_clbits == verifier.num_qubits
