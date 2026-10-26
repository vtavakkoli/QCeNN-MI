from pathlib import Path

import numpy as np
import pytest

from qcenn_mi.temporal_chebyshev_publication import (
    certified_degree,
    direct_postselect,
    independently_verify,
    matrix_family,
    response,
    run_case,
    run_grid,
)


def test_response_preserves_kernel_and_signed_modes():
    gap = 1 / 101
    degree = certified_degree(gap)
    values = response(np.r_[0, np.linspace(gap, 1, 100)], gap, degree)
    assert abs(values[0] - 1) < 1e-12
    assert max(abs(values[1:])) <= 0.1 + 1e-12
    assert min(values) < 0


def test_matrix_stays_conditioned_and_changes():
    first, rhs = matrix_family(4, 10, 23, 0, 6, "abrupt")
    last, other_rhs = matrix_family(4, 10, 23, 5, 6, "abrupt")
    assert np.isclose(np.linalg.cond(first), 10)
    assert np.isclose(np.linalg.cond(last), 10)
    assert not np.allclose(first, last)
    assert not np.allclose(rhs, other_rhs)


def test_matched_budget_retains_negative_outcomes_and_checks_normalization():
    result = run_case(size=2, snapshots=4, query_budget=32)
    assert set(result["summary"]) == {"resident", "fresh", "one_shot"}
    for item in result["trajectory"]:
        for obs in item["observations"].values():
            assert obs["trace_error"] < 1e-9
            assert obs["minimum_eigenvalue"] > -1e-9
            assert obs["charged_queries_per_snapshot"] <= 32
        assert item["observations"]["one_shot"]["expected_queries_one_accepted_preparation"] >= 32
    assert independently_verify(result)["status"] == "passed"


def test_fresh_and_resident_identical_at_first_snapshot_not_forced_later():
    result = run_case(size=2, snapshots=6, query_budget=32, drift="abrupt")
    first = result["trajectory"][0]["observations"]
    assert np.isclose(first["resident"]["relative_residual"], first["fresh"]["relative_residual"])
    values = [row["observations"] for row in result["trajectory"][1:]]
    assert any(abs(row["resident"]["relative_residual"] -
                   row["fresh"]["relative_residual"]) > 1e-10 for row in values)


def test_one_shot_postselection_and_query_coupled_noise():
    rho = np.diag([0.0, 1.0])
    B = np.diag([0.5, 0.8])
    out, success = direct_postselect(B, rho, queries=4, noise_per_query=1e-3)
    assert np.isclose(success, 0.64)
    assert np.isclose(np.trace(out), 1)
    assert 0 < out[0, 0] < 0.5


def test_fresh_directory_and_budget_guards(tmp_path: Path):
    run_grid(output_dir=tmp_path / "evidence", sizes=(2,), conditions=(2,),
             budgets=(32,), drifts=("slow",), noises=(0,), seeds=(23,), snapshots=3)
    with pytest.raises(FileExistsError):
        run_grid(output_dir=tmp_path / "evidence", sizes=(2,), conditions=(2,),
                 budgets=(32,), drifts=("slow",), noises=(0,), seeds=(23,))
    with pytest.raises(ValueError):
        run_case(condition=10, query_budget=2)
