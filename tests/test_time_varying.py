import numpy as np
import pytest

from qcenn_mi.coherent_large import build_large_inverse_action
from qcenn_mi.compact import build_compact_inverse
from qcenn_mi.time_varying_benchmark import (
    campaign,
    simulated_operator,
    tracking_bound,
    trajectory_bases,
    trajectory_matrix,
)


@pytest.mark.parametrize("size", [3, 7])
@pytest.mark.parametrize("family", ["slow_spd", "fast_nonsymmetric"])
def test_rotating_non_power_of_two_trajectory_has_prescribed_condition(size, family):
    bases = trajectory_bases(size)
    for time in [0, 0.37, 1]:
        matrix = trajectory_matrix(time, family, bases)
        assert np.linalg.cond(matrix) == pytest.approx(10 ** (1 + 2 * time), rel=1e-10)
        if family == "slow_spd":
            assert np.allclose(matrix, matrix.T)
            assert np.min(np.linalg.eigvalsh(matrix)) > 0


def test_drift_and_acquisition_bound_for_noncommuting_operators():
    bases = trajectory_bases(3)
    original = trajectory_matrix(0.7, "fast_nonsymmetric", bases)
    current = trajectory_matrix(0.71, "fast_nonsymmetric", bases)
    minimum = float(np.min(np.linalg.svd(original, compute_uv=False)))
    acquisition = np.diag([0.01, -0.02, 0.03]) * minimum
    noisy = original + acquisition
    operation = build_compact_inverse(noisy, tolerance=1e-12)
    inverse = simulated_operator(operation)
    bound = tracking_bound(
        operation.residual_bound,
        minimum,
        float(np.linalg.norm(current - original, 2)),
        float(np.linalg.norm(acquisition, 2)),
    )
    assert np.linalg.norm(current @ inverse - np.eye(3), 2) <= bound + 1e-10
    assert np.linalg.norm(inverse @ current - np.eye(3), 2) <= bound + 1e-10
    rhs = np.array([1.0, -2.0, 3.0])
    actual = inverse @ rhs
    reference = np.linalg.solve(current, rhs)
    assert np.linalg.norm(actual - reference) / np.linalg.norm(reference) <= bound + 1e-10
    state_error = np.linalg.norm(
        actual / np.linalg.norm(actual) - reference / np.linalg.norm(reference)
    )
    assert state_error <= 2 * bound + 1e-10


@pytest.mark.parametrize("family", ["slow_spd", "fast_nonsymmetric"])
def test_dense_rank_loss_is_reported_before_the_degree_budget(family):
    matrix = trajectory_matrix(1, family, trajectory_bases(100))
    left, values, right = np.linalg.svd(matrix)
    rank_loss_limit = matrix - values[-1] * np.outer(left[:, -1], right[-1])
    with pytest.raises(ValueError, match="numerically nonsingular"):
        build_compact_inverse(rank_loss_limit, tolerance=1e-12)
    # A resolved positive singular value is not rejected solely for being small.
    resolved = np.diag(np.concatenate([np.ones(99), [1e-12]]))
    assert build_large_inverse_action(resolved, degree=1).condition_number > 1e11


@pytest.mark.parametrize(
    "parameters",
    [
        (-1, 1, 0, 0),
        (1, 1, 0, 0),
        (0, 0, 0, 0),
        (0, 1, -1, 0),
        (0, 1, 0, -1),
        (0, 1, 0, 1),
        (0, np.nan, 0, 0),
    ],
)
def test_tracking_bound_rejects_missing_gap_or_invalid_noise(parameters):
    with pytest.raises(ValueError):
        tracking_bound(*parameters)


def test_temporal_campaign_records_degradation_without_hiding_it(tmp_path):
    result = campaign(tmp_path, sizes=(2, 3), times=(0.0, 0.5, 1.0))
    assert result["passed"]
    assert len(result["snapshots"]) == 12
    assert len(result["held_operator_pairs"]) == 8
    assert len(result["acquisition_errors"]) == 8
    assert all(r["explicitly_rejected"] for r in result["singular_endpoints"])
    assert max(r["left_residual"] for r in result["held_operator_pairs"]) > 1
    assert min(r["operator_residual"] for r in result["acquisition_errors"]) > 1e-4
    assert not result["physical_hardware_tested"]
    assert not result["real_time_deadline_tested"]
