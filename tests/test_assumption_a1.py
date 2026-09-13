import numpy as np
import pytest

from qcenn_mi.quantum import statevector_direction
from qcenn_mi.solver import solve_inverse_2x2


def test_ry_quantum_cell_prepares_expected_real_state() -> None:
    theta = -0.7
    observed = statevector_direction(theta)
    expected = np.array([np.cos(theta / 2.0), np.sin(theta / 2.0)])
    assert observed == pytest.approx(expected, abs=1e-12)


def test_assumption_a1_recovers_reference_inverse() -> None:
    matrix = np.array([[2.0, 0.5], [0.5, 1.5]])
    result = solve_inverse_2x2(matrix, grid_points=257)

    estimate = np.asarray(result.inverse_estimate)
    expected = np.linalg.inv(matrix)

    assert estimate == pytest.approx(expected, abs=1e-8)
    assert result.frobenius_residual < 1e-7
    assert result.relative_inverse_error < 1e-7


def test_assumption_a1_supports_signed_inverse_entries() -> None:
    matrix = np.array([[2.0, 0.5], [0.5, 1.5]])
    result = solve_inverse_2x2(matrix)
    estimate = np.asarray(result.inverse_estimate)

    assert estimate[0, 1] < 0.0
    assert estimate[1, 0] < 0.0


def test_assumption_a1_rejects_singular_matrix() -> None:
    with pytest.raises(ValueError, match="nonsingular"):
        solve_inverse_2x2(np.array([[1.0, 2.0], [2.0, 4.0]]))


def test_assumption_a1_is_explicitly_2x2() -> None:
    with pytest.raises(ValueError, match="2x2"):
        solve_inverse_2x2(np.eye(3))
