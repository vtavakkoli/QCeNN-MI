import numpy as np
import pytest
from scipy.linalg import expm

from qcenn_mi.classical_polynomial import classical_factors, implicit_classical_velocity


def test_noncommuting_residual_has_proved_exponential_rate():
    matrix = np.array([[1.4, 0.2], [-0.3, 0.6]])
    residual = np.array([[0.2, 0.7], [-0.5, 0.3]])
    weighting, factor, rate = classical_factors(matrix, 0.5, 0.1, 4)
    inverse_factor = np.linalg.solve(factor, np.eye(2))
    generator = -np.kron(inverse_factor.T, weighting)
    for time in [0, 0.1, 1, 3]:
        evolved = (expm(generator * time) @ residual.ravel(order="F")).reshape(2, 2, order="F")
        assert np.linalg.norm(evolved) <= np.exp(-rate * time) * np.linalg.norm(residual) + 1e-14


def test_time_derivative_compensation_and_static_equation():
    matrix = np.array([[1.4, 0.2], [-0.3, 0.6]])
    estimate = np.array([[0.2, 0.3], [0.1, 0.7]])
    derivative = np.array([[0.1, -0.3], [0.2, 0.1]])
    weighting, factor, _ = classical_factors(matrix, 0.5, 0.1, 2)
    residual = np.eye(2) - matrix @ estimate
    velocity = implicit_classical_velocity(matrix, estimate, 0.5, 0.1, 2, derivative)
    expected = -weighting @ residual @ np.linalg.solve(factor, np.eye(2))
    assert np.allclose(-derivative @ estimate - matrix @ velocity, expected)


def test_unscaled_derivative_factor_can_be_singular_or_unstable():
    assert np.linalg.matrix_rank(np.eye(2) - np.eye(2)) == 0
    factor = np.eye(2) - np.diag([2.0, 0.5]) ** 2
    assert np.linalg.eigvalsh(factor)[0] < 0
    with pytest.raises(ValueError):
        classical_factors(np.eye(2), 1, 1, 0)
