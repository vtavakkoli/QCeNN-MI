"""Explicitly implicit classical motivation; these routines perform solves."""

from __future__ import annotations

import numpy as np

from .recurrent_quantum import _integer


def classical_factors(matrix, alpha, mu, degree):
    _integer(degree, "polynomial_degree")
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[-1] <= 0 or not np.isfinite(matrix).all():
        raise ValueError("matrix must be nonsingular and finite")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive")
    if not np.isfinite(mu) or not 0 < mu < 1 / singular[0] ** 2:
        raise ValueError("mu must satisfy 0 < mu < 1/||M||^2")
    gram = matrix @ matrix.T
    derivative_factor = np.eye(len(matrix)) - mu * gram
    weighting = np.linalg.matrix_power(np.eye(len(matrix)) + alpha * gram, degree)
    rate = (1 + alpha * singular[-1] ** 2) ** degree / (1 - mu * singular[-1] ** 2)
    return weighting, derivative_factor, float(rate)


def implicit_classical_velocity(matrix, estimate, alpha, mu, degree, matrix_derivative=None):
    """Solve M X' B = P(I-MX)-M'XB; not an inverse-free method."""
    matrix = np.asarray(matrix, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    weighting, derivative_factor, _ = classical_factors(matrix, alpha, mu, degree)
    residual = np.eye(len(matrix)) - matrix @ estimate
    right = weighting @ residual
    if matrix_derivative is not None:
        right -= np.asarray(matrix_derivative) @ estimate @ derivative_factor
    first = np.linalg.solve(matrix, right)
    return np.linalg.solve(derivative_factor.T, first.T).T
