"""QCeNN-MI Assumption A1 solver.

The first feasibility assumption is deliberately narrow and testable:

    A1. For a real, nonsingular 2x2 matrix M, each column of M^{-1} can be
        represented by a normalized one-qubit real-amplitude state, up to a
        classical scalar, and recovered by minimizing ||M x - e_j||_2.

This module uses a Qiskit-prepared RY state as the quantum cell state.  It is a
proof-of-feasibility building block, not a claim of quantum advantage and not
yet the full locally coupled QCeNN dynamical model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .quantum import shot_direction, statevector_direction

StateProvider = Callable[[float], np.ndarray]


@dataclass(frozen=True)
class ColumnResult:
    """Solution metadata for one inverse column."""

    column_index: int
    theta: float
    scale: float
    residual_norm: float
    state_direction: list[float]
    solution_column: list[float]


@dataclass(frozen=True)
class InversionResult:
    """Complete result for the 2x2 A1 feasibility experiment."""

    matrix: list[list[float]]
    inverse_estimate: list[list[float]]
    exact_inverse: list[list[float]]
    frobenius_residual: float
    relative_inverse_error: float
    condition_number: float
    columns: list[ColumnResult]
    backend: str
    shot_inverse_estimate: list[list[float]] | None = None
    shot_frobenius_residual: float | None = None
    shot_relative_inverse_error: float | None = None
    shots: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError("Assumption A1 currently supports real 2x2 matrices only.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix contains non-finite values")
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError("matrix must be nonsingular")
    return matrix


def _optimal_scale(matrix: np.ndarray, direction: np.ndarray, target: np.ndarray) -> float:
    """Least-squares scalar s minimizing ||M (s q) - b||_2."""
    transformed = matrix @ direction
    denominator = float(transformed @ transformed)
    if denominator <= np.finfo(float).eps:
        raise RuntimeError("Degenerate transformed quantum direction.")
    return float((transformed @ target) / denominator)


def _evaluate_direction(
    matrix: np.ndarray,
    direction: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    scale = _optimal_scale(matrix, direction, target)
    column = scale * direction
    residual = matrix @ column - target
    return float(residual @ residual), column, scale


def _solve_column(
    matrix: np.ndarray,
    column_index: int,
    *,
    state_provider: StateProvider,
    grid_points: int = 257,
) -> ColumnResult:
    if grid_points < 33:
        raise ValueError("grid_points must be >= 33 for a stable A1 search")

    target = np.eye(2, dtype=float)[:, column_index]

    def objective(theta: float) -> float:
        direction = state_provider(float(theta))
        cost, _, _ = _evaluate_direction(matrix, direction, target)
        return cost

    # q and -q are equivalent because the optimal classical scale may change
    # sign, so one 2*pi interval is sufficient for the real one-qubit ansatz.
    theta_grid = np.linspace(-np.pi, np.pi, grid_points)
    costs = np.array([objective(theta) for theta in theta_grid], dtype=float)
    best_index = int(np.argmin(costs))

    left_index = max(0, best_index - 1)
    right_index = min(grid_points - 1, best_index + 1)
    lower = float(theta_grid[left_index])
    upper = float(theta_grid[right_index])

    if lower == upper:
        theta = float(theta_grid[best_index])
    else:
        optimized = minimize_scalar(
            objective,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-12, "maxiter": 500},
        )
        theta = float(optimized.x)

    direction = state_provider(theta)
    cost, column, scale = _evaluate_direction(matrix, direction, target)

    return ColumnResult(
        column_index=column_index,
        theta=theta,
        scale=scale,
        residual_norm=float(np.sqrt(cost)),
        state_direction=direction.tolist(),
        solution_column=column.tolist(),
    )


def _metrics(
    matrix: np.ndarray,
    inverse_estimate: np.ndarray,
    exact_inverse: np.ndarray,
) -> tuple[float, float]:
    residual = float(np.linalg.norm(matrix @ inverse_estimate - np.eye(2), ord="fro"))
    relative_error = float(
        np.linalg.norm(inverse_estimate - exact_inverse, ord="fro")
        / np.linalg.norm(exact_inverse, ord="fro")
    )
    return residual, relative_error


def solve_inverse_2x2(
    matrix: np.ndarray,
    *,
    grid_points: int = 257,
    verification_shots: int | None = None,
    seed: int = 42,
) -> InversionResult:
    """Solve a real 2x2 inverse using the A1 QCeNN quantum-cell ansatz.

    Optimization uses Qiskit's exact statevector representation of the actual
    ``RY(theta)`` circuit.  If ``verification_shots`` is supplied, the final
    optimized states are re-read using finite-shot X/Z measurements on
    ``AerSimulator``.  This validates the hardware-compatible readout path.

    The exact NumPy inverse is computed only after solving, solely to report an
    independent accuracy metric.
    """
    matrix = _validate_matrix(matrix)

    columns = [
        _solve_column(
            matrix,
            column_index,
            state_provider=statevector_direction,
            grid_points=grid_points,
        )
        for column_index in range(2)
    ]

    inverse_estimate = np.column_stack(
        [np.asarray(column.solution_column, dtype=float) for column in columns]
    )
    exact_inverse = np.linalg.inv(matrix)
    residual, relative_error = _metrics(matrix, inverse_estimate, exact_inverse)

    shot_inverse: np.ndarray | None = None
    shot_residual: float | None = None
    shot_relative_error: float | None = None

    if verification_shots is not None:
        if verification_shots <= 0:
            raise ValueError("verification_shots must be positive")

        measured_columns: list[np.ndarray] = []
        for index, column in enumerate(columns):
            direction = shot_direction(
                column.theta,
                shots=verification_shots,
                seed=seed + index,
            )
            target = np.eye(2, dtype=float)[:, index]
            _, measured_column, _ = _evaluate_direction(matrix, direction, target)
            measured_columns.append(measured_column)

        shot_inverse = np.column_stack(measured_columns)
        shot_residual, shot_relative_error = _metrics(matrix, shot_inverse, exact_inverse)

    return InversionResult(
        matrix=matrix.tolist(),
        inverse_estimate=inverse_estimate.tolist(),
        exact_inverse=exact_inverse.tolist(),
        frobenius_residual=residual,
        relative_inverse_error=relative_error,
        condition_number=float(np.linalg.cond(matrix)),
        columns=columns,
        backend="qiskit-statevector",
        shot_inverse_estimate=None if shot_inverse is None else shot_inverse.tolist(),
        shot_frobenius_residual=shot_residual,
        shot_relative_inverse_error=shot_relative_error,
        shots=verification_shots,
    )
