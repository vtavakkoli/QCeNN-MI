"""A3 stateful hybrid tracking and A4 classical polynomial residual operators."""

from __future__ import annotations

import numpy as np

from .coupled import solve_coupled, validate_matrix


def polynomial_correct(matrix, estimate, *, order=2, max_backtracks=20):
    """X' = X sum_{k=0}^order (I-MX)^k, with residual-decrease safeguard.

    In exact arithmetic the new residual is E**(order+1), E=I-MX.
    This is a classical Neumann/Newton-Schulz inspired correction, not a
    block-encoded quantum polynomial operator or a reproduction of the paper.
    Backtracking rejects nonfinite/non-improving proposals for large residuals.
    """
    m = validate_matrix(matrix)
    x = np.asarray(estimate, dtype=float)
    if x.shape != m.shape or not np.isfinite(x).all():
        raise ValueError("Estimate must be finite and match matrix shape")
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise ValueError("order must be a positive integer")
    if not isinstance(max_backtracks, int) or max_backtracks < 0:
        raise ValueError("max_backtracks must be a nonnegative integer")
    identity = np.eye(len(m))
    e = identity - m @ x
    power = identity.copy()
    series = identity.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(order):
            power = power @ e
            series += power
        delta = x @ series - x
        baseline = np.linalg.norm(e)
        for attempt in range(max_backtracks + 1):
            candidate = x + delta * 2.0 ** (-attempt)
            if np.isfinite(candidate).all():
                if np.linalg.norm(identity - m @ candidate) <= baseline:
                    return candidate
    return x.copy()


class InverseTracker:
    """Persist circuit parameters and corrected classical estimate across M(t).

    Prediction uses -X dM X (first-order inverse derivative); correction is
    classical. Each new matrix also receives a warm-started quantum variational
    solve. Both paths and their residuals remain distinct in the returned record.
    State is committed only after successful validation and quantum convergence.
    """

    def __init__(self, *, depth=2, order=2, max_nfev=400, tolerance=1e-7):
        self.depth = depth
        self.order = order
        self.max_nfev = max_nfev
        self.tolerance = tolerance
        self.state = None
        self.estimate = None
        self.time = None

    def update(self, matrix, *, time):
        m = validate_matrix(matrix)
        if not np.isfinite(time) or (self.time is not None and time <= self.time):
            raise ValueError("Times must be finite and strictly increasing")
        if self.state is not None and m.shape != self.state.matrix.shape:
            raise ValueError("Tracking matrix size cannot change")
        predicted = None
        if self.state is not None:
            predicted = self.estimate - self.estimate @ (m - self.state.matrix) @ self.estimate
            predicted = polynomial_correct(m, predicted, order=self.order)
        solved = solve_coupled(
            m,
            depth=self.depth,
            initial=self.state,
            max_nfev=self.max_nfev,
            tolerance=self.tolerance,
        )
        if not solved.success:
            raise RuntimeError(f"Quantum tracking solve did not converge: {solved.residual}")
        corrected = polynomial_correct(m, solved.estimate, order=self.order)
        if predicted is not None:
            if np.linalg.norm(m @ predicted - np.eye(len(m))) < np.linalg.norm(
                m @ corrected - np.eye(len(m))
            ):
                corrected = predicted
        record = {
            "time": float(time),
            "quantum_residual": solved.residual,
            "corrected_residual": float(np.linalg.norm(m @ corrected - np.eye(len(m)))),
            "prediction_residual": None
            if predicted is None
            else float(np.linalg.norm(m @ predicted - np.eye(len(m)))),
            "evaluations": solved.evaluations,
            "warm_started": self.state is not None,
        }
        self.state, self.estimate, self.time = solved, corrected.copy(), float(time)
        return record
