"""Polynomial residual response with explicit dense quantum collision circuits.

This is not an oracle-efficient QSVT implementation. Dense SVDs compile the
input-dependent residual dilation without computing a solution or an inverse.
Large runs use the exact reduced channel and must be labeled as such.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .recurrent_density import SpectralRecurrentChannel, build_density_channel
from .recurrent_quantum import RecurrentQuantumSolver, _integer, build_recurrent_solver


def polynomial_response(eigenvalues, degree):
    """Return 1-(1-g)**(m+1), evaluated without cancellation at small g."""
    _integer(degree, "polynomial_degree")
    values = np.asarray(eigenvalues, dtype=float)
    if not np.isfinite(values).all() or np.any(values < -1e-12) or np.any(values > 1 + 1e-12):
        raise ValueError("residual eigenvalues must be in [0,1]")
    values = np.clip(values, 0, 1)
    if degree == 0:
        return values.copy()
    with np.errstate(divide="ignore"):
        return -np.expm1((degree + 1) * np.log1p(-values))


def accelerated_gap(singular_ratio, degree):
    if not np.isfinite(singular_ratio) or not 0 < singular_ratio <= 1:
        raise ValueError("singular_ratio must be in (0,1]")
    g = singular_ratio**2 / (1 + singular_ratio**2)
    return float(polynomial_response(g, degree))


@dataclass(frozen=True)
class PolynomialRecurrentSolver(RecurrentQuantumSolver):
    polynomial_degree: int

    @property
    def contraction_bound(self):
        return 1 - accelerated_gap(self.singular_ratio, self.polynomial_degree) / 2

    def certificate(self, steps, *, reference_initial_state=True):
        result = super().certificate(steps, reference_initial_state=reference_initial_state)
        result.update(
            polynomial_degree=self.polynomial_degree,
            residual_polynomial_order=self.polynomial_degree + 1,
            accelerated_residual_gap_lower_bound=accelerated_gap(
                self.singular_ratio, self.polynomial_degree
            ),
            efficient_polynomial_oracle_implemented=False,
        )
        return result


def build_polynomial_solver(matrix, rhs, degree=0, *, max_encoding_dimension=32):
    """Explicit Julia dilation of F_m=U sqrt(q_m(S**2)) V†.

    The baseline F=U S V† is constructed from M and b. q_m(g)=1-(1-g)^(m+1).
    The same environment/reference-swap circuit therefore realizes E_m. No
    target state, inverse, measured iterate or reference solve enters compilation.
    """
    _integer(degree, "polynomial_degree")
    base = build_recurrent_solver(matrix, rhs, max_encoding_dimension=max_encoding_dimension)
    if degree == 0:
        return PolynomialRecurrentSolver(
            *(getattr(base, f.name) for f in fields(RecurrentQuantumSolver)), degree
        )
    left, values, right_h = np.linalg.svd(base.residual_operator)
    response = polynomial_response(values**2, degree)
    roots = np.maximum(0, 1 - np.minimum(values**2, 1)) ** ((degree + 1) / 2)
    residual = (left * np.sqrt(response)) @ right_h
    no_jump = (right_h.T * roots) @ right_h
    other = (left * roots) @ left.T
    dilation = np.block([[no_jump, -residual.T], [residual, other]])
    if not np.allclose(dilation.T @ dilation, np.eye(len(dilation)), atol=5e-12, rtol=0):
        raise ArithmeticError("Polynomial collision dilation is not unitary")
    return PolynomialRecurrentSolver(
        base.matrix,
        base.rhs,
        base.matrix_scale,
        base.singular_ratio,
        residual,
        no_jump,
        dilation,
        degree,
    )


@dataclass(frozen=True)
class PolynomialDensityChannel(SpectralRecurrentChannel):
    polynomial_degree: int

    def certificate(self, steps, *, reference_initial_state=True):
        _integer(steps, "steps")
        theta = accelerated_gap(self.singular_ratio, self.polynomial_degree) / 2
        initial = 0.5 if reference_initial_state else 1.0
        error = initial * (1 - theta) ** steps
        return {
            "augmented_infidelity_upper_bound": error,
            "augmented_trace_distance_upper_bound": float(np.sqrt(error)),
            "infidelity_contraction_upper_bound": 1 - theta,
            "polynomial_degree": self.polynomial_degree,
        }


def accelerate_density_channel(channel, degree=0):
    """Transform only spectral responses; retain the physical basis/reference."""
    _integer(degree, "polynomial_degree")
    values = {f.name: getattr(channel, f.name) for f in fields(SpectralRecurrentChannel)}
    values["residual_eigenvalues"] = polynomial_response(channel.residual_eigenvalues, degree)
    values["no_jump_values"] = channel.no_jump_values ** (degree + 1)
    return PolynomialDensityChannel(**values, polynomial_degree=degree)


def build_polynomial_density_channel(matrix, rhs, degree=0):
    return accelerate_density_channel(build_density_channel(matrix, rhs), degree)
