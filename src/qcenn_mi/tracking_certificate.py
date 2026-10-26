"""Rigorous moving-target bound for signed-filter CPTP reset channels.

For every snapshot t, B_t is a Hermitian contraction with B_t |z_t> = |z_t>,
||B_t (I-Pi_t)|| <= s_t < 1, Pi_t = |z_t><z_t|.
E_t(rho) = B_t rho B_t + Tr[(I-B_t**2)rho] R has fixed point Pi_t
and contracts target infidelity by a_t = 1-q_t(1-s_t**2), with
q_t = Tr(Pi_t R). This statement does not require trace-norm contraction.

A noisy implementation is assumed CPTP with uniform per-macro half-trace-norm
deviation <= tau_t. Delta_t = ||Pi_t - Pi_(t-1)||_infinity.
After k_t updates:
e_t <= a_t**k_t * min(1, e_(t-1) + Delta_t)
       + tau_t * sum_(j=0)^(k_t-1) a_t**j.
The first step uses e_prior=1-q_0 for the reference initial state.

The resident/fresh comparison is a comparison of *upper certificates* only:
resident certificate is tighter if the prior bound plus target drift is below
1-q_t, not a proof of actual performance superiority or quantum speedup.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackingStep:
    degree: int
    gap: float
    updates: int
    reference_overlap: float
    drift_projector_norm: float
    half_trace_noise_per_macro: float
    resident_bound: float
    fresh_bound: float
    alpha: float
    spectral_suppression: float
    certification_advantage: bool


def chebyshev_suppression(gap: float, degree: int) -> float:
    """Upper bound sech(2 degree atanh sqrt(gap)), avoiding cosh overflow."""
    if not 0 < gap < 1 or isinstance(degree, bool) or degree < 1:
        raise ValueError("gap in (0,1) and positive integer degree required")
    if not isinstance(degree, int):
        raise ValueError("degree must be an integer")
    x = 2 * degree * np.arctanh(np.sqrt(gap))
    return float(2 * np.exp(-x) / (1 + np.exp(-2 * x)))


def projector_drift(previous: np.ndarray | None, current: np.ndarray) -> float:
    """Exact operator norm of the difference of two rank-one projectors."""
    value = np.asarray(current, dtype=complex)
    if value.ndim != 1 or not np.isclose(np.linalg.norm(value), 1, atol=1e-9):
        raise ValueError("current target must be a unit vector")
    if previous is None:
        return 0.0
    old = np.asarray(previous, dtype=complex)
    if old.shape != value.shape or not np.isclose(np.linalg.norm(old), 1, atol=1e-9):
        raise ValueError("previous target must have the same normalized shape")
    fidelity = float(np.clip(abs(np.vdot(old, value)) ** 2, 0, 1))
    return float(np.sqrt(max(0, 1 - fidelity)))


def noise_budget_per_macro(p_per_query: float, calls: int, dimension: int) -> float:
    """Worst-case half trace norm of depolarization vs ideal output.

    This is a *logical per-oracle depolarization model*, not a native gate model.
    """
    if not 0 <= p_per_query < 1 or calls < 0 or dimension < 2:
        raise ValueError("invalid logical noise model")
    p_macro = -np.expm1(calls * np.log1p(-p_per_query))
    return float(p_macro * (1 - 1 / dimension))


def geometric_factor(alpha: float, updates: int) -> float:
    if not 0 <= alpha <= 1 or isinstance(updates, bool) or updates < 1:
        raise ValueError("invalid contraction factor or update count")
    return float(updates if alpha == 1 else -np.expm1(updates * np.log(alpha)) / (1 - alpha))


def certify_step(*, degree: int, gap: float, updates: int,
                 reference_overlap: float, drift: float,
                 per_macro_error: float, previous_bound: float | None) -> TrackingStep:
    """Compute sufficient target-infidelity certificates with honest scope."""
    if not 0 < reference_overlap <= 1 or not 0 <= drift <= 1:
        raise ValueError("reference overlap and drift are probabilities/norms")
    if not 0 <= per_macro_error <= 1:
        raise ValueError("half trace implementation error must be in [0,1]")
    if previous_bound is not None and not 0 <= previous_bound <= 1:
        raise ValueError("previous bound must be a valid infidelity bound")
    sigma = chebyshev_suppression(gap, degree)
    alpha = 1 - reference_overlap * (1 - sigma**2)
    residual = alpha**updates
    additive = per_macro_error * geometric_factor(alpha, updates)
    prior = (1 - reference_overlap) if previous_bound is None else min(
        1.0, previous_bound + drift
    )
    fresh_prior = 1 - reference_overlap
    return TrackingStep(
        degree=degree, gap=float(gap), updates=updates,
        reference_overlap=float(reference_overlap),
        drift_projector_norm=float(drift),
        half_trace_noise_per_macro=float(per_macro_error),
        resident_bound=float(min(1, residual * prior + additive)),
        fresh_bound=float(min(1, residual * fresh_prior + additive)),
        alpha=float(alpha), spectral_suppression=float(sigma),
        certification_advantage=bool(
            previous_bound is not None and previous_bound + drift < fresh_prior
        ),
    )
