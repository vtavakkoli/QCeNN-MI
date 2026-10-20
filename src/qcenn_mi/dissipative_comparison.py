"""Matched reduced-density comparisons with Shang, arXiv:2609.12284v1.

The baseline integrates the published Lindblad equation, with its jump rates
divided by two. All generators then have ||G|| <= 1. Continuous model time is
not a digital simulation query count or a hardware runtime. No classical solve
or observed solution enters a channel update or ODE right-hand side.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .recurrent_density import SpectralRecurrentChannel, build_density_channel


def shang_channel(matrix, rhs):
    """Encode F_S=(A,-beta)/sqrt(2), including an absorbing padding extension.

    On the active N+1 coordinates this is exactly Shang's generator / 2.
    Padding is initially empty in the ideal comparison. The reference weight
    is 1 rather than gamma; this changes the augmented target, not M^-1 b.
    """
    checked = build_density_channel(matrix, rhs)
    size, dimension = checked.size, checked.dimension
    residual = np.eye(dimension)
    residual[:size, :size] = checked.matrix / checked.matrix_scale
    residual[:size, size] = -checked.rhs / np.linalg.norm(checked.rhs)
    residual[size, :] = 0
    residual /= np.sqrt(2.0)
    _, singular, right = np.linalg.svd(residual)
    eigenvalues = singular**2
    basis = right.T
    padding = basis[size + 1 :, :]
    return SpectralRecurrentChannel(
        checked.matrix,
        checked.rhs,
        checked.matrix_scale,
        1.0,
        basis,
        eigenvalues,
        np.sqrt(np.maximum(0, 1 - eigenvalues)),
        basis[size, :].copy(),
        None if not len(padding) else padding.T @ padding,
    )


def lindblad_rhs(channel, density):
    """L(rho)=Tr(G rho)R - {G,rho}/2, in G's eigenbasis."""
    values = channel.residual_eigenvalues
    jump = np.dot(values, density.diagonal())
    return -0.5 * (values[:, None] + values[None, :]) * density + jump * np.outer(
        channel.reference, channel.reference
    )


def continuous_trajectory(channel, times, *, rtol=2e-10, atol=2e-12):
    """Numerically integrate the continuous semigroup, without Euler updates.

    DOP853 error tolerances are numerical controls, not an algorithmic noise
    model. Independent small Liouvillian exponentials test this implementation.
    """
    times = np.asarray(times, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or times[0] != 0
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("times must start at zero and increase strictly")
    dimension = channel.dimension
    rates = 0.5 * (channel.residual_eigenvalues[:, None] + channel.residual_eigenvalues[None, :])
    reset = np.outer(channel.reference, channel.reference)

    def rhs(_time, vector):
        density = vector.reshape(dimension, dimension)
        jump = np.dot(channel.residual_eigenvalues, density.diagonal())
        return (-rates * density + jump * reset).ravel()

    initial = channel.initial_density()
    result = solve_ivp(
        rhs,
        (0, times[-1]),
        initial.ravel(),
        method="DOP853",
        t_eval=times,
        rtol=rtol,
        atol=atol,
    )
    if not result.success:
        raise ArithmeticError(result.message)
    return result.y.T.reshape(-1, dimension, dimension), int(result.nfev)


def stationary_depolarized(channel, probability):
    """Exact fixed point of D_p o E for p>0; no reference solution is used.

    Solve the scalar reset flux and the entrywise fixed-point equation in the
    spectral basis. The returned density is not trace-renormalized.
    """
    if not np.isfinite(probability) or not 0 < probability <= 1:
        raise ValueError("stationary depolarization requires 0 < p <= 1")
    dimension = channel.dimension
    g = channel.residual_eigenvalues
    c = channel.no_jump_values
    r = channel.reference
    diagonal_denominator = probability + (1 - probability) * g
    reset_feedback = (1 - probability) * np.dot(g * r**2, 1 / diagonal_denominator)
    injection = probability / dimension * np.sum(g / diagonal_denominator)
    jump = injection / (1 - reset_feedback)
    # 1-(1-p)c_i*c_j is evaluated stably near the dark mode.
    denominator = probability + (1 - probability) * (1 - c[:, None] * c[None, :])
    density = (1 - probability) * jump * np.outer(r, r) / denominator
    density.flat[:: dimension + 1] += probability / dimension / diagonal_denominator
    return density


def change_channel_basis(old, new, density):
    """Represent the same physical density in a freshly compiled channel basis.

    This coordinate change is a simulator operation, not a quantum-state
    correction, measurement, warm start, or preparation of the target.
    """
    if old.dimension != new.dimension:
        raise ValueError("state-preserving basis changes require a fixed dimension")
    transition = new.basis.T @ old.basis
    return transition @ density @ transition.T
