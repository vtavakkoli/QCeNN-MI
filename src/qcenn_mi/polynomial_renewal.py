"""Exact renewal representation of static collision-channel observables.

The diagonal evolves by the channel's rank-one reset. Off-diagonal reference
coherences are reconstructed by the exact reset-flux convolution, accelerated
classically by FFT. No solution estimate drives evolution. This is a simulator,
not a quantum algorithm, and does not form the whole large density matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft


def renewal_observables(channel, steps, times, initial, reference_solution, probability=0.0):
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0,1]")
    times = np.asarray(times, dtype=int)
    if steps < 1 or np.any(times < 0) or np.any(times > steps):
        raise ValueError("observation times exceed the update budget")
    dimension = channel.dimension
    c, g, r = channel.no_jump_values, channel.residual_eigenvalues, channel.reference
    mixed = initial == "mixed"
    if initial == "reference":
        v = r.copy()
    elif initial == "basis":
        v = channel.basis[0, :].copy()
    elif initial == "random":
        physical = np.zeros(dimension)
        physical[: channel.size + 1] = np.random.default_rng(831 + channel.size).normal(
            size=channel.size + 1
        )
        physical /= np.linalg.norm(physical)
        v = channel.basis.T @ physical
    elif mixed:
        v = None
    else:
        raise ValueError("initial must be reference, basis, random or mixed")
    diagonal = np.full(dimension, 1 / dimension) if mixed else v**2
    recorded = {}
    flux = np.empty(steps)
    wanted = set(times)
    a = 1 - probability
    for k in range(steps + 1):
        if k in wanted:
            recorded[k] = diagonal.copy()
        if k < steps:
            flux[k] = np.dot(g, diagonal)
            diagonal = a * (c**2 * diagonal + flux[k] * r**2) + probability / dimension
    powers = np.arange(steps)
    moments = np.zeros(steps)
    for start in range(0, dimension, 32):
        sl = slice(start, min(start + 32, dimension))
        moments += r[sl] ** 2 @ (c[sl, None] ** powers)
    fft_size = next_fast_len(2 * steps - 1)
    reset_fft = rfft(flux, n=fft_size)
    columns = np.zeros((dimension, len(times)))
    selected = times > 0
    for start in range(0, dimension, 32):
        sl = slice(start, min(start + 32, dimension))
        kernels = (a * c[sl, None]) ** powers * (a * moments)
        convolved = irfft(rfft(kernels, n=fft_size, axis=1) * reset_fft, n=fft_size, axis=1)
        columns[sl, selected] = r[sl, None] * convolved[:, times[selected] - 1]
        if mixed:
            columns[sl] += (a * c[sl, None] ** 2) ** times * r[sl, None] / dimension
        else:
            ck = c[sl, None] ** times
            overlaps = (v * r) @ (c[:, None] ** times)
            columns[sl] += a**times * ck * v[sl, None] * overlaps
        denominator = 1 - a * c[sl] ** 2
        geometric = np.empty((min(32, dimension - start), len(times)))
        mask = abs(denominator) > 1e-15
        geometric[mask] = (1 - (a * c[sl][mask, None] ** 2) ** times) / denominator[mask, None]
        geometric[~mask] = times
        columns[sl] += probability / dimension * r[sl, None] * geometric
    dark = int(np.argmin(g))
    q = r[dark] ** 2
    sign = 1.0 if r[dark] >= 0 else -1.0
    pad_size = dimension - channel.size - 1
    degree = getattr(channel, "polynomial_degree", 0)
    padding_c2 = (channel.singular_ratio**2 / (1 + channel.singular_ratio**2)) ** (degree + 1)
    rows = []
    norm_b = np.linalg.norm(channel.rhs)
    for j, k in enumerate(times):
        column = columns[:, j]
        population = float(np.dot(r, column))
        trace = float(np.sum(recorded[k]))
        pad_rate = a * padding_c2
        padding = (
            pad_size
            / dimension
            * ((pad_rate**k if mixed else 0) + probability * (1 - pad_rate**k) / (1 - pad_rate))
        )
        success = trace - population - padding
        fidelity = float(recorded[k][dark])
        conditional = None
        if success > 1e-14:
            numerator = (fidelity - 2 * np.sqrt(q) * sign * column[dark] + q * population) / (1 - q)
            conditional = float(numerator / success)
        estimate = None
        if population > 1e-14:
            estimate = (channel.basis @ column)[: channel.size]
            estimate *= norm_b / (population * channel.singular_ratio * channel.matrix_scale)
        rows.append(
            {
                "time": int(k),
                "trace": trace,
                "reference_probability": population,
                "solution_probability": success,
                "padding_probability": float(padding),
                "augmented_fidelity": fidelity,
                "augmented_purified_distance": float(np.sqrt(max(0, 1 - fidelity))),
                "conditional_solution_fidelity": conditional,
                "relative_residual": None
                if estimate is None
                else float(np.linalg.norm(channel.matrix @ estimate - channel.rhs) / norm_b),
                "relative_solution_error": None
                if estimate is None
                else float(
                    np.linalg.norm(estimate - reference_solution)
                    / np.linalg.norm(reference_solution)
                ),
                "x1": None if estimate is None else float(estimate[0]),
                "reference_x1": float(reference_solution[0]),
                "absolute_x1_error": None
                if estimate is None
                else float(abs(estimate[0] - reference_solution[0])),
                "coherence_readout_defined": estimate is not None,
            }
        )
    return rows
