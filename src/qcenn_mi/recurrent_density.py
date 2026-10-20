"""Exact reduced-density simulation of the recurrent residual quantum channel.

This is a classical simulator, not native-gate or QPU execution. In the right
singular basis of F, C=sqrt(I-F^*F) is diagonal. Therefore a complete reduced
quantum channel update costs O(P^2), rather than simulating an unused P-sized
environment. It is the same channel as recurrent_quantum.py, with no classical
residual correction. Logical depolarization is applied once per channel update,
and must not be interpreted as a two-qubit gate error rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectralRecurrentChannel:
    matrix: np.ndarray
    rhs: np.ndarray
    matrix_scale: float
    singular_ratio: float
    basis: np.ndarray
    residual_eigenvalues: np.ndarray
    no_jump_values: np.ndarray
    reference: np.ndarray
    padding_projector: np.ndarray | None

    @property
    def size(self):
        return len(self.matrix)

    @property
    def dimension(self):
        return len(self.reference)

    def initial_density(self):
        return np.outer(self.reference, self.reference)

    def step(self, density, *, depolarizing_probability=0.0, work=None):
        """Update a simulated density matrix in place; observations are not inputs.

        rho' = C rho C + Tr(G rho)|r><r|, followed by
        rho''=(1-p)rho' + p Tr(rho') I/P. The trace factor is retained rather
        than renormalizing a numerically drifting simulated state.
        """
        probability = depolarizing_probability
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("depolarizing_probability must be in [0,1]")
        if density.shape != (self.dimension, self.dimension):
            raise ValueError("density shape does not match channel")
        jump = np.dot(self.residual_eigenvalues, density.diagonal())
        density *= self.no_jump_values[:, None]
        density *= self.no_jump_values[None, :]
        if work is None:
            work = np.empty_like(density)
        np.multiply(self.reference[:, None], self.reference[None, :], out=work)
        work *= jump
        density += work
        if probability:
            trace = np.trace(density)
            density *= 1 - probability
            density.flat[:: self.dimension + 1] += probability * trace / self.dimension
        return density

    def change_rotation_basis(self, density, delta):
        """Retain the physical state when M(theta)=M(0)G(theta) changes.

        G rotates the first two solution coordinates; the reference and padding
        are fixed. V(theta)=T(theta)^*V(0), so the change of simulation basis is
        W=I+U(G(delta)-I)U^*. This is an exact rank-two basis change, not a
        predictor, a reinitialization, or an update toward a classical solution.
        """
        if not np.isfinite(delta):
            raise ValueError("rotation increment must be finite")
        if delta == 0:
            return density
        cosine, sine = np.cos(delta), np.sin(delta)
        block = np.array([[cosine - 1, -sine], [sine, cosine - 1]])
        rows = self.basis[:2, :].T
        left = density + (rows @ block) @ (rows.T @ density)
        density[:] = left + ((left @ rows) @ block.T) @ rows.T
        return density

    def physical_density(self, density, *, theta=0.0):
        """Diagnostic reconstruction, used only for small independent cross-checks."""
        basis = self.basis.copy()
        cosine, sine = np.cos(theta), np.sin(theta)
        basis[:2] = np.array([[cosine, sine], [-sine, cosine]]) @ basis[:2]
        return basis @ density @ basis.T

    def certificate(self, steps, *, reference_initial_state=True):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        a2 = self.singular_ratio**2
        rate = 1 - a2 / (2 * (1 + a2))
        return (0.5 if reference_initial_state else 1.0) * rate**steps


def build_density_channel(matrix, rhs, *, max_memory_mib=512):
    """Compile F's spectral encoding without allocating its full Julia dilation.

    Dense input SVDs remain classical preprocessing. No inverse, linear solve,
    reciprocal polynomial, or reference solution is evaluated here.
    """
    raw = np.asarray(matrix)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or len(raw) < 2:
        raise ValueError("matrix must be square with order >= 2")
    if np.iscomplexobj(raw) or np.iscomplexobj(rhs):
        raise ValueError("Only real matrices and right-hand sides are supported")
    matrix = np.asarray(raw, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    if not np.isfinite(matrix).all() or rhs.shape != (len(matrix),):
        raise ValueError("matrix/rhs must be finite and have matching dimensions")
    norm = float(np.linalg.norm(rhs))
    if not np.isfinite(rhs).all() or not np.isfinite(norm) or norm == 0:
        raise ValueError("rhs must be finite and nonzero")
    if not np.isfinite(max_memory_mib) or max_memory_mib <= 0:
        raise ValueError("max_memory_mib must be finite and positive")
    dimension = 1 << len(matrix).bit_length()
    estimate = 16 * 12 * dimension**2 + 16 * len(matrix) ** 2
    if estimate > max_memory_mib * 1024**2:
        raise MemoryError("Spectral channel exceeds its declared working-memory budget")
    singular = np.linalg.svd(matrix, compute_uv=False)
    scale = float(singular[0])
    if scale == 0 or singular[-1] <= np.finfo(float).eps * len(matrix) * scale:
        raise ValueError("matrix must be numerically nonsingular")
    ratio = float(singular[-1] / scale)
    residual = np.eye(dimension)
    residual[: len(matrix), : len(matrix)] = matrix / scale
    residual[: len(matrix), len(matrix)] = -ratio * rhs / norm
    residual[len(matrix), :] = 0
    residual /= np.sqrt(1 + ratio**2)
    _, values, right = np.linalg.svd(residual)
    if values[0] > 1 + 1e-12:
        raise ArithmeticError("Residual operator is not a contraction")
    eigenvalues = np.minimum(values, 1.0) ** 2
    basis = right.T
    reference = basis[len(matrix), :].copy()
    padding = basis[len(matrix) + 1 :, :]
    projector = None if not len(padding) else padding.T @ padding
    return SpectralRecurrentChannel(
        matrix.copy(),
        rhs.copy(),
        scale,
        ratio,
        basis,
        eigenvalues,
        np.sqrt(np.maximum(0.0, 1 - eigenvalues)),
        reference,
        projector,
    )


@dataclass(frozen=True)
class DensityObserver:
    """Read-only diagnostics using a separately computed reference solution.

    x is reconstructed from final-state solution/reference coherences. Recording
    it during a simulator trajectory is not a free hardware measurement and
    must not feed a later update. Rotated trajectories use the same observer
    because both target and spectral basis rotate covariantly.
    """

    channel: SpectralRecurrentChannel
    reference_solution: np.ndarray
    target_coordinates: np.ndarray
    solution_coordinates: np.ndarray

    @classmethod
    def create(cls, channel, reference_solution):
        solution = np.asarray(reference_solution, dtype=float)
        if solution.shape != (channel.size,) or not np.isfinite(solution).all():
            raise ValueError("reference_solution must be finite and match matrix size")
        state = np.zeros(channel.dimension)
        state[: channel.size] = (
            channel.singular_ratio * channel.matrix_scale * solution / np.linalg.norm(channel.rhs)
        )
        state[channel.size] = 1
        state /= np.linalg.norm(state)
        direction = np.zeros(channel.dimension)
        direction[: channel.size] = solution / np.linalg.norm(solution)
        return cls(channel, solution.copy(), channel.basis.T @ state, channel.basis.T @ direction)

    def read(self, density, *, theta=0.0, return_solution=False):
        channel = self.channel
        trace = float(np.trace(density).real)
        column = density @ channel.reference
        reference_probability = float(np.vdot(channel.reference, column).real)
        padding = (
            0.0
            if channel.padding_projector is None
            else float(np.einsum("ij,ji->", channel.padding_projector, density).real)
        )
        success = trace - reference_probability - padding
        fidelity = float(np.vdot(self.target_coordinates, density @ self.target_coordinates).real)
        conditional = (
            None
            if success <= 1e-15
            else float(
                np.vdot(self.solution_coordinates, density @ self.solution_coordinates).real
                / success
            )
        )
        if reference_probability <= 1e-15:
            raise ArithmeticError("Reference sector is too small for coherence diagnostics")
        estimate = (channel.basis @ column)[: channel.size]
        estimate *= np.linalg.norm(channel.rhs) / (
            reference_probability * channel.singular_ratio * channel.matrix_scale
        )
        residual = np.linalg.norm(channel.matrix @ estimate - channel.rhs)
        cosine, sine = np.cos(theta), np.sin(theta)
        x1 = cosine * estimate[0] + sine * estimate[1]
        target_x1 = cosine * self.reference_solution[0] + sine * self.reference_solution[1]
        result = {
            "x1": float(np.real(x1)),
            "reference_x1": float(target_x1),
            "absolute_x1_error": float(abs(x1 - target_x1)),
            "relative_residual": float(residual / np.linalg.norm(channel.rhs)),
            "relative_solution_error": float(
                np.linalg.norm(estimate - self.reference_solution)
                / np.linalg.norm(self.reference_solution)
            ),
            "augmented_fidelity": fidelity,
            "conditional_solution_fidelity": conditional,
            "solution_probability": success,
            "reference_probability": reference_probability,
            "padding_probability": padding,
            "trace": trace,
            "purity": float(np.vdot(density, density).real),
        }
        if return_solution:
            estimate = estimate.copy()
            estimate[:2] = np.array([[cosine, sine], [-sine, cosine]]) @ estimate[:2]
            result["solution"] = estimate
        return result
