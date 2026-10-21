"""Bounded Chebyshev residual filtering and explicit real-Julia QSP circuits.

The ideal signed filter preserves the kernel. Finite phase synthesis uses a
declared attenuation below one; its small kernel leakage is never called an
exact stationary solution. Dense diagnostics do not drive circuit updates.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, fields

import numpy as np
from numpy.polynomial.chebyshev import chebinterpolate
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, UnitaryGate

from .recurrent_density import SpectralRecurrentChannel
from .recurrent_quantum import _integer, _prepare_reference


def _gap(gap):
    if not np.isfinite(gap) or not 0 < gap < 1:
        raise ValueError("gap must be in (0,1)")


def _logcosh(value):
    value = np.abs(value)
    return value + np.log1p(np.exp(-2 * value)) - np.log(2)


def filter_degree(gap, suppression=0.1):
    _gap(gap)
    if not np.isfinite(suppression) or not 0 < suppression < 1:
        raise ValueError("suppression must be in (0,1)")
    return max(1, int(np.ceil(np.arccosh(1 / suppression) / (2 * np.arctanh(np.sqrt(gap))))))


def chebyshev_response(eigenvalues, gap, degree):
    """Stable r_d(lambda); works below the gap for kernel/drift diagnostics."""
    _gap(gap)
    _integer(degree, "filter_degree")
    values = np.asarray(eigenvalues, dtype=float)
    if not np.isfinite(values).all() or np.any(values < -1e-12) or np.any(values > 1 + 1e-12):
        raise ValueError("residual eigenvalues must be in [0,1]")
    values = np.clip(values, 0, 1)
    scaled = (1 + gap - 2 * values) / (1 - gap)
    denominator = _logcosh(2 * degree * np.arctanh(np.sqrt(gap)))
    oscillating = np.cos(degree * np.arccos(np.clip(scaled, -1, 1))) * np.exp(-denominator)
    argument = 2 * np.arcsinh(np.sqrt(np.maximum(gap - values, 0) / (1 - gap)))
    growing = np.exp(_logcosh(degree * argument) - denominator)
    return np.where(scaled > 1, growing, oscillating)


def suppression_bound(gap, degree):
    _gap(gap)
    _integer(degree, "filter_degree")
    return float(np.exp(-_logcosh(2 * degree * np.arctanh(np.sqrt(gap)))))


@dataclass(frozen=True, kw_only=True)
class ChebyshevDensityChannel(SpectralRecurrentChannel):
    filter_degree: int
    certified_gap: float

    def certificate(self, steps, *, reference_initial_state=True):
        _integer(steps, "steps")
        eta = suppression_bound(self.certified_gap, self.filter_degree)
        initial = 0.5 if reference_initial_state else 1.0
        return {
            "ideal_augmented_infidelity_upper_bound": initial * ((1 + eta**2) / 2) ** steps,
            "ideal_residual_unitary_queries": 2 * self.filter_degree * steps,
            "scope": "ideal real-Julia signal access; excludes synthesis, readout and noise",
        }


def filter_density_channel(channel, degree=None, *, gap=None, suppression=0.1):
    """Exact signed spectral simulator; no target inverse or state is constructed."""
    gap = channel.singular_ratio**2 / (1 + channel.singular_ratio**2) if gap is None else gap
    _gap(gap)
    if gap > channel.singular_ratio**2 / (1 + channel.singular_ratio**2) + 1e-14:
        raise ValueError("declared gap exceeds the available certified gap")
    degree = filter_degree(gap, suppression) if degree is None else degree
    response = chebyshev_response(channel.residual_eigenvalues, gap, degree)
    values = {f.name: getattr(channel, f.name) for f in fields(SpectralRecurrentChannel)}
    values.update(no_jump_values=response, residual_eigenvalues=np.maximum(0, 1 - response**2))
    return ChebyshevDensityChannel(**values, filter_degree=degree, certified_gap=gap)


@dataclass(frozen=True)
class FilterPhases:
    gap: float
    degree: int
    attenuation: float
    phases: np.ndarray
    coefficient_error_estimate: float
    iterations: int

    @property
    def queries_per_update(self):
        return 2 * self.degree


def synthesize_filter(gap, degree=None, *, suppression=0.1, margin=1e-12, max_degree=64):
    """Phases depend on a gap and degree only, never on M^-1 b.

    The l1 Chebyshev coefficient residual is a floating-point estimate, not an
    interval certificate. Excess degree is rejected before an expensive solve.
    """
    _gap(gap)
    degree = filter_degree(gap, suppression) if degree is None else degree
    _integer(degree, "filter_degree", 1)
    if degree > max_degree:
        raise MemoryError("Chebyshev phase synthesis exceeds its declared degree budget")
    if not np.isfinite(margin) or not 0 < margin < 0.1:
        raise ValueError("margin must be in (0,0.1)")
    from pyqsp.sym_qsp_opt import newton_solver

    coefficients = chebinterpolate(
        lambda x: (1 - margin) * chebyshev_response(1 - x**2, gap, degree), 2 * degree
    )
    coefficients[1::2] = 0
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, iterations, protocol = newton_solver(
            coefficients[::2], parity=0, crit=1e-12, maxiter=80
        )
    achieved, _ = protocol.gen_jacobian()
    error = float(np.sum(np.abs(achieved - coefficients[::2])))
    if not np.isfinite(error) or error > 1e-10:
        raise ArithmeticError(f"QSP phase synthesis failed its coefficient check: {error}")
    return FilterPhases(gap, degree, 1 - margin, protocol.full_phases.copy(), error, iterations)


def qsp_filter_circuit(baseline, specification):
    """Success block is the imaginary QSP response; exactly 2d U0 calls.

    This construction uses the real Julia dilation, not an arbitrary block
    encoding. Conjugate phase branches share the same real U0, so no controlled
    oracle or additional oracle call is hidden in the quadrature extraction.
    """
    n = baseline.data_qubits
    if np.max(np.abs(np.asarray(baseline.collision_unitary).imag)) > 1e-14:
        raise ValueError("real-Julia conjugate-branch construction requires a real dilation")
    bound = baseline.singular_ratio**2 / (1 + baseline.singular_ratio**2)
    if specification.gap > bound + 1e-14:
        raise ValueError("filter gap exceeds the baseline certificate")
    flag, selector = n, n + 1
    circuit = QuantumCircuit(n + 2, name="Chebyshev QSP filter")
    circuit.h(selector)
    phases = specification.phases[::-1]
    circuit.rzz(-2 * phases[0], flag, selector)
    residual = UnitaryGate(baseline.collision_unitary, label="input U0")
    for phase in phases[1:]:
        circuit.rzz(-np.pi / 2, flag, selector)
        circuit.append(residual, list(range(n)) + [flag])
        circuit.rzz(np.pi / 2, flag, selector)
        circuit.rzz(-2 * phase, flag, selector)
    circuit.z(selector)
    circuit.h(selector)
    circuit.global_phase -= np.pi / 2
    circuit.metadata = {
        "filter_degree": specification.degree,
        "residual_unitary_calls": len(phases) - 1,
        "adjoint_calls": 0,
        "access_model": "real Julia U0 with conjugate phase control",
        "target_solution_loaded": False,
    }
    return circuit


def chebyshev_schedule_circuit(schedule, *, workspace="reset", initial_state=None, max_qubits=24):
    """Retain data; reset every non-success signal outcome to the known reference."""
    schedule = list(schedule)
    if not schedule or workspace not in {"coherent", "reset"}:
        raise ValueError("nonempty schedule and coherent/reset workspace required")
    n = schedule[0][0].data_qubits
    if any(b.data_qubits != n or b.size != schedule[0][0].size for b, _ in schedule):
        raise ValueError("all schedule entries must have the same data dimension")
    width = n + (n + 3) * (len(schedule) if workspace == "coherent" else 1)
    if width > max_qubits:
        raise MemoryError("Complete Chebyshev circuit exceeds the declared qubit budget")
    circuit = QuantumCircuit(width)
    data = list(range(n))
    if initial_state is None:
        _prepare_reference(circuit, data, schedule[0][0].reference_index)
    else:
        state = np.asarray(initial_state, dtype=complex)
        if (
            state.shape != (1 << n,)
            or not np.isfinite(state).all()
            or not np.isclose(np.linalg.norm(state), 1)
        ):
            raise ValueError("initial state must be finite, normalized and match the data")
        circuit.append(StatePreparation(state), data)
    queries = 0
    for k, (baseline, specification) in enumerate(schedule):
        start = n + k * (n + 3) if workspace == "coherent" else n
        flag, selector, failed = start, start + 1, start + 2
        environment = list(range(start + 3, start + 3 + n))
        if workspace == "reset" and k:
            circuit.reset([flag, selector, failed] + environment)
        _prepare_reference(circuit, environment, baseline.reference_index)
        circuit.compose(
            qsp_filter_circuit(baseline, specification), data + [flag, selector], inplace=True
        )
        circuit.cx(flag, failed)
        circuit.cx(selector, failed)
        circuit.ccx(flag, selector, failed)
        for first, second in zip(data, environment, strict=True):
            circuit.cswap(failed, first, second)
        queries += specification.queries_per_update
    circuit.metadata = {
        "residual_unitary_calls": queries,
        "target_solution_loaded": False,
        "data_reinitialized": False,
        "intermediate_data_measurements": 0,
        "ancilla_resets": (len(schedule) - 1) * (n + 3) if workspace == "reset" else 0,
    }
    return circuit
