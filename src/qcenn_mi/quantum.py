"""Quantum primitives for the first QCeNN-MI feasibility assumption.

Assumption A1 uses a single real-amplitude qubit state for each column of a
2x2 inverse.  A rotation around Y prepares

    |q(theta)> = cos(theta/2)|0> + sin(theta/2)|1>.

This is intentionally small and hardware-friendly.  It establishes that the
solution direction can be represented and read out on a gate-based QPU before
introducing multi-cell coupling in later QCeNN milestones.
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator


def ry_cell_circuit(theta: float) -> QuantumCircuit:
    """Create the one-qubit QCeNN cell ansatz used in Assumption A1."""
    circuit = QuantumCircuit(1, name="qcenn_cell")
    circuit.ry(float(theta), 0)
    return circuit


def statevector_direction(theta: float) -> np.ndarray:
    """Return the real normalized state prepared by the QCeNN cell circuit.

    The implementation deliberately obtains the state through Qiskit's quantum
    circuit/statevector machinery rather than substituting the closed-form
    trigonometric expression in the solver.
    """
    state = np.asarray(Statevector.from_instruction(ry_cell_circuit(theta)).data)
    state = np.real_if_close(state, tol=1000)
    if np.iscomplexobj(state):
        raise ValueError("Assumption A1 expects a real-amplitude one-qubit state.")
    direction = np.asarray(state, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise RuntimeError("Quantum state has zero norm.")
    return direction / norm


def _expectation_from_counts(counts: dict[str, int], shots: int) -> float:
    zeros = int(counts.get("0", 0))
    ones = int(counts.get("1", 0))
    return (zeros - ones) / float(shots)


def shot_direction(
    theta: float,
    *,
    shots: int = 8192,
    seed: int | None = 42,
    backend: AerSimulator | None = None,
) -> np.ndarray:
    """Estimate the real solution direction using finite-shot measurements.

    For a real one-qubit state prepared with ``RY(theta)``, the Bloch
    expectations satisfy <Z>=cos(theta) and <X>=sin(theta).  Measuring both
    observables allows the state direction to be reconstructed without direct
    statevector access.  The same measurement pattern can later be adapted to
    a real QPU backend.
    """
    if shots <= 0:
        raise ValueError("shots must be a positive integer")

    simulator = backend or AerSimulator()

    z_circuit = QuantumCircuit(1, 1, name="qcenn_measure_z")
    z_circuit.ry(float(theta), 0)
    z_circuit.measure(0, 0)

    x_circuit = QuantumCircuit(1, 1, name="qcenn_measure_x")
    x_circuit.ry(float(theta), 0)
    x_circuit.h(0)
    x_circuit.measure(0, 0)

    compiled = transpile([z_circuit, x_circuit], simulator, optimization_level=1)
    result = simulator.run(compiled, shots=shots, seed_simulator=seed).result()

    z_exp = _expectation_from_counts(result.get_counts(0), shots)
    x_exp = _expectation_from_counts(result.get_counts(1), shots)

    # Reconstruct theta modulo 2*pi from the measured Bloch-vector projection.
    reconstructed_theta = math.atan2(x_exp, z_exp)
    direction = np.array(
        [
            math.cos(reconstructed_theta / 2.0),
            math.sin(reconstructed_theta / 2.0),
        ],
        dtype=float,
    )
    return direction / np.linalg.norm(direction)
