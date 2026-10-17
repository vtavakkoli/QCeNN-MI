"""Small dense quantum dilation of A4's polynomial correction operator.

This explicitly constructs the polynomial classically and compiles its dense
unitary dilation. It tests quantum execution/postselection, not efficient quantum
polynomial synthesis, QSVT, or a scalable block-encoding/data-access scheme.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .coupled import readout_direction, validate_matrix


@dataclass
class PolynomialResult:
    estimate: np.ndarray
    residual: float
    success_probabilities: list[float]
    circuits: list[QuantumCircuit]
    normalization: float

    def measure(self, matrix, *, backend=None, shots=8192, seed=42):
        """Finite-shot postselected directions with classical least-squares scales."""
        m = validate_matrix(matrix)
        if m.shape != self.estimate.shape:
            raise ValueError("Matrix shape does not match polynomial result")
        columns, diagnostics = [], []
        for j, circuit in enumerate(self.circuits):
            record = {}
            state, minimum, purity = readout_direction(
                circuit,
                backend=backend,
                shots=shots,
                seed=seed + j,
                postselect=True,
                audit=record,
            )
            transformed = m @ state
            columns.append(state * transformed[j] / (transformed @ transformed))
            record.update(raw_min_eigenvalue=minimum, raw_purity=purity)
            diagnostics.append(record)
        return np.column_stack(columns), diagnostics


def quantum_polynomial_correct(matrix, estimate, *, order=2):
    """Apply sum_k (I-XM)^k to each amplitude-encoded column using an ancilla.

    The upper block of the dilation is P/alpha. Ancilla=0 postselection yields
    P x/(alpha ||x||). Classical normalization restores the unnormalized vector.
    The full candidate is reported even if it worsens the residual: this is an
    execution experiment; use polynomial_correct for a safeguarded update.
    """
    m = validate_matrix(matrix)
    raw = np.asarray(estimate)
    if np.iscomplexobj(raw):
        raise ValueError("Only real estimates are supported")
    x = np.asarray(raw, dtype=float)
    if x.shape != m.shape or not np.isfinite(x).all():
        raise ValueError("Estimate must be finite and match matrix shape")
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise ValueError("order must be a positive integer")
    n = len(m)
    identity = np.eye(n)
    error = identity - x @ m
    polynomial = identity.copy()
    power = identity.copy()
    for _ in range(order):
        power = power @ error
        polynomial += power
    if not np.isfinite(polynomial).all():
        raise ValueError("Polynomial overflow")
    u, singular, vh = np.linalg.svd(polynomial)
    alpha = max(1.0, float(singular[0]))
    a = polynomial / alpha
    root = np.sqrt(np.maximum(0.0, 1 - (singular / alpha) ** 2))
    left = (u * root) @ u.T
    right = (vh.T * root) @ vh
    dilation = np.block([[a, left], [right, -a.T]])
    columns, probabilities, circuits = [], [], []
    qubits = int(np.log2(n))
    for j in range(n):
        norm = np.linalg.norm(x[:, j])
        if norm == 0:
            raise ValueError("Cannot amplitude encode a zero column")
        circuit = QuantumCircuit(qubits + 1, name="qcenn_polynomial")
        circuit.prepare_state(x[:, j] / norm, list(range(qubits)))
        circuit.unitary(dilation, list(range(qubits + 1)), label="P/alpha dilation")
        state = np.asarray(Statevector.from_instruction(circuit).data)
        block = state[:n]  # Highest-index qubit is the postselection ancilla.
        probabilities.append(float(np.vdot(block, block).real))
        columns.append(np.real_if_close(block * alpha * norm).real)
        circuits.append(circuit)
    value = np.column_stack(columns)
    return PolynomialResult(
        value, float(np.linalg.norm(m @ value - identity)), probabilities, circuits, alpha
    )
