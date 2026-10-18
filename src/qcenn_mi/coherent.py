"""Coherent Neumann-LCU inverse action for small real positive-definite matrices.

No inverse, target solution, variational optimizer or classical residual update
is used in circuit construction. Classical dense dilation of I-tau*M is an
explicit input-encoding cost. This prototype is not an efficient oracle/QSVT
implementation and does not implement QCeNN local dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from .coupled import readout_direction, validate_matrix


@dataclass
class CoherentInverse:
    matrix: np.ndarray
    degree: int
    tau: float
    contraction_bound: float
    operator: QuantumCircuit
    data_qubits: int
    simulation_operator: QuantumCircuit

    @property
    def normalization(self):
        return self.tau * (self.degree + 1)

    def circuit(self, rhs, *, simulator=False):
        raw = np.asarray(rhs)
        if np.iscomplexobj(raw):
            raise ValueError("This prototype supports real right-hand sides")
        b = np.asarray(raw, dtype=float)
        if b.shape != (len(self.matrix),) or not np.isfinite(b).all() or np.linalg.norm(b) == 0:
            raise ValueError("rhs must be a finite nonzero vector matching the matrix")
        qc = QuantumCircuit(self.operator.num_qubits, name="coherent_inverse_action")
        qc.append(StatePreparation(b / np.linalg.norm(b)), range(self.data_qubits))
        qc.compose(self.simulation_operator if simulator else self.operator, inplace=True)
        return qc

    def ideal(self, rhs):
        """Read ideal postselected amplitudes for validation, not circuit fitting."""
        qc = self.circuit(rhs, simulator=True)
        block = np.asarray(Statevector.from_instruction(qc).data[: len(self.matrix)])
        probability = float(np.vdot(block, block).real)
        solution = np.real_if_close(block * self.normalization * np.linalg.norm(rhs)).real
        return {
            "solution": solution,
            "success_probability": probability,
            "residual": float(np.linalg.norm(self.matrix @ solution - rhs)),
        }

    def measure(self, rhs, *, shots=32768, seed=42, backend=None):
        """Postselected tomography; scale comes from measured success probability.

        No least-squares scaling or residual correction is applied. Real SPD
        matrices satisfy b.T @ inv(M) @ b > 0; positive b-overlap fixes the
        tomography global sign for this real-output experiment. Output is a
        classical validation readout; the preparation circuit itself is coherent.
        """
        simulator = backend is None or isinstance(backend, AerSimulator)
        qc = self.circuit(rhs, simulator=simulator)
        audit = {}
        state, minimum, purity = readout_direction(
            qc,
            backend=backend,
            shots=shots,
            seed=seed,
            postselect_ancillas=qc.num_qubits - self.data_qubits,
            basis_gates=["unitary", "u", "cx"] if simulator else None,
            audit=audit,
        )
        audit["simulator_fused_controlled_unitaries"] = simulator
        b = np.asarray(rhs, dtype=float)
        if state @ b < 0:
            state = -state
        probability = sum(audit["retained_shots"]) / (shots * len(audit["retained_shots"]))
        solution = state * self.normalization * np.linalg.norm(b) * np.sqrt(probability)
        return {
            "solution": solution,
            "success_probability": probability,
            "residual": float(np.linalg.norm(self.matrix @ solution - b)),
            "raw_min_eigenvalue": minimum,
            "raw_purity": purity,
            "audit": audit,
        }


def build_inverse_action(matrix, *, degree=7):
    """Build tau*sum_{k=0}^degree (I-tau*M)^k via coherent SELECT and postselection.

    Supports real symmetric 2x2/4x4 matrices certified SPD by a positive
    Gershgorin lower bound. This is a sufficient, not necessary, SPD condition.
    One fresh block flag per factor avoids resetting or measuring mid-circuit.
    The selector superposes polynomial orders; its preparation is uncomputed.
    Postselect all non-data qubits to zero. Reusing the same operator supports
    arbitrary input states without knowing their inverse solutions.
    """
    m = validate_matrix(matrix)
    if not np.allclose(m, m.T, atol=1e-12, rtol=0):
        raise ValueError("Coherent prototype requires a real symmetric matrix")
    if isinstance(degree, bool) or not isinstance(degree, int) or not 1 <= degree <= 7:
        raise ValueError("degree must be an integer from 1 to 7 (simulator resource limit)")
    radius = np.sum(np.abs(m), axis=1) - np.abs(np.diag(m))
    lower, upper = float(np.min(np.diag(m) - radius)), float(np.max(np.diag(m) + radius))
    if lower <= 0:
        raise ValueError("Positive Gershgorin lower bound required to certify convergence")
    tau = 2 / (lower + upper)
    bound = (upper - lower) / (upper + lower)
    b = np.eye(len(m)) - tau * m
    # Dense block encoding, not inverse/polynomial construction.
    values, vectors = np.linalg.eigh(np.eye(len(m)) - b @ b)
    root = (vectors * np.sqrt(np.maximum(0, values))) @ vectors.T
    dilation = np.block([[b, root], [root, -b]])
    block_gate = UnitaryGate(dilation, label="encode_I-tauM")
    ndata = int(np.log2(len(m)))
    nselect = int(np.ceil(np.log2(degree + 1)))
    flags = list(range(ndata, ndata + degree))
    selector = list(range(ndata + degree, ndata + degree + nselect))
    qc = QuantumCircuit(ndata + degree + nselect, name="coherent_neumann_lcu")
    weights = np.zeros(2**nselect)
    weights[: degree + 1] = 1 / np.sqrt(degree + 1)
    prepare = StatePreparation(weights)
    qc.append(prepare, selector)
    simulation = QuantumCircuit(qc.num_qubits, name="coherent_neumann_simulation")
    simulation.append(prepare, selector)
    # For selector k, apply B k times, each with a distinct flag ancilla.
    for k in range(1, degree + 1):
        selected_block = block_gate.control(nselect, ctrl_state=k)
        # Exact local controlled-unitary matrix accelerates ideal simulation.
        # This only fuses an individual SELECT block, never the inverse/polynomial.
        controlled = np.eye((2**nselect) * len(dilation), dtype=complex)
        indices = k + (np.arange(len(dilation)) << nselect)
        controlled[np.ix_(indices, indices)] = dilation
        fused = UnitaryGate(controlled, label="SELECT block (simulator fusion)")
        for step in range(k):
            targets = selector + list(range(ndata)) + [flags[step]]
            qc.append(selected_block, targets)
            simulation.append(fused, targets)
    qc.append(prepare.inverse(), selector)
    simulation.append(prepare.inverse(), selector)
    return CoherentInverse(m.copy(), degree, tau, bound, qc, ndata, simulation)
