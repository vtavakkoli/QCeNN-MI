"""Padded coherent inverse action with a memory-bounded spectral simulator.

The circuit remains a Neumann-LCU circuit with fresh flags. Classical dense
spectral/SVD input encoding is explicit and costs O(N^3); no quantum speed-up is
claimed. Larger runs simulate the full ancilla state in the encoding basis,
not an Aer native-gate execution or a classical residual-correction solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, UCRYGate
from qiskit.quantum_info import Operator


def resources(size, degree, *, max_memory_mib=512):
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise ValueError("size must be an integer >= 2")
    if isinstance(degree, bool) or not isinstance(degree, int) or not 1 <= degree <= 9:
        raise ValueError("degree must be an integer from 1 to 9")
    if not np.isfinite(max_memory_mib) or max_memory_mib <= 0:
        raise ValueError("max_memory_mib must be positive")
    data = (size - 1).bit_length()
    selector = degree.bit_length()
    padded = 1 << data
    total = data + degree + selector
    state_bytes = (1 << total) * 16
    # State, uncomputed state, update buffers, input/eigenvectors and eigensolver workspace.
    estimate = 4 * state_bytes + 8 * padded * padded * 8
    report = {
        "size": size,
        "padded_size": padded,
        "data_qubits": data,
        "flag_qubits": degree,
        "selector_qubits": selector,
        "total_qubits": total,
        "statevector_bytes": state_bytes,
        "estimated_working_bytes": estimate,
        "controlled_block_calls": degree * (degree + 1) // 2,
    }
    if estimate > max_memory_mib * 1024**2:
        raise MemoryError(
            f"Estimated {estimate / 1024**2:.1f} MiB exceeds {max_memory_mib} MiB budget"
        )
    return report


@dataclass
class LargeCoherentInverse:
    matrix: np.ndarray
    input_basis: np.ndarray
    output_basis: np.ndarray
    encoding_kind: str
    contractions: np.ndarray
    tau: float
    condition_number: float
    singular_values: np.ndarray
    degree: int
    resource_report: dict
    selector_unitary: np.ndarray

    @property
    def normalization(self):
        return self.tau * (self.degree + 1)

    def _rhs(self, rhs):
        raw = np.asarray(rhs)
        if np.iscomplexobj(raw):
            raise ValueError("Only real input vectors are supported")
        b = np.asarray(raw, dtype=float)
        if b.shape != (len(self.matrix),) or not np.isfinite(b).all() or np.linalg.norm(b) == 0:
            raise ValueError("rhs must be finite, nonzero and match matrix size")
        return b

    def simulate(self, rhs, *, return_state=False):
        """Exact structured statevector simulation, including every flag and selector.

        The spectral/SVD encoding basis diagonalizes the input contraction, never its
        inverse. Each SELECT operation updates flag amplitudes unitarily. There
        is no polynomial evaluation, inverse or residual update in this loop.
        Final data-basis conversion is performed on the success sector only;
        other sectors retain a lazy unitary basis change, preserving all norms.
        """
        b = self._rhs(rhs)
        n = len(b)
        padded = self.resource_report["padded_size"]
        nselector = 1 << self.resource_report["selector_qubits"]
        coefficients = np.zeros(padded, dtype=complex)
        coefficients[:n] = self.input_basis.T @ (b / np.linalg.norm(b))
        state = np.zeros((nselector, 1 << self.degree, padded), dtype=complex)
        state[:, 0, :] = self.selector_unitary[:, 0, None] * coefficients
        root = np.sqrt(np.maximum(0.0, 1 - self.contractions**2))
        for k in range(1, self.degree + 1):
            for flag in range(k):
                view = state[k].reshape(1 << (self.degree - flag - 1), 2, 1 << flag, padded)
                zero, one = view[:, 0].copy(), view[:, 1].copy()
                view[:, 0] = self.contractions * zero + root * one
                view[:, 1] = root * zero - self.contractions * one
        state = np.tensordot(self.selector_unitary.conj().T, state, axes=(1, 0))
        norm = float(np.linalg.norm(state.ravel()) ** 2)
        success_sector = state[0, 0]
        probability = float(np.vdot(success_sector, success_sector).real)
        amplitudes = self.output_basis @ success_sector[:n]
        solution = np.real_if_close(amplitudes * self.normalization * np.linalg.norm(b)).real
        residual = float(np.linalg.norm(self.matrix @ solution - b))
        result = {
            "solution": solution,
            "success_probability": probability,
            "relative_residual": residual / float(np.linalg.norm(b)),
            "residual": residual,
            "state_norm": norm,
            "padding_success_weight": float(np.linalg.norm(success_sector[n:]) ** 2),
            "method": "exact structured statevector simulation in spectral encoding basis",
        }
        if return_state:
            result["spectral_statevector"] = state
        return result

    def sample_z(self, result, *, shots=131072, seed=42):
        """Sample the marginal of all-zero ancillas and computational-basis data.

        This is sampling from the exact simulated circuit distribution, not an
        Aer execution. Z readout alone does not reconstruct signed amplitudes.
        """
        if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
            raise ValueError("shots must be a positive integer")
        rng = np.random.default_rng(seed)
        accepted = int(rng.binomial(shots, np.clip(result["success_probability"], 0, 1)))
        probabilities = np.square(np.abs(result["solution"]))
        probabilities /= probabilities.sum()
        counts = rng.multinomial(accepted, probabilities)
        return {
            "shots": shots,
            "accepted": accepted,
            "success_frequency": accepted / shots,
            "conditional_counts": counts.tolist(),
            "conditional_total_variation": None
            if accepted == 0
            else float(0.5 * np.sum(np.abs(counts / accepted - probabilities))),
            "signed_inverse_reconstructed": False,
            "method": "exact-distribution Z sampling",
        }

    def circuit(self, rhs, *, max_export_size=128, allow_large=False):
        """Export the corresponding Qiskit circuit, lazily controlled for compilation.

        Exporting/compiling dense spectral basis changes is expensive. Default
        guard prevents accidental 1000-dimensional native-circuit synthesis.
        """
        b = self._rhs(rhs)
        if len(b) > max_export_size and not allow_large:
            raise MemoryError("Large dense circuit export requires explicit allow_large=True")
        p = self.resource_report["padded_size"]
        data = list(range(self.resource_report["data_qubits"]))
        flags = list(range(len(data), len(data) + self.degree))
        selector = list(range(len(data) + self.degree, self.resource_report["total_qubits"]))
        input_basis, output_basis = np.eye(p), np.eye(p)
        input_basis[: len(b), : len(b)] = self.input_basis
        output_basis[: len(b), : len(b)] = self.output_basis
        padded_rhs = np.zeros(p)
        padded_rhs[: len(b)] = b / np.linalg.norm(b)
        qc = QuantumCircuit(self.resource_report["total_qubits"], name="coherent_padded_inverse")
        qc.append(StatePreparation(padded_rhs), data)
        qc.unitary(input_basis.T, data, label="spectral input basis")
        weights = self.selector_unitary[:, 0].real
        preparation = StatePreparation(weights)
        qc.append(preparation, selector)
        block = QuantumCircuit(len(data) + 1, name="spectral_B_encoding")
        block.z(len(data))
        block.append(
            UCRYGate((2 * np.arccos(np.clip(self.contractions, -1, 1))).tolist()),
            [len(data)] + list(range(len(data))),
        )
        for k in range(1, self.degree + 1):
            controlled = block.to_gate().control(len(selector), ctrl_state=k, annotated=True)
            for flag in flags[:k]:
                qc.append(controlled, selector + data + [flag])
        qc.append(preparation.inverse(), selector)
        qc.unitary(output_basis, data, label="restore data basis")
        return qc


def build_large_inverse_action(matrix, *, degree=7, max_memory_mib=512):
    raw = np.asarray(matrix)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("matrix must be square")
    report = resources(len(raw), degree, max_memory_mib=max_memory_mib)
    if np.iscomplexobj(raw):
        raise ValueError("Only real matrices are supported")
    m = np.asarray(raw, dtype=float)
    if not np.isfinite(m).all():
        raise ValueError("Matrix must be finite")
    # Explicit classical spectral encoding. No singular/eigenvalue reciprocals.
    if np.allclose(m, m.T, atol=0, rtol=1e-12):
        values, vectors = np.linalg.eigh(m)
    else:
        values, vectors = None, None
    if values is not None and values[0] > np.finfo(float).eps * float(values[-1]):
        input_basis = output_basis = vectors
        encoding_kind = "classical SPD eigendecomposition"
    else:
        input_basis, values, vh = np.linalg.svd(m, full_matrices=False)
        output_basis = vh.T
        encoding_kind = "classical real singular-value decomposition"
    smallest, largest = float(np.min(values)), float(np.max(values))
    # Dimension-aware numerical-rank threshold. Dense decompositions accumulate
    # rounding error; an epsilon-only cutoff can misclassify rank-loss limits.
    if smallest <= np.finfo(float).eps * len(m) * largest:
        raise ValueError("Matrix must be numerically nonsingular")
    tau = (2 / largest) / (1 + smallest / largest)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("Matrix scale exceeds supported floating-point range")
    contractions = np.zeros(report["padded_size"])
    contractions[: len(m)] = 1 - tau * values
    # Padding represents eigenvalue 1/tau and is decoupled from valid input data.
    weights = np.zeros(1 << report["selector_qubits"])
    weights[: degree + 1] = 1 / np.sqrt(degree + 1)
    selector = np.asarray(Operator(StatePreparation(weights)).data)
    return LargeCoherentInverse(
        m.copy(),
        input_basis,
        output_basis,
        encoding_kind,
        contractions,
        tau,
        largest / smallest,
        values.copy(),
        degree,
        report,
        selector,
    )
