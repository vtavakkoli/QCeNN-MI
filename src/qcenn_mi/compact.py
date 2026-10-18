"""Compact Chebyshev-LCU inverse action with one walk ancilla.

Uses the existing explicit classical spectral/SVD encoding; no efficient data
oracle or quantum speed-up is claimed. Binary powers of the known multiplexed
walk are angle-scaled input rotations, not precomputed inverse eigenvalues.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, UCRYGate
from qiskit.quantum_info import Operator

from .coherent_large import LargeCoherentInverse, build_large_inverse_action


def _resources(size, degree):
    data = (size - 1).bit_length()
    selector = max(1, degree.bit_length())
    padded = 1 << data
    state_bytes = 16 * (1 << (data + 1 + selector))
    return {
        "data_qubits": data,
        "walk_ancillas": 1,
        "selector_qubits": selector,
        "total_qubits": data + 1 + selector,
        "padded_size": padded,
        "degree": degree,
        "controlled_walk_power_gadgets": selector,
        "statevector_bytes": state_bytes,
        # State, selected-branch copies and dense encoding workspace.
        "estimated_working_bytes": 6 * state_bytes + 64 * padded**2,
        "selector_uncomputation": "lazy exact success projection",
    }


@dataclass
class CompactInverse:
    encoding: LargeCoherentInverse
    degree: int
    coefficients: np.ndarray
    selector_weights: np.ndarray
    walk_angles: np.ndarray
    residual_bound: float
    target_tolerance: float
    max_memory_mib: float

    @property
    def selector_unitary(self):
        """Explicit selector unitary for small-state inspection (expensive).

        Normal construction and simulation retain only its first column.
        """
        self._check_memory(dense_selector=True)
        return np.asarray(Operator(StatePreparation(self.selector_weights)).data)

    @property
    def normalization(self):
        return float(np.sum(np.abs(self.coefficients)))

    @property
    def resource_report(self):
        return {
            **_resources(len(self.encoding.matrix), self.degree),
            "classical_encoding": self.encoding.encoding_kind,
        }

    def _check_memory(self, *, dense_selector=False):
        estimate = self.resource_report["estimated_working_bytes"]
        if dense_selector:
            estimate += 4 * 16 * len(self.selector_weights) ** 2
        if estimate > self.max_memory_mib * 1024**2:
            raise MemoryError(
                f"Estimated {estimate / 1024**2:.1f} MiB exceeds "
                f"{self.max_memory_mib} MiB budget"
            )

    def circuit(self, rhs, *, verification_state=None, fuse_input=True):
        """Coherent inverse action; optional independent final reference projection.

        verification_state is used only by hardware benchmarking AFTER inverse
        action. It never affects coefficients, encoding or the inverse circuit.
        """
        b = self.encoding._rhs(rhs)
        n, p = len(b), self.resource_report["padded_size"]
        data = list(range(self.resource_report["data_qubits"]))
        flag = len(data)
        selector = list(range(flag + 1, self.resource_report["total_qubits"]))
        qc = QuantumCircuit(self.resource_report["total_qubits"], name="compact_inverse_action")
        padded = np.zeros(p)
        padded[:n] = (self.encoding.input_basis.T @ b if fuse_input else b) / np.linalg.norm(b)
        qc.append(StatePreparation(padded), data)
        before, after = np.eye(p), np.eye(p)
        before[:n, :n] = self.encoding.input_basis.T
        after[:n, :n] = self.encoding.output_basis
        if not fuse_input:
            qc.unitary(before, data, label="input spectral encoding")
        prepare = StatePreparation(self.selector_weights)
        qc.append(prepare, selector)
        for j, control in enumerate(selector):
            angles = np.concatenate([np.zeros(p), -2 * (1 << j) * self.walk_angles])
            qc.append(UCRYGate(angles.tolist()), [flag] + data + [control])
        # Sign of the analytic reciprocal Chebyshev coefficient is (-1)^k.
        qc.z(selector[0])
        qc.append(prepare.inverse(), selector)
        qc.unitary(after, data, label="output spectral encoding")
        if verification_state is not None:
            target = np.asarray(verification_state, dtype=float)
            if target.shape != (n,) or not np.isfinite(target).all() or np.linalg.norm(target) == 0:
                raise ValueError("Invalid independent verification state")
            padded_target = np.zeros(p)
            padded_target[:n] = target / np.linalg.norm(target)
            qc.append(StatePreparation(padded_target).inverse(), data)
        return qc

    def operator_circuit(self):
        """Return an input-independent unitary for already prepared data qubits.

        The helper removes only the first state-preparation instruction from
        the unfused circuit. The remaining spectral basis changes, walk and LCU
        selector are independent of any particular right-hand side.
        """
        n = len(self.encoding.matrix)
        circuit = self.circuit(np.eye(1, n, 0).ravel(), fuse_input=False)
        circuit.data = circuit.data[1:]
        circuit.name = "compact_inverse_operator"
        return circuit

    def simulate(self, rhs, *, return_state=False, return_success=False):
        """Simulate SELECT gates, then project the selector onto its prepared state.

        This is exactly the all-zero sector of PREPARE-dagger, without allocating
        that dense unitary. Other sectors retain a lazy selector basis change;
        state_norm is measured before that norm-preserving change. return_state
        explicitly uncomputes every selector sector, for small native comparisons.
        No reciprocal table, polynomial evaluation or residual correction is used.
        """
        self._check_memory(dense_selector=return_state)
        b = self.encoding._rhs(rhs)
        p = self.resource_report["padded_size"]
        coefficients = np.zeros(p, dtype=complex)
        coefficients[: len(b)] = self.encoding.input_basis.T @ (b / np.linalg.norm(b))
        state = np.zeros((len(self.selector_weights), 2, p), dtype=complex)
        state[:, 0, :] = self.selector_weights[:, None] * coefficients
        for j in range(self.resource_report["selector_qubits"]):
            selected = np.flatnonzero(np.arange(len(state)) & (1 << j))
            zero, one = state[selected, 0].copy(), state[selected, 1].copy()
            c, s = np.cos((1 << j) * self.walk_angles), np.sin((1 << j) * self.walk_angles)
            state[selected, 0] = c * zero + s * one
            state[selected, 1] = -s * zero + c * one
        state[1::2] *= -1
        norm = float(np.linalg.norm(state.ravel()) ** 2)
        if return_state:
            state = np.tensordot(self.selector_unitary.conj().T, state, axes=(1, 0))
            block = state[0, 0]
        else:
            block = np.tensordot(self.selector_weights, state[:, 0], axes=(0, 0))
        probability = float(np.clip(np.vdot(block, block).real, 0, 1))
        solution = np.real_if_close(self.encoding.output_basis @ block[: len(b)]).real
        solution *= self.normalization * np.linalg.norm(b)
        result = {
            "solution": solution,
            "success_probability": probability,
            "relative_residual": float(
                np.linalg.norm(self.encoding.matrix @ solution - b) / np.linalg.norm(b)
            ),
            "state_norm": norm,
            "padding_success_weight": float(np.linalg.norm(block[len(b) :]) ** 2),
            "method": "structured statevector with lazy selector and output-basis changes",
        }
        if return_success:
            result["success_spectral_amplitudes"] = block.copy()
        if return_state:
            result["spectral_statevector"] = state
        return result


def build_compact_inverse(
    matrix, *, tolerance=1e-6, degree=None, max_degree=1023, max_memory_mib=512
):
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("tolerance must be finite and between zero and one")
    if isinstance(max_degree, bool) or not isinstance(max_degree, int) or max_degree < 1:
        raise ValueError("max_degree must be a positive integer")
    encoding = build_large_inverse_action(matrix, degree=1, max_memory_mib=max_memory_mib)
    n = len(encoding.matrix)
    # Keep the input decomposition values: reversing 1-tau*sigma loses low bits.
    sigma = encoding.singular_values
    lo, hi = float(np.min(sigma)), float(np.max(sigma))
    halfwidth = (hi - lo) / 2
    root_product = np.sqrt(lo) * np.sqrt(hi)
    sqrt_condition = np.sqrt(hi) / np.sqrt(lo)
    ratio = (sqrt_condition - 1) / (sqrt_condition + 1)
    prefactor = 1 / root_product
    if degree is None:
        if ratio < np.finfo(float).eps:
            degree = 1
        else:
            tail_factor = 2 * sqrt_condition / (1 - ratio)
            log_target = np.log(tolerance) - np.log(tail_factor)
            degree = max(1, int(np.ceil(log_target / np.log(ratio))) - 1)
    if isinstance(degree, bool) or not isinstance(degree, int) or not 1 <= degree <= max_degree:
        raise ValueError(f"Requested accuracy needs degree {degree}; budget is {max_degree}")
    estimate = _resources(n, degree)["estimated_working_bytes"]
    if estimate > max_memory_mib * 1024**2:
        raise MemoryError(
            f"Estimated {estimate / 1024**2:.1f} MiB exceeds {max_memory_mib} MiB budget"
        )
    coefficients = prefactor * np.concatenate([[1.0], 2 * (-ratio) ** np.arange(1, degree + 1)])
    selector_dim = 1 << degree.bit_length()
    weights = np.zeros(selector_dim)
    weights[: degree + 1] = np.sqrt(np.abs(coefficients) / np.sum(np.abs(coefficients)))
    padded = encoding.resource_report["padded_size"]
    angles = np.full(padded, np.pi / 2)
    if halfwidth > 0:
        # Stable endpoints without first subtracting a possibly much larger center.
        angles[:n] = 2 * np.arctan2(np.sqrt(hi - sigma), np.sqrt(sigma - lo))
    bound = 2 * sqrt_condition * ratio ** (degree + 1) / (1 - ratio)
    operation = CompactInverse(
        encoding, degree, coefficients, weights, angles, float(bound), tolerance, max_memory_mib
    )
    operation._check_memory()
    return operation
