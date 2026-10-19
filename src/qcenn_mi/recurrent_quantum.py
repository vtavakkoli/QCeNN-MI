"""A recurrent quantum channel with no classical iterate reconstruction.

This is a dense circuit prototype, separate from the reusable Chebyshev inverse
operator. Given M and b, its data register converges to an augmented state
encoding M^{-1}b. Each update transfers residual information into an environment.
The environment is either retained coherently, or reset on the device. There
are no intermediate measurements, classical conditions, or optimizer updates.

Gate compilation uses dense classical SVDs of the input and residual operator;
it never computes an inverse or a solution vector. This is not an efficient
matrix-oracle implementation, a neural-network architecture, or a speedup claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit.quantum_info import Statevector, partial_trace


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def recurrent_resources(size, steps, *, workspace="coherent"):
    """Width and conservative full-circuit simulator memory, without allocation."""
    _integer(size, "size", 2)
    _integer(steps, "steps")
    if workspace not in {"coherent", "reset"}:
        raise ValueError("workspace must be 'coherent' or 'reset'")
    data = size.bit_length()  # ceil(log2(N + 1)), including a reference coordinate.
    total = data + steps * (data + 1) if workspace == "coherent" else 2 * data + 1
    state_bytes = 16 * (1 << (total if workspace == "coherent" else 2 * total))
    return {
        "size": size,
        "padded_size": 1 << data,
        "steps": steps,
        "workspace": workspace,
        "data_qubits": data,
        "total_qubits": total,
        "environment_qubits_per_step": data + 1,
        "simulator_state_bytes": state_bytes,
        "estimated_working_bytes": 6 * state_bytes,
        "intermediate_data_measurements": 0,
        "classical_iterate_updates": 0,
        "environment_resets": (steps - 1) * (data + 1)
        if workspace == "reset" and steps
        else 0,
    }


def _prepare_reference(circuit, qubits, index):
    for bit, qubit in enumerate(qubits):
        if (index >> bit) & 1:
            circuit.x(qubit)


@dataclass(frozen=True)
class RecurrentQuantumSolver:
    matrix: np.ndarray
    rhs: np.ndarray
    matrix_scale: float
    singular_ratio: float
    residual_operator: np.ndarray
    no_jump: np.ndarray
    collision_unitary: np.ndarray

    @property
    def size(self):
        return len(self.matrix)

    @property
    def data_qubits(self):
        return self.size.bit_length()

    @property
    def reference_index(self):
        return self.size

    @property
    def contraction_bound(self):
        a2 = self.singular_ratio**2
        return 1 - a2 / (2 * (1 + a2))

    def certificate(self, steps, *, reference_initial_state=True):
        """Ideal static-matrix bound on the augmented data state, not hardware noise.

        The target is proportional to [a A^{-1}|b>, 1, 0_padding], A=M/||M||.
        Its reference weight is >= 1/2. The positive residual eigenvalues are
        >= a^2/(1+a^2), giving an infidelity contraction per update. The stated
        bound is conservative; it does not apply unchanged to a moving target.
        """
        _integer(steps, "steps")
        initial = 0.5 if reference_initial_state else 1.0
        infidelity = initial * self.contraction_bound**steps
        a2 = self.singular_ratio**2
        return {
            "augmented_infidelity_upper_bound": infidelity,
            "augmented_trace_distance_upper_bound": float(np.sqrt(infidelity)),
            "infidelity_contraction_upper_bound": self.contraction_bound,
            "asymptotic_solution_probability_lower_bound": a2 / (1 + a2),
            "scope": "ideal static augmented data state; exact compiled gates",
        }

    def circuit(self, steps, *, workspace="coherent", initial_state=None, max_qubits=42):
        return recurrent_circuit(
            [(self, steps)],
            workspace=workspace,
            initial_state=initial_state,
            max_qubits=max_qubits,
        )

    def simulate(self, steps, *, workspace="reset", initial_state=None, max_memory_mib=256):
        result = simulate_recurrent_schedule(
            [(self, steps)],
            workspace=workspace,
            initial_state=initial_state,
            max_memory_mib=max_memory_mib,
        )
        result["certificate"] = self.certificate(
            steps, reference_initial_state=initial_state is None
        )
        return result


def build_recurrent_solver(matrix, rhs, *, max_encoding_dimension=32):
    """Compile a residual channel from input data, without solving Mx=b.

    Let A=M/||M||_2, a=sigma_min(M)/||M||_2, and |b>=b/||b||. Embed
    F=[A, -a|b>] with one reference column, zero reference row, and identity
    on unused coordinates, and divide by sqrt(1+a^2). Its unique null vector
    encodes the answer, but that vector is not computed by this constructor.

    The Julia dilation of F is [[sqrt(I-F^*F), -F^*],
                                [F, sqrt(I-FF^*)]].
    A controlled swap resets the jump branch to the known reference coordinate.
    Both dense SVDs are input/gate preprocessing, not iterative readback.
    """
    _integer(max_encoding_dimension, "max_encoding_dimension", 4)
    raw = np.asarray(matrix)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or len(raw) < 2:
        raise ValueError("matrix must be square with order >= 2")
    if np.iscomplexobj(raw) or np.iscomplexobj(rhs):
        raise ValueError("Only real matrices and right-hand sides are supported")
    matrix = np.asarray(raw, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("matrix must be finite")
    if rhs.shape != (len(matrix),) or not np.isfinite(rhs).all():
        raise ValueError("rhs must be finite and match matrix size")
    rhs_norm = float(np.linalg.norm(rhs))
    if not np.isfinite(rhs_norm) or rhs_norm == 0:
        raise ValueError("rhs must have a finite nonzero norm")
    padded = 1 << len(matrix).bit_length()
    if padded > max_encoding_dimension:
        raise MemoryError(
            f"Dense encoding dimension {padded} exceeds {max_encoding_dimension}; "
            "use recurrent_resources for width estimates without dense compilation"
        )
    singular = np.linalg.svd(matrix, compute_uv=False)
    scale = float(singular[0])
    if singular[-1] <= np.finfo(float).eps * len(matrix) * scale or scale == 0:
        raise ValueError("matrix must be numerically nonsingular")
    ratio = float(singular[-1] / scale)
    residual = np.eye(padded)
    residual[: len(matrix), : len(matrix)] = matrix / scale
    residual[: len(matrix), len(matrix)] = -ratio * rhs / rhs_norm
    residual[len(matrix), :] = 0
    residual /= np.sqrt(1 + ratio**2)
    left, values, right_h = np.linalg.svd(residual)
    if values[0] > 1 + 1e-12:
        raise ArithmeticError("Residual encoding is not a contraction")
    roots = np.sqrt(np.maximum(0.0, 1 - np.minimum(values, 1.0) ** 2))
    no_jump = (right_h.T * roots) @ right_h
    other_root = (left * roots) @ left.T
    dilation = np.block([[no_jump, -residual.T], [residual, other_root]])
    if not np.allclose(dilation.T @ dilation, np.eye(2 * padded), atol=2e-12, rtol=0):
        raise ArithmeticError("Residual collision dilation is not unitary")
    return RecurrentQuantumSolver(
        matrix.copy(), rhs.copy(), scale, ratio, residual, no_jump, dilation
    )


def _schedule(schedule):
    schedule = list(schedule)
    if not schedule:
        raise ValueError("schedule must contain at least one solver/steps pair")
    size = None
    for operation, steps in schedule:
        if not isinstance(operation, RecurrentQuantumSolver):
            raise ValueError("schedule entries must contain a RecurrentQuantumSolver")
        _integer(steps, "steps")
        if size is not None and operation.size != size:
            raise ValueError("scheduled matrices must have the same size")
        size = operation.size
    return schedule


def recurrent_circuit(schedule, *, workspace="coherent", initial_state=None, max_qubits=42):
    """One circuit for a static or preprogrammed matrix/RHS sequence.

    Data qubits are allocated once. Only the first state is prepared. Subsequent
    matrix snapshots change gates, never reconstruct/reprepare the data state.
    'coherent' retains all environments and uses only unitary instructions.
    'reset' discards the old environment on-device; the data register is not reset.
    Both modes have zero classical bits and no intermediate measurements.
    """
    schedule = _schedule(schedule)
    _integer(max_qubits, "max_qubits", 1)
    operation = schedule[0][0]
    steps = sum(count for _, count in schedule)
    resources = recurrent_resources(operation.size, steps, workspace=workspace)
    if resources["total_qubits"] > max_qubits:
        raise MemoryError(
            f"Circuit needs {resources['total_qubits']} qubits; budget is {max_qubits}"
        )
    data = list(range(operation.data_qubits))
    width = operation.data_qubits
    circuit = QuantumCircuit(resources["total_qubits"], name=f"recurrent_quantum_{workspace}")
    if initial_state is None:
        _prepare_reference(circuit, data, operation.reference_index)
    else:
        state = np.asarray(initial_state, dtype=complex)
        if state.shape != (1 << width,) or not np.isfinite(state).all():
            raise ValueError("initial_state must be finite and match the augmented data register")
        norm = float(np.linalg.norm(state))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("initial_state must have a finite nonzero norm")
        circuit.append(StatePreparation(state / norm), data)
    index = 0
    for operation, count in schedule:
        gate = UnitaryGate(operation.collision_unitary, label="residual collision")
        for _ in range(count):
            start = width + index * (width + 1) if workspace == "coherent" else width
            environment = list(range(start, start + width))
            flag = start + width
            if workspace == "reset" and index:
                circuit.reset(environment + [flag])
            _prepare_reference(circuit, environment, operation.reference_index)
            circuit.append(gate, data + [flag])
            for qubit, reference in zip(data, environment, strict=True):
                circuit.cswap(flag, qubit, reference)
            index += 1
    circuit.metadata = {
        "method": "residual collision channel; dense input encoding",
        "workspace": workspace,
        "schedule_steps": [count for _, count in schedule],
        "rhs_dependent_encoding": True,
        "classical_iterate_updates": 0,
        "intermediate_data_measurements": 0,
        "physical_hardware_tested": False,
    }
    return circuit


def density_diagnostics(operation, density):
    """Final-state simulator diagnostics only; these never drive circuit updates.

    Recovering these coherences on hardware would require final tomography or
    other measurements. A density matrix is not a free hardware readout.
    """
    density = np.asarray(density, dtype=complex)
    padded = 1 << operation.data_qubits
    if density.shape != (padded, padded) or not np.isfinite(density).all():
        raise ValueError("density must match the augmented data register")
    size = operation.size
    solution_probability = float(np.trace(density[:size, :size]).real)
    reference_probability = float(density[size, size].real)
    residual = operation.residual_operator
    output = {
        "density_matrix": density,
        "trace": float(np.trace(density).real),
        "purity": float(np.trace(density @ density).real),
        "solution_probability": solution_probability,
        "reference_probability": reference_probability,
        "padding_probability": float(np.trace(density[size + 1 :, size + 1 :]).real),
        "encoded_residual_expectation": float(np.trace(residual @ density @ residual.T).real),
        "conditional_solution_density": None,
        "coherence_solution": None,
        "coherence_relative_residual": None,
    }
    if solution_probability > 1e-15:
        output["conditional_solution_density"] = density[:size, :size] / solution_probability
    if reference_probability > 1e-15:
        scale = np.linalg.norm(operation.rhs) / (operation.singular_ratio * operation.matrix_scale)
        estimate = scale * density[:size, size] / reference_probability
        output["coherence_solution"] = estimate
        output["coherence_relative_residual"] = float(
            np.linalg.norm(operation.matrix @ estimate - operation.rhs)
            / np.linalg.norm(operation.rhs)
        )
    return output


def simulate_recurrent_schedule(
    schedule, *, workspace="reset", initial_state=None, max_memory_mib=256
):
    """Execute the complete circuit once; no stepwise classical state feedback.

    The coherent mode simulates its full statevector and traces out environments
    only after execution. The reset mode runs the complete Aer density-matrix
    circuit. A simulator save instruction is appended only at the end.
    """
    schedule = _schedule(schedule)
    if not np.isfinite(max_memory_mib) or max_memory_mib <= 0:
        raise ValueError("max_memory_mib must be finite and positive")
    resources = recurrent_resources(
        schedule[0][0].size, sum(count for _, count in schedule), workspace=workspace
    )
    if resources["estimated_working_bytes"] > max_memory_mib * 1024**2:
        raise MemoryError("Full-circuit simulation exceeds the declared memory budget")
    circuit = recurrent_circuit(
        schedule,
        workspace=workspace,
        initial_state=initial_state,
        max_qubits=resources["total_qubits"],
    )
    data_width = schedule[0][0].data_qubits
    if workspace == "coherent":
        state = Statevector.from_instruction(circuit)
        environment = list(range(data_width, circuit.num_qubits))
        density = partial_trace(state, environment).data
        method = "Qiskit full unitary statevector; final partial trace"
    else:
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        backend = AerSimulator(method="density_matrix", max_parallel_threads=1)
        executable = circuit.copy()
        executable.save_density_matrix(list(range(data_width)), label="data_density")
        executable = transpile(executable, backend, optimization_level=0, seed_transpiler=42)
        result = backend.run(executable, shots=1, seed_simulator=42).result()
        if not result.success:
            raise RuntimeError(f"Recurrent simulator execution failed: {result.status}")
        density = np.asarray(result.data(0)["data_density"])
        method = "Aer full ancilla-reset density-matrix circuit; final reduced state"
    output = density_diagnostics(schedule[-1][0], density)
    output.update(resources=resources, method=method, physical_hardware_tested=False)
    return output
