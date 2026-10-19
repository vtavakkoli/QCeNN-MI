import io

import numpy as np
import pytest
from qiskit import qpy, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

from qcenn_mi.recurrent_quantum import (
    build_recurrent_solver,
    recurrent_circuit,
    recurrent_resources,
    simulate_recurrent_schedule,
)


def target_state(operation):
    # A reference solve is deliberately confined to the independent test oracle.
    solution = np.linalg.solve(operation.matrix, operation.rhs)
    state = np.zeros(1 << operation.data_qubits)
    state[: operation.size] = (
        operation.singular_ratio
        * operation.matrix_scale
        * solution
        / np.linalg.norm(operation.rhs)
    )
    state[operation.reference_index] = 1
    return state / np.linalg.norm(state)


def channel_reference(operation, density):
    f = operation.residual_operator
    updated = operation.no_jump @ density @ operation.no_jump.T
    updated[operation.reference_index, operation.reference_index] += np.trace(f @ density @ f.T)
    return updated


def test_compilation_uses_no_inverse_or_solution_oracle(monkeypatch):
    matrix = np.array([[2.0, 0.4], [-0.1, 1.5]])
    rhs = np.array([1.0, -0.5])
    reference = np.linalg.solve(matrix, rhs)

    def forbidden(*args, **kwargs):
        raise AssertionError("A classical solution was requested by the quantum construction")

    monkeypatch.setattr(np.linalg, "inv", forbidden)
    monkeypatch.setattr(np.linalg, "pinv", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    operation = build_recurrent_solver(matrix, rhs)
    operation.circuit(4)
    target = np.zeros(4)
    target[:2] = (
        operation.singular_ratio * operation.matrix_scale * reference / np.linalg.norm(rhs)
    )
    target[2] = 1
    assert np.linalg.norm(operation.residual_operator @ target) < 1e-14


@pytest.mark.parametrize("initial", [0, 2, 3, "complex"])
def test_coherent_update_and_device_reset_realize_same_cptp_map(initial):
    operation = build_recurrent_solver([[2, 0.4], [-0.1, 1.5]], [1, -0.5])
    if initial == "complex":
        state = np.array([1, 2j, -0.5, 1j])
        state /= np.linalg.norm(state)
    else:
        state = np.eye(4)[initial]
    density = np.outer(state, state.conj())
    expected = channel_reference(operation, density)
    for workspace in ("coherent", "reset"):
        result = operation.simulate(1, initial_state=state, workspace=workspace)
        assert result["density_matrix"] == pytest.approx(expected, abs=2e-12)
        assert result["trace"] == pytest.approx(1, abs=2e-12)
        assert np.linalg.eigvalsh(result["density_matrix"])[0] > -2e-12


def test_closed_circuit_is_unitary_and_has_no_midcircuit_readback():
    operation = build_recurrent_solver([[2, 0.4], [-0.1, 1.5]], [1, -0.5])
    circuit = operation.circuit(1, workspace="coherent")
    assert circuit.num_clbits == 0
    assert set(circuit.count_ops()) <= {"x", "unitary", "cswap"}
    unitary = Operator(circuit).data
    assert unitary.conj().T @ unitary == pytest.approx(np.eye(32), abs=2e-12)
    reset = operation.circuit(4, workspace="reset")
    for instruction in reset.data:
        assert instruction.operation.name not in {"measure", "if_else", "while_loop"}
        if instruction.operation.name == "reset":
            assert all(reset.find_bit(q).index >= operation.data_qubits for q in instruction.qubits)


@pytest.mark.parametrize(
    "matrix,rhs",
    [
        ([[2, 0.4], [0.4, 1.5]], [1, -0.5]),
        ([[2, 0.4], [-0.1, 1.5]], [-0.2, 1]),
        ([[-2, 0.2], [0.1, 1.5]], [1, 0.3]),
        ([[1.5, 0.1, 0], [-0.1, 1.3, 0.2], [0, 0.1, 1.6]], [1, -0.2, 0.5]),
    ],
)
def test_full_circuit_converges_and_obeys_static_certificate(matrix, rhs):
    operation = build_recurrent_solver(matrix, rhs)
    target = target_state(operation)
    fidelities = []
    for steps in (0, 1, 8, 32, 96):
        result = operation.simulate(steps)
        fidelity = float(np.vdot(target, result["density_matrix"] @ target).real)
        fidelities.append(fidelity)
        assert 1 - fidelity <= result["certificate"]["augmented_infidelity_upper_bound"] + 2e-11
    assert np.diff(fidelities).min() > -1e-12
    assert fidelities[-1] > 0.9999
    assert result["coherence_relative_residual"] < 1e-3


def test_target_is_fixed_and_padding_initial_state_is_not_a_false_fixed_point():
    operation = build_recurrent_solver([[2, 0.4], [-0.1, 1.5]], [1, -0.5])
    target = target_state(operation)
    result = operation.simulate(3, initial_state=target, workspace="coherent")
    assert result["density_matrix"] == pytest.approx(np.outer(target, target), abs=2e-12)
    padding = np.eye(4)[3]
    result = operation.simulate(64, initial_state=padding)
    assert result["padding_probability"] < 1e-12
    assert np.vdot(target, result["density_matrix"] @ target).real > 0.9999


def test_matrix_changes_reuse_quantum_state_without_repreparing_data():
    first = build_recurrent_solver([[2, 0.4], [-0.1, 1.5]], [1, -0.5])
    second = build_recurrent_solver([[1.8, 0.2], [0.1, 1.6]], [1, -0.5])
    schedule = [(first, 2), (second, 2)]
    density = np.zeros((4, 4), dtype=complex)
    density[2, 2] = 1
    for operation, count in schedule:
        for _ in range(count):
            density = channel_reference(operation, density)
    coherent = simulate_recurrent_schedule(schedule, workspace="coherent")
    reset = simulate_recurrent_schedule(schedule, workspace="reset")
    assert coherent["density_matrix"] == pytest.approx(density, abs=2e-12)
    assert reset["density_matrix"] == pytest.approx(density, abs=2e-12)
    assert recurrent_circuit(schedule).num_clbits == 0
    assert coherent["resources"]["classical_iterate_updates"] == 0
    assert recurrent_circuit(schedule).count_ops()["unitary"] == 4


def test_native_gates_and_qpy_preserve_final_quantum_state():
    operation = build_recurrent_solver([[2, 0.4], [-0.1, 1.5]], [1, -0.5])
    native = transpile(
        operation.circuit(3, workspace="reset"),
        basis_gates=["rz", "sx", "x", "cx", "reset"],
        optimization_level=1,
        seed_transpiler=42,
    )
    assert set(native.count_ops()) <= {"rz", "sx", "x", "cx", "reset"}
    buffer = io.BytesIO()
    qpy.dump(native, buffer)
    buffer.seek(0)
    restored = qpy.load(buffer)[0]
    backend = AerSimulator(method="density_matrix", max_parallel_threads=1)
    restored.save_density_matrix([0, 1], label="final_data")
    density = np.asarray(backend.run(restored).result().data(0)["final_data"])
    assert density == pytest.approx(operation.simulate(3)["density_matrix"], abs=2e-11)


def test_resources_and_budget_guards_precede_simulator_allocation():
    assert recurrent_resources(1000, 6)["total_qubits"] == 76
    assert recurrent_resources(1000, 6, workspace="reset")["total_qubits"] == 21
    operation = build_recurrent_solver(np.eye(2), [1, 0])
    with pytest.raises(MemoryError, match="budget"):
        operation.circuit(20)
    with pytest.raises(MemoryError, match="memory budget"):
        operation.simulate(12, workspace="coherent", max_memory_mib=1)
    with pytest.raises(MemoryError, match="Dense encoding"):
        build_recurrent_solver(np.eye(100), np.ones(100))


@pytest.mark.parametrize(
    "matrix,rhs",
    [
        ([[1, 1], [1, 1]], [1, 0]),
        ([[1, 0], [0, np.nan]], [1, 0]),
        (np.eye(2), [0, 0]),
        (np.eye(2) * 1j, [1, 0]),
        (np.eye(2), [1, 0, 0]),
        (np.ones((2, 3)), [1, 0]),
    ],
)
def test_invalid_inputs_are_rejected(matrix, rhs):
    with pytest.raises(ValueError):
        build_recurrent_solver(matrix, rhs)
