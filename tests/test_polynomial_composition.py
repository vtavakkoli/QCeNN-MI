import numpy as np
import pytest
from qiskit.quantum_info import Statevector, partial_trace

from qcenn_mi.polynomial_composition import polynomial_composition_circuit
from qcenn_mi.polynomial_recurrent import build_polynomial_solver
from qcenn_mi.recurrent_quantum import build_recurrent_solver


@pytest.mark.parametrize("degree", [0, 1, 2, 4])
@pytest.mark.parametrize("initial", ["reference", "random", "complex"])
def test_deferred_reset_equals_polynomial_channel_without_response_synthesis(degree, initial):
    matrix = np.array([[1.0, 0.2], [-0.1, 0.6]])
    rhs = np.array([1.0, -0.5])
    base = build_recurrent_solver(matrix, rhs)
    pure = (
        np.array([0.1, -0.2, 0.8, 0.0]) if initial == "random" else np.array([0.0, 0.0, 1.0, 0.0])
    )
    pure /= np.linalg.norm(pure)
    if initial == "complex":
        pure = np.array([0.1j, -0.2, 0.8j, 0.3])
        pure /= np.linalg.norm(pure)
    circuit = polynomial_composition_circuit(base, degree, 2, initial_state=pure)
    actual = partial_trace(Statevector(circuit), list(range(2, circuit.num_qubits))).data
    solver = build_polynomial_solver(matrix, rhs, degree)
    density = np.outer(pure, pure.conj())
    reset = np.zeros(4)
    reset[2] = 1
    for _ in range(2):
        f, c = solver.residual_operator, solver.no_jump
        density = c @ density @ c.T + np.trace(f @ density @ f.T) * np.outer(reset, reset)
    assert np.max(abs(actual - density)) < 2e-11
    assert circuit.metadata["polynomial_composition"]["residual_unitary_calls"] == 2 * (degree + 1)


def test_large_degree_resource_failure_is_explicit():
    base = build_recurrent_solver([[1, 0.2], [0.1, 0.5]], [1, -1])
    with pytest.raises(MemoryError):
        polynomial_composition_circuit(base, 16, 2)


def test_device_resets_reuse_ancillas_without_resetting_data():
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    matrix = [[1.0, 0.2], [-0.1, 0.6]]
    rhs = [1.0, -0.5]
    base = build_recurrent_solver(matrix, rhs)
    circuit = polynomial_composition_circuit(base, 2, 3, workspace="reset")
    assert circuit.count_ops()["reset"] == 10
    assert all(
        circuit.find_bit(instruction.qubits[0]).index >= base.data_qubits
        for instruction in circuit.data
        if instruction.operation.name == "reset"
    )
    native = transpile(circuit, basis_gates=["u", "cx"], optimization_level=0)
    native.save_density_matrix(qubits=list(range(base.data_qubits)))
    actual = np.asarray(
        AerSimulator(method="density_matrix").run(native).result().data()["density_matrix"]
    )
    solver = build_polynomial_solver(matrix, rhs, 2)
    reference = np.diag([0.0, 0.0, 1.0, 0.0])
    expected = reference.copy()
    for _ in range(3):
        c, f = solver.no_jump, solver.residual_operator
        expected = c @ expected @ c.T + np.trace(f @ expected @ f.T) * reference
    assert np.max(abs(actual - expected)) < 2e-10
