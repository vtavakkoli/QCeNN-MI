import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qcenn_mi.compact import build_compact_inverse
from qcenn_mi.hardware import (
    compile_native,
    logical_statevector,
    noise_model,
    read_echo_counts,
    run_verification,
    wilson,
)


def test_42_qubit_capacity_is_not_matrix_dimension():
    operation = build_compact_inverse(np.diag(np.linspace(1.5, 2.5, 100)), degree=7)
    assert operation.resource_report["data_qubits"] == 7
    assert operation.resource_report["total_qubits"] == 11
    assert operation.resource_report["total_qubits"] < 42


def test_native_layout_restoration_and_echo():
    matrix = np.array([[2.0, 0.3], [0.3, 1.5]])
    rhs = np.array([1.0, -2.0])
    operation = build_compact_inverse(matrix)
    qc = compile_native(operation.circuit(rhs))
    native = logical_statevector(qc, np.asarray(Statevector.from_instruction(qc).data))
    solution = native[:2] * operation.normalization * np.linalg.norm(rhs)
    assert np.linalg.norm(matrix @ solution - rhs) / np.linalg.norm(rhs) < 1e-6
    verifier = operation.circuit(rhs, verification_state=np.linalg.solve(matrix, rhs))
    verifier.measure_all()
    counts, _ = run_verification(compile_native(verifier), shots=2048)
    measured = read_echo_counts(counts, data_qubits=1, total_qubits=verifier.num_qubits)
    assert measured["quality_gate"]
    assert measured["observed_echo_return"] > 0.99


def test_input_fusion_is_equivalent_and_not_solution_fitting():
    matrix = np.array([[2.0, 0.3], [0.3, 1.5]])
    operation = build_compact_inverse(matrix)
    rhs = np.array([1.0, -2.0])
    fused = Statevector.from_instruction(operation.circuit(rhs)).data
    unfused = Statevector.from_instruction(operation.circuit(rhs, fuse_input=False)).data
    assert fused == pytest.approx(unfused, abs=1e-10)


def test_noise_executes_on_native_gates():
    operation = build_compact_inverse(np.diag([1.5, 2.5]))
    rhs = np.ones(2)
    verifier = operation.circuit(rhs, verification_state=np.array([1 / 1.5, 1 / 2.5]))
    verifier.measure_all()
    profile = {"one_qubit": 1e-4, "two_qubit": 1e-3, "readout": 0.01}
    backend = AerSimulator(
        method="density_matrix", noise_model=noise_model(profile), max_parallel_threads=1
    )
    counts, _ = run_verification(compile_native(verifier), backend=backend, shots=1024)
    result = read_echo_counts(counts, data_qubits=1, total_qubits=verifier.num_qubits)
    assert result["accepted"] > 20
    assert result["observed_echo_return"] > 0.5


def test_resource_gate_and_insufficient_evidence():
    with pytest.raises(ValueError, match="budget"):
        compile_native(QuantumCircuit(43), qubit_budget=42)
    result = read_echo_counts({"000": 5}, data_qubits=1, total_qubits=3)
    assert not result["quality_gate"]
    assert wilson(0, 0) == [0.0, 1.0]


def test_perfect_tiny_sample_does_not_certify_noisy_accuracy():
    result = read_echo_counts({"000": 20}, data_qubits=1, total_qubits=3)
    assert result["observed_echo_return"] == 1
    assert not result["quality_gate"]  # Lower confidence bound is below 90%.


def test_invalid_shot_budget_is_rejected():
    with pytest.raises(ValueError, match="shots"):
        run_verification(QuantumCircuit(1), shots=0)
