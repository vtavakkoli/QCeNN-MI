import numpy as np
import pytest
from qiskit import transpile
from qiskit.quantum_info import Statevector

from qcenn_mi.coherent_large_benchmark import fixture
from qcenn_mi.compact import build_compact_inverse


@pytest.mark.parametrize("size", [2, 10, 100, 1000])
def test_compact_accuracy_and_qubits(size):
    op = build_compact_inverse(fixture(size, "dense_spd"), degree=7)
    result = op.simulate(np.ones(size))
    assert result["relative_residual"] < 1e-6
    assert result["state_norm"] == pytest.approx(1.0, abs=1e-10)
    assert result["success_probability"] > 0.25
    assert op.resource_report["total_qubits"] == (size - 1).bit_length() + 4


@pytest.mark.parametrize("family", ["dense_spd", "dense_nonsymmetric", "signed_indefinite"])
def test_compact_matches_native_circuit(family):
    op = build_compact_inverse(fixture(10, family), degree=7)
    rhs = np.random.default_rng(42).normal(size=10)
    expected = op.simulate(rhs, return_state=True)
    qc = transpile(op.circuit(rhs), basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
    actual = np.asarray(Statevector.from_instruction(qc).data)
    basis = np.eye(16)
    basis[:10, :10] = op.encoding.output_basis
    assert actual == pytest.approx((expected["spectral_statevector"] @ basis.T).ravel(), abs=1e-9)


def test_adaptive_accuracy_fixes_previous_condition_100_failure():
    op = build_compact_inverse(np.diag(np.linspace(1.0, 100.0, 10)), tolerance=1e-6)
    result = op.simulate(np.eye(10)[0])
    assert result["relative_residual"] < 1e-6
    assert op.residual_bound <= 1e-6
    assert op.resource_report["total_qubits"] < 15


def test_degree_budget_failure_is_explicit():
    with pytest.raises(ValueError, match="budget"):
        build_compact_inverse(np.diag([1.0, 10000.0]), tolerance=1e-8, max_degree=7)


def test_no_classical_inverse_or_solve_in_compact_path(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Classical inverse/solve forbidden")

    monkeypatch.setattr(np.linalg, "inv", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    operation = build_compact_inverse(fixture(10, "dense_nonsymmetric"))
    assert operation.simulate(np.ones(10))["relative_residual"] < 1e-6
    operation.circuit(np.ones(10))


def test_same_operator_applies_to_different_prepared_input_states():
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation

    operation = build_compact_inverse(fixture(10, "dense_nonsymmetric"))
    operator = operation.operator_circuit()
    data = operation.resource_report["data_qubits"]
    for rhs in [np.ones(10), np.arange(-5.0, 5.0)]:
        circuit = QuantumCircuit(operator.num_qubits)
        padded = np.zeros(1 << data)
        padded[:10] = rhs / np.linalg.norm(rhs)
        circuit.append(StatePreparation(padded), range(data))
        circuit.compose(operator, inplace=True)
        state = Statevector.from_instruction(circuit).data
        observed = state[:10] * operation.normalization * np.linalg.norm(rhs)
        assert observed == pytest.approx(operation.simulate(rhs)["solution"], abs=1e-9)
