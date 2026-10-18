import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qcenn_mi.coherent import build_inverse_action
from qcenn_mi.coupled import readout_direction

M2 = np.array([[2.0, 0.3], [0.3, 1.5]])
M4 = np.array([[2.0, 0.1, 0, 0.1], [0.1, 1.7, 0.1, 0], [0, 0.1, 1.8, 0.1], [0.1, 0, 0.1, 2.1]])


@pytest.mark.parametrize("matrix", [M2, M4])
def test_coherent_inverse_all_columns_and_superposition(matrix):
    operation = build_inverse_action(matrix, degree=7)
    columns = []
    for rhs in np.eye(len(matrix)):
        result = operation.ideal(rhs)
        columns.append(result["solution"])
        assert 0 < result["success_probability"] <= 1
        assert result["residual"] <= operation.contraction_bound**8 + 1e-10
    inverse = np.column_stack(columns)
    assert np.linalg.norm(matrix @ inverse - np.eye(len(matrix))) < 1e-3
    rhs = np.arange(1.0, len(matrix) + 1)
    assert operation.ideal(rhs)["solution"] == pytest.approx(inverse @ rhs, abs=1e-10)


def test_degree_convergence_and_probability_normalization():
    rhs = np.array([1.0, -2.0])
    residuals = []
    for degree in (1, 3, 7):
        op = build_inverse_action(M2, degree=degree)
        result = op.ideal(rhs)
        residuals.append(result["residual"])
        probability = (
            np.linalg.norm(result["solution"] / (op.normalization * np.linalg.norm(rhs))) ** 2
        )
        assert probability == pytest.approx(result["success_probability"], abs=1e-12)
    assert np.all(np.diff(residuals) < 0)


def test_no_inverse_or_linear_solve_used_in_circuit_build(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Classical inverse/solve is forbidden in coherent construction")

    monkeypatch.setattr(np.linalg, "inv", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    result = build_inverse_action(M2, degree=3).ideal([1.0, 0.0])
    assert result["residual"] < 0.003


def test_finite_shot_magnitude_from_postselection_without_scale_refit():
    op = build_inverse_action(M2, degree=3)
    result = op.measure([1.0, 0.0], shots=32768, seed=42)
    ideal = op.ideal([1.0, 0.0])
    assert result["residual"] < 0.15
    assert result["success_probability"] == pytest.approx(ideal["success_probability"], abs=0.006)
    assert result["audit"]["postselected_ancillas"] == op.operator.num_qubits - 1


def test_multiple_ancilla_filter_requires_all_zero():
    qc = QuantumCircuit(3)
    qc.x(2)  # Last ancilla is one; checking only the adjacent ancilla would be wrong.
    with pytest.raises(RuntimeError, match="No accepted"):
        readout_direction(
            qc, postselect_ancillas=2, shots=64, backend=AerSimulator(max_parallel_threads=1)
        )


@pytest.mark.parametrize(
    "matrix", [np.array([[1.0, 2.0], [0, 1.0]]), np.array([[1.0, 2.0], [2.0, 1.0]])]
)
def test_unsupported_matrix_is_rejected(matrix):
    with pytest.raises(ValueError):
        build_inverse_action(matrix)


@pytest.mark.parametrize("rhs", [[0.0, 0.0], [1.0, 2.0, 3.0], [1j, 0], [np.nan, 0]])
def test_invalid_rhs(rhs):
    with pytest.raises(ValueError):
        build_inverse_action(M2, degree=1).circuit(rhs)


def test_simulator_fusion_matches_exported_circuit():
    from qiskit.quantum_info import Statevector

    operation = build_inverse_action(M2, degree=3)
    rhs = np.array([1.0, -2.0])
    structured = Statevector.from_instruction(operation.circuit(rhs)).data
    fused = Statevector.from_instruction(operation.circuit(rhs, simulator=True)).data
    assert structured == pytest.approx(fused, abs=1e-10)


@pytest.mark.parametrize("degree", [0, 8, True, 1.5])
def test_degree_resource_limit(degree):
    with pytest.raises(ValueError, match="degree"):
        build_inverse_action(M2, degree=degree)


def test_slow_convergence_remains_visible_for_larger_condition_number():
    operation = build_inverse_action(np.diag([1.0, 10.0]), degree=7)
    result = operation.ideal([1.0, 0.0])
    assert result["residual"] > 0.1
    assert result["residual"] <= operation.contraction_bound**8 + 1e-12
