import numpy as np
import pytest
from qiskit import transpile
from qiskit.quantum_info import Statevector

from qcenn_mi.coherent_large_benchmark import fixture
from qcenn_mi.compact import CompactInverse, build_compact_inverse
from qcenn_mi.precision_benchmark import audit_operator, controlled_fixture


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


def test_high_precision_1000_condition_1000_without_dense_selector(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dense selector must not be constructed")

    monkeypatch.setattr(CompactInverse, "selector_unitary", property(forbidden))
    operation = build_compact_inverse(np.diag(np.geomspace(1, 1000, 1000)), tolerance=1e-12)
    result = operation.simulate(np.ones(1000))
    assert operation.degree > 255
    assert result["relative_residual"] < 1e-11
    assert result["state_norm"] == pytest.approx(1.0, abs=1e-10)
    assert operation.residual_bound <= 1e-12
    assert operation.resource_report["total_qubits"] == 21


def test_lazy_success_projection_matches_full_uncomputation():
    operation = build_compact_inverse(fixture(10, "dense_nonsymmetric"), tolerance=1e-12)
    rhs = np.random.default_rng(24).normal(size=10)
    lazy = operation.simulate(rhs, return_success=True)
    full = operation.simulate(rhs, return_state=True)
    assert lazy["success_spectral_amplitudes"] == pytest.approx(
        full["spectral_statevector"][0, 0], abs=1e-13
    )
    assert lazy["solution"] == pytest.approx(full["solution"], abs=1e-13)


def test_high_degree_precision_matches_native_statevector():
    matrix = controlled_fixture(2, "controlled_nonsymmetric", 100)
    operation = build_compact_inverse(matrix, tolerance=1e-12)
    rhs = np.array([1.0, -0.3])
    actual = Statevector.from_instruction(operation.circuit(rhs)).data
    solution = actual[:2] * operation.normalization * np.linalg.norm(rhs)
    assert np.linalg.norm(matrix @ solution - rhs) / np.linalg.norm(rhs) < 1e-10
    assert solution == pytest.approx(operation.simulate(rhs)["solution"], abs=1e-10)


@pytest.mark.parametrize(
    "family", ["controlled_spd", "controlled_nonsymmetric", "controlled_indefinite"]
)
def test_full_operator_audit_and_prescribed_condition(family):
    matrix = controlled_fixture(12, family, 100)
    operation = build_compact_inverse(matrix, tolerance=1e-12)
    assert operation.encoding.condition_number == pytest.approx(100, rel=1e-12)
    audit = audit_operator(operation)
    assert audit["operator_relative_residual_2"] < 1e-10
    assert audit["relative_inverse_error_frobenius"] < 1e-10
    assert audit["minimum_success_probability_all_inputs"] > 0
    assert audit["padding_success_weight"] == 0


def test_compact_memory_budget_checked_before_large_selector_allocation():
    with pytest.raises(MemoryError, match="budget"):
        build_compact_inverse(np.eye(16), degree=1023, max_memory_mib=0.1)


def test_positive_definite_case_can_exceed_accuracy_degree_budget():
    with pytest.raises(ValueError, match="budget is 1023"):
        build_compact_inverse(np.diag([1.0, 10000.0]), tolerance=1e-12)


def test_extremely_small_truncation_target_has_explicit_budget_failure():
    with pytest.raises(ValueError, match="budget"):
        build_compact_inverse(np.diag([1.0, 1000.0]), tolerance=np.nextafter(0.0, 1.0))


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_constant_spectrum_precision_and_scale(scale):
    operation = build_compact_inverse(scale * np.eye(10), tolerance=1e-12)
    assert operation.simulate(np.ones(10))["relative_residual"] < 1e-13
    assert operation.residual_bound == 0
