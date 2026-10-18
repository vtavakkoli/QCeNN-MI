import numpy as np
import pytest
from qiskit import transpile
from qiskit.quantum_info import Statevector

from qcenn_mi.coherent_large import build_large_inverse_action, resources


def fixture(size):
    rng = np.random.default_rng(100 + size)
    off = rng.normal(0, 0.02 / np.sqrt(size), (size, size))
    return np.diag(np.linspace(1.5, 2.5, size)) + (off + off.T) / 2


@pytest.mark.parametrize("size", [10, 100, 1000])
def test_requested_sizes_with_padding_and_full_ancillas(size):
    m = fixture(size)
    operation = build_large_inverse_action(m)
    rhs = np.random.default_rng(42).normal(size=size)
    result = operation.simulate(rhs)
    assert result["relative_residual"] < 1e-4
    assert result["state_norm"] == pytest.approx(1.0, abs=1e-10)
    assert result["padding_success_weight"] < 1e-20
    assert 0 < result["success_probability"] < 1
    assert operation.resource_report["padded_size"] >= size


@pytest.mark.parametrize("nonsymmetric", [False, True])
def test_structured_simulator_matches_exported_qiskit_full_state(nonsymmetric):
    matrix = fixture(10)
    if nonsymmetric:
        matrix += np.triu(np.ones((10, 10)), 1) * 0.01
    op = build_large_inverse_action(matrix, degree=3)
    rhs = np.arange(1.0, 11)
    result = op.simulate(rhs, return_state=True)
    circuit = transpile(op.circuit(rhs), basis_gates=["u", "cx"], optimization_level=1)
    exact = np.asarray(Statevector.from_instruction(circuit).data)
    state = result["spectral_statevector"]
    p = op.resource_report["padded_size"]
    basis = np.eye(p)
    basis[:10, :10] = op.output_basis
    computational = (state @ basis.T).ravel()
    assert abs(np.vdot(exact, computational)) ** 2 == pytest.approx(1.0, abs=1e-9)
    assert exact[:10] * op.normalization * np.linalg.norm(rhs) == pytest.approx(
        result["solution"], abs=1e-9
    )


def test_no_classical_inverse_or_solve_in_large_path(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("inverse/solve must not be used")

    monkeypatch.setattr(np.linalg, "inv", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    op = build_large_inverse_action(fixture(10))
    assert op.simulate(np.ones(10))["relative_residual"] < 1e-4


def test_resource_and_export_guards():
    with pytest.raises(MemoryError):
        resources(1000, 7, max_memory_mib=1)
    op = build_large_inverse_action(fixture(10))
    with pytest.raises(MemoryError):
        op.circuit(np.ones(10), max_export_size=4)


def test_z_sampling_is_not_signed_inverse_tomography():
    op = build_large_inverse_action(fixture(10), degree=3)
    result = op.simulate(np.ones(10))
    shots = op.sample_z(result, shots=10000)
    assert sum(shots["conditional_counts"]) == shots["accepted"]
    assert shots["accepted"] > 0
    assert not shots["signed_inverse_reconstructed"]


@pytest.mark.parametrize(
    "matrix",
    [np.array([[np.nan, 0.0], [0.0, 1.0]]), np.diag([1.0, 0.0]), np.eye(2) * 1j],
)
def test_unsupported_cases_fail_explicitly(matrix):
    with pytest.raises(ValueError):
        build_large_inverse_action(matrix)


def test_ill_conditioning_prevents_false_universal_success():
    op = build_large_inverse_action(np.diag(np.linspace(1.0, 100.0, 10)))
    assert op.simulate(np.eye(10)[0])["relative_residual"] > 0.5


@pytest.mark.parametrize("size", [10, 1000])
@pytest.mark.parametrize("family", ["dense_nonsymmetric", "signed_indefinite"])
def test_real_nonsymmetric_and_indefinite_inverse_action(size, family):
    from qcenn_mi.coherent_large_benchmark import fixture as general_fixture

    matrix = general_fixture(size, family)
    op = build_large_inverse_action(matrix)
    rhs = np.random.default_rng(51).normal(size=size)
    result = op.simulate(rhs)
    assert op.encoding_kind == "classical real singular-value decomposition"
    assert result["relative_residual"] < 1e-4
    assert result["state_norm"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("scale", [1e-12, 1e12])
@pytest.mark.parametrize("nonsymmetric", [False, True])
def test_matrix_scaling_does_not_change_convergence(scale, nonsymmetric):
    matrix = fixture(10)
    if nonsymmetric:
        matrix += 0.01 * np.triu(np.ones((10, 10)), 1)
    matrix *= scale
    op = build_large_inverse_action(matrix)
    assert op.simulate(np.ones(10))["relative_residual"] < 1e-4
