import numpy as np
import pytest
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_aer import AerSimulator

from qcenn_mi.coupled import local_template, readout_direction, solve_coupled
from qcenn_mi.tracking import InverseTracker, polynomial_correct


def test_local_interaction_entangles_cells():
    qc = local_template([[np.pi / 2, 0], [0, 0]], [[np.pi]])
    rho = partial_trace(Statevector.from_instruction(qc), [1]).data
    assert np.trace(rho @ rho).real == pytest.approx(0.5)
    assert set(qc.count_ops()) == {"ry", "cry"}


@pytest.mark.parametrize(
    "matrix",
    [
        [[2, 0.5], [0.5, 1.5]],
        [[2, -0.2, 0.1, 0], [0.3, 1.8, 0.2, 0.1], [0, -0.1, 1.6, 0.2], [0.1, 0, 0.3, 2.1]],
        [[-2, 0.2], [0.3, 1.5]],
    ],
)
def test_coupled_inverse(matrix):
    m = np.array(matrix)
    solved = solve_coupled(m)
    assert solved.success
    assert solved.estimate == pytest.approx(np.linalg.inv(m), abs=1e-7)


def test_tomography_preserves_entangled_relative_signs():
    qc = local_template([[np.pi / 2, 0], [0, 0]], [[-np.pi]])
    state, _, _ = readout_direction(qc, shots=8192)
    expected = np.array([1, 0, 0, -1]) / np.sqrt(2)
    assert abs(state @ expected) ** 2 > 0.995


def test_backend_measurement_path():
    solved = solve_coupled([[2, 0.2], [0.2, 1.5]])
    estimate, diagnostics = solved.measure(backend=AerSimulator(), shots=4096)
    assert np.linalg.norm(solved.matrix @ estimate - np.eye(2)) < 0.08
    assert len(diagnostics) == 2


def test_polynomial_identity_and_safeguard():
    m = np.array([[2.0, 0.2], [0.1, 1.5]])
    x = np.eye(2) / 2
    e = np.eye(2) - m @ x
    corrected = polynomial_correct(m, x, order=3)
    assert np.eye(2) - m @ corrected == pytest.approx(np.linalg.matrix_power(e, 4))
    bad = -10 * np.eye(2)
    safe = polynomial_correct(m, bad)
    assert np.linalg.norm(m @ safe - np.eye(2)) <= np.linalg.norm(m @ bad - np.eye(2))


def test_tracker_warm_start_and_transactional_failure():
    tracker = InverseTracker()
    first = tracker.update([[2, 0.2], [0.2, 1.5]], time=0)
    second = tracker.update([[2.01, 0.21], [0.21, 1.49]], time=0.1)
    assert not first["warm_started"] and second["warm_started"]
    assert second["corrected_residual"] < 1e-10
    assert second["prediction_residual"] is not None
    old = tracker.estimate.copy()
    with pytest.raises(ValueError):
        tracker.update([[1, 1], [1, 1]], time=0.2)
    with pytest.raises(ValueError):
        tracker.update(np.eye(2), time=0.1)
    assert np.array_equal(old, tracker.estimate)
    assert tracker.time == 0.1


@pytest.mark.parametrize(
    "matrix", [np.eye(3), [[1, 1], [1, 1]], [[np.nan, 0], [0, 1]], np.eye(2) * 1j]
)
def test_invalid_matrices(matrix):
    with pytest.raises(ValueError):
        solve_coupled(matrix)


def test_nonconvergence_is_reported():
    solved = solve_coupled([[2, 0.5], [0.5, 1.5]], max_nfev=1)
    assert not solved.success
    tracker = InverseTracker(max_nfev=1)
    with pytest.raises(RuntimeError, match="did not converge"):
        tracker.update([[2, 0.5], [0.5, 1.5]], time=0)
    assert tracker.state is None


def test_quantum_polynomial_dilation_matches_classical_correction():
    from qcenn_mi.polynomial import quantum_polynomial_correct

    m = np.array(
        [[2.0, -0.2, 0.1, 0], [0.3, 1.8, 0.2, 0.1], [0, -0.1, 1.6, 0.2], [0.1, 0, 0.3, 2.1]]
    )
    x = np.eye(4) / 2
    result = quantum_polynomial_correct(m, x, order=3)
    expected = polynomial_correct(m, x, order=3)
    assert result.estimate == pytest.approx(expected, abs=1e-12)
    assert all(0 < p <= 1 + 1e-12 for p in result.success_probabilities)
    assert all(qc.num_qubits == 3 for qc in result.circuits)
    assert all(qc.count_ops()["unitary"] == 1 for qc in result.circuits)


def test_polynomial_finite_shot_postselection_and_audit():
    from qcenn_mi.polynomial import quantum_polynomial_correct

    m = np.array([[2.0, 0.2], [0.2, 1.5]])
    initial = m.T / (np.linalg.norm(m, 1) * np.linalg.norm(m, np.inf))
    result = quantum_polynomial_correct(m, initial, order=3)
    estimate, records = result.measure(m, shots=4096)
    assert np.linalg.norm(m @ estimate - np.eye(2)) < 0.08
    assert all(r["postselect_ancilla_zero"] for r in records)
    assert all(0 < retained <= 4096 for r in records for retained in r["retained_shots"])
    assert all(r["job_id"] and r["raw_counts"] for r in records)


def test_quantum_polynomial_rejects_zero_columns():
    from qcenn_mi.polynomial import quantum_polynomial_correct

    with pytest.raises(ValueError, match="zero column"):
        quantum_polynomial_correct(np.eye(2), np.zeros((2, 2)))
