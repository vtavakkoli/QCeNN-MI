import numpy as np
import pytest
from qiskit.quantum_info import Statevector, partial_trace

from qcenn_mi.polynomial_recurrent import (
    accelerate_density_channel,
    accelerated_gap,
    build_polynomial_solver,
    polynomial_response,
)
from qcenn_mi.recurrent_density import build_density_channel
from qcenn_mi.recurrent_quantum import build_recurrent_solver, recurrent_circuit


@pytest.mark.parametrize("degree", [0, 1, 2, 4, 8, 16])
def test_polynomial_identity_cptp_kernel_and_independent_complete_circuit(degree):
    matrix = np.array([[1.0, 0.3], [-0.2, 0.6]])
    rhs = np.array([0.7, -0.4])
    baseline = build_recurrent_solver(matrix, rhs)
    solver = build_polynomial_solver(matrix, rhs, degree)
    gram = baseline.residual_operator.T @ baseline.residual_operator
    response = np.eye(len(gram)) - np.linalg.matrix_power(np.eye(len(gram)) - gram, degree + 1)
    assert np.allclose(solver.residual_operator.T @ solver.residual_operator, response, atol=3e-14)
    assert np.allclose(solver.no_jump, np.linalg.matrix_power(baseline.no_jump, degree + 1))
    kraus = [solver.no_jump] + [
        np.outer(np.eye(len(gram))[solver.reference_index], row) for row in solver.residual_operator
    ]
    assert np.allclose(sum(k.T @ k for k in kraus), np.eye(len(gram)), atol=3e-14)
    choi = sum(np.outer(k.ravel(order="F"), k.ravel(order="F")) for k in kraus)
    assert np.linalg.eigvalsh(choi)[0] > -2e-14
    target = np.zeros(len(gram))
    target[:2] = (
        solver.singular_ratio
        * solver.matrix_scale
        * np.linalg.solve(matrix, rhs)
        / np.linalg.norm(rhs)
    )
    target[2] = 1
    target /= np.linalg.norm(target)
    assert np.linalg.norm(solver.residual_operator @ target) < 2e-14
    density = np.outer(target, target)
    assert np.allclose(sum(k @ density @ k.T for k in kraus), density)
    initial = np.array([1, -2, 0.7, 0.3], dtype=float)
    initial /= np.linalg.norm(initial)
    expected = np.outer(initial, initial)
    for _ in range(3):
        expected = sum(k @ expected @ k.T for k in kraus)
    circuit = solver.circuit(3, workspace="coherent", initial_state=initial)
    actual = partial_trace(Statevector(circuit), list(range(2, circuit.num_qubits))).data
    assert np.max(abs(actual - expected)) < 3e-13


def test_constructor_never_uses_inverse_or_solution_oracle(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("solution oracle used by constructor")

    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    solver = build_polynomial_solver([[1, 0.2], [0.4, 0.7]], [1, -1], 8)
    assert solver.polynomial_degree == 8


def test_zero_degree_exact_baseline_and_gap_bounds():
    base = build_recurrent_solver([[1, 0.3], [0.2, 0.8]], [1, 1])
    current = build_polynomial_solver(base.matrix, base.rhs, 0)
    assert np.array_equal(base.collision_unitary, current.collision_unitary)
    for ratio in [1, 0.5, 0.1, 0.01, 0.001]:
        g = ratio**2 / (1 + ratio**2)
        for degree in [0, 1, 4, 16]:
            gm = accelerated_gap(ratio, degree)
            assert g - 1e-15 <= gm <= min(1, (degree + 1) * g) + 1e-15


def test_moving_matrix_rhs_preserves_data_and_matches_circuit():
    matrices = [np.array([[1, 0.2], [-0.1, 0.8]]), np.array([[0.7, -0.2], [0.3, 1]])]
    rhs = [np.array([1, -0.5]), np.array([0.2, 1])]
    solvers = [
        build_polynomial_solver(m, b, d) for m, b, d in zip(matrices, rhs, [1, 4], strict=True)
    ]
    state = np.array([0, 0, 1, 0.0], dtype=float)
    density = np.outer(state, state)
    for solver in solvers:
        f, c = solver.residual_operator, solver.no_jump
        density = c @ density @ c.T + np.trace(f @ density @ f.T) * np.outer(state, state)
    circuit = recurrent_circuit([(s, 1) for s in solvers], workspace="coherent")
    actual = partial_trace(Statevector(circuit), list(range(2, circuit.num_qubits))).data
    assert np.max(abs(actual - density)) < 3e-13


def test_reduced_response_and_noise_match_physical_kraus_update():
    matrix = np.array([[1, 0.2], [-0.3, 0.6]])
    rhs = np.array([1.0, -0.5])
    base = build_density_channel(matrix, rhs)
    for degree in [0, 2, 16]:
        spectral = accelerate_density_channel(base, degree)
        solver = build_polynomial_solver(matrix, rhs, degree)
        rho = spectral.initial_density()
        spectral.step(rho, depolarizing_probability=0.003)
        actual = spectral.physical_density(rho)
        reference = np.eye(4)[2]
        initial = np.outer(reference, reference)
        f, c = solver.residual_operator, solver.no_jump
        expected = c @ initial @ c.T + np.trace(f @ initial @ f.T) * initial
        expected = 0.997 * expected + 0.003 * np.eye(4) / 4
        assert np.max(abs(actual - expected)) < 3e-14


@pytest.mark.parametrize("degree", [-1, True, 1.5])
def test_degree_domain(degree):
    with pytest.raises(ValueError):
        polynomial_response([0.1, 0.2], degree)
