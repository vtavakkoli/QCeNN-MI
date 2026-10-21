import numpy as np
import pytest
from qiskit import transpile
from qiskit.quantum_info import Operator, Statevector, partial_trace
from qiskit_aer import AerSimulator

from qcenn_mi.chebyshev_recurrent import (
    chebyshev_response,
    chebyshev_schedule_circuit,
    filter_degree,
    filter_density_channel,
    qsp_filter_circuit,
    suppression_bound,
    synthesize_filter,
)
from qcenn_mi.recurrent_density import build_density_channel
from qcenn_mi.recurrent_quantum import build_recurrent_solver


@pytest.mark.parametrize("condition", [2, 10, 100, 1000])
def test_signed_spectrum_preserves_only_kernel_and_suppresses_positive_modes(condition):
    gap = 1 / (condition**2 + 1)
    degree = filter_degree(gap)
    xs = np.r_[0, np.geomspace(gap, 1, 600)]
    result = chebyshev_response(xs, gap, degree)
    assert abs(result[0] - 1) < 1e-12
    assert np.max(abs(result[1:])) <= suppression_bound(gap, degree) + 1e-12
    assert suppression_bound(gap, degree) <= 0.1
    assert np.any(result[1:] < 0)  # signed C is intentional, not clipped away
    assert np.max(abs(chebyshev_response(np.linspace(0, 1, 1000), gap, degree))) <= 1 + 1e-12


def test_complete_circuits_use_actual_baseline_queries_and_preserve_complex_data():
    matrix = np.array([[1.0, 0.2], [0.1, 0.6]])
    rhs = np.array([0.4, 1.0])
    base = build_recurrent_solver(matrix, rhs)
    gap = base.singular_ratio**2 / (1 + base.singular_ratio**2)
    phases = synthesize_filter(gap)
    signal = qsp_filter_circuit(base, phases)
    assert signal.count_ops().get("unitary") == 2 * phases.degree
    success = Operator(signal).data[:4, :4]
    values, vectors = np.linalg.eigh(base.residual_operator.T @ base.residual_operator)
    expected = (
        vectors * (phases.attenuation * chebyshev_response(values, gap, phases.degree))
    ) @ vectors.T
    assert np.max(abs(success - expected)) < 1e-10
    kernel = vectors[:, np.argmin(values)]
    stationary = np.outer(kernel, kernel)
    reference = np.eye(4)[:, 2]
    stationary_after = success @ stationary @ success.conj().T + np.trace(
        (np.eye(4) - success.conj().T @ success) @ stationary
    ) * np.outer(reference, reference)
    assert 1e-14 < np.max(abs(stationary_after - stationary)) < 1e-9
    initial = np.array([1, 1j, -0.3, 0.1j])
    initial /= np.linalg.norm(initial)
    density = np.outer(initial, initial.conj())
    reference = np.eye(4)[:, 2]
    for _ in range(2):
        density = success @ density @ success.conj().T + np.trace(
            (np.eye(4) - success.conj().T @ success) @ density
        ) * np.outer(reference, reference)
    coherent = chebyshev_schedule_circuit(
        [(base, phases)] * 2, workspace="coherent", initial_state=initial
    )
    actual = partial_trace(
        Statevector.from_instruction(coherent), list(range(2, coherent.num_qubits))
    ).data
    assert np.max(abs(actual - density)) < 1e-10
    reset = chebyshev_schedule_circuit(
        [(base, phases)] * 2, workspace="reset", initial_state=initial
    )
    reset.save_density_matrix([0, 1], label="data")
    backend = AerSimulator(method="density_matrix", max_parallel_threads=1)
    result = backend.run(transpile(reset, backend, optimization_level=0), shots=1).result()
    assert np.max(abs(np.asarray(result.data(0)["data"]) - density)) < 1e-10
    assert reset.num_clbits == 0


def test_ideal_channel_cptp_global_contraction_and_finite_phase_kernel_leakage():
    base = build_density_channel(np.diag([1.0, 0.1]), np.array([1.0, 0]))
    channel = filter_density_channel(base)
    diagonal = channel.no_jump_values
    reference = channel.reference
    C = np.diag(diagonal)
    F = np.diag(np.sqrt(channel.residual_eigenvalues))
    kraus = [C] + [np.outer(reference, np.eye(4)[j]) @ F for j in range(4)]
    assert np.max(abs(sum(k.T @ k for k in kraus) - np.eye(4))) < 1e-12
    choi = sum(np.outer(k.reshape(-1), k.reshape(-1).conj()) for k in kraus)
    assert np.linalg.eigvalsh(choi).min() > -1e-12
    kernel = np.argmin(base.residual_eigenvalues)
    target = np.eye(4)[:, kernel]
    rho = np.eye(4, dtype=complex) / 4
    initial = 1 - np.real(target @ rho @ target)
    for _ in range(20):
        channel.step(rho)
    assert 1 - np.real(target @ rho @ target) <= initial * 0.625**20 + 1e-12
    with pytest.raises(MemoryError):
        synthesize_filter(1e-6, max_degree=64)
    with pytest.raises(ValueError):
        filter_density_channel(base, gap=0.2)


def test_lower_bound_family_retains_the_slow_mode_from_orthogonal_initial_state():
    condition = 10
    channel = build_density_channel(np.diag([1.0, 1 / condition]), np.array([1.0, 0]))
    v = channel.basis[1]
    rho = np.outer(v, v)
    steps = 100
    for _ in range(steps):
        channel.step(rho)
    retained = float(v @ rho @ v)
    assert abs(retained - (1 - 1 / (condition**2 + 1)) ** steps) < 1e-12


def test_changing_matrix_schedule_keeps_the_resident_state():
    matrices = [np.array([[1.0, 0.2], [0.1, 0.6]]), np.array([[0.8, -0.25], [0.35, 0.45]])]
    rhs = [np.array([0.4, 1.0]), np.array([1.0, -0.2])]
    baselines = [build_recurrent_solver(m, b) for m, b in zip(matrices, rhs, strict=True)]
    gap = min(b.singular_ratio**2 / (1 + b.singular_ratio**2) for b in baselines)
    spec = synthesize_filter(gap)
    schedule = [(b, spec) for b in baselines]
    initial = np.eye(4)[:, 2]
    rho = np.outer(initial, initial)
    for b in baselines:
        B = Operator(qsp_filter_circuit(b, spec)).data[:4, :4]
        rho = B @ rho @ B.conj().T + np.trace((np.eye(4) - B.conj().T @ B) @ rho) * np.outer(
            initial, initial
        )
    circuit = chebyshev_schedule_circuit(schedule, workspace="coherent")
    actual = partial_trace(Statevector.from_instruction(circuit), range(2, circuit.num_qubits)).data
    assert np.max(abs(actual - rho)) < 1e-10
