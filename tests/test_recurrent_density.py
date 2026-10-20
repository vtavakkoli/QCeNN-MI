import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import depolarizing_error

from qcenn_mi.precision_benchmark import controlled_fixture
from qcenn_mi.recurrent_density import DensityObserver, build_density_channel
from qcenn_mi.recurrent_quantum import build_recurrent_solver, recurrent_circuit
from qcenn_mi.recurrent_scaling_report import digest, tables, verify


def full_circuit_density(schedule, probability=0):
    circuit = recurrent_circuit(schedule, workspace="reset")
    data_width = schedule[0][0].data_qubits
    executable = QuantumCircuit(circuit.num_qubits)
    swaps = 0
    for item in circuit.data:
        executable.append(item.operation, [circuit.find_bit(q).index for q in item.qubits])
        if item.operation.name == "cswap":
            swaps += 1
            if probability and swaps % data_width == 0:
                executable.append(
                    depolarizing_error(probability, data_width).to_instruction(),
                    list(range(data_width)),
                )
    executable.save_density_matrix(list(range(data_width)), label="final_data")
    backend = AerSimulator(method="density_matrix", max_parallel_threads=1)
    executable = transpile(executable, backend, optimization_level=0)
    return np.asarray(backend.run(executable, shots=1).result().data(0)["final_data"])


@pytest.mark.parametrize("size", [2, 3, 10])
@pytest.mark.parametrize("probability", [0, 0.003])
def test_structured_density_matches_complete_quantum_circuit(size, probability):
    matrix = controlled_fixture(size, "controlled_nonsymmetric", 2)
    rhs = np.random.default_rng(721 + size).normal(size=size)
    dense = build_recurrent_solver(matrix, rhs)
    channel = build_density_channel(matrix, rhs)
    density = channel.initial_density()
    for _ in range(3):
        channel.step(density, depolarizing_probability=probability)
    physical = channel.physical_density(density)
    native = full_circuit_density([(dense, 3)], probability)
    assert physical == pytest.approx(native, abs=3e-12)
    assert np.trace(density) == pytest.approx(1, abs=3e-12)


@pytest.mark.parametrize("probability", [0, 0.003])
def test_time_varying_covariance_keeps_the_same_quantum_state(probability):
    matrix = controlled_fixture(3, "controlled_nonsymmetric", 2)
    rhs = np.array([1.0, -0.2, 0.6])
    channel = build_density_channel(matrix, rhs)
    density = channel.initial_density()
    angles = [0.0, 0.2, -0.4, 0.1]
    previous = 0
    schedule = []
    for theta in angles:
        cosine, sine = np.cos(theta), np.sin(theta)
        rotation = np.eye(3)
        rotation[:2, :2] = [[cosine, -sine], [sine, cosine]]
        schedule.append((build_recurrent_solver(matrix @ rotation, rhs), 1))
        channel.change_rotation_basis(density, theta - previous)
        channel.step(density, depolarizing_probability=probability)
        previous = theta
    expected = full_circuit_density(schedule, probability)
    assert channel.physical_density(density, theta=angles[-1]) == pytest.approx(expected, abs=3e-12)


def test_diagnostics_include_absolute_solution_scale_and_rotation():
    matrix = controlled_fixture(10, "controlled_nonsymmetric", 2)
    rhs = np.random.default_rng(921).normal(size=10)
    channel = build_density_channel(matrix, rhs)
    reference = np.linalg.solve(matrix, rhs)
    observer = DensityObserver.create(channel, reference)
    density = channel.initial_density()
    initial = observer.read(density)
    assert initial["x1"] == pytest.approx(0, abs=1e-12)
    assert initial["relative_residual"] == pytest.approx(1, abs=1e-12)
    work = np.empty_like(density)
    for _ in range(256):
        channel.step(density, work=work)
    result = observer.read(density, return_solution=True)
    assert result["solution"] == pytest.approx(reference, abs=1e-9)
    assert result["relative_residual"] < 1e-9
    rotated = observer.read(density, theta=0.3, return_solution=True)
    assert rotated["x1"] == pytest.approx(
        np.cos(0.3) * reference[0] + np.sin(0.3) * reference[1], abs=1e-9
    )


def test_constructor_and_updates_never_request_a_classical_solution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A solution oracle entered the recurrent simulation")

    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    channel = build_density_channel([[2, 0.2], [-0.1, 1.5]], [1, -0.5])
    density = channel.initial_density()
    for _ in range(4):
        channel.step(density, depolarizing_probability=0.001)


def test_depolarization_is_padded_register_noise_and_not_trace_renormalization():
    channel = build_density_channel(np.eye(3), [1, 0, 0])
    density = 0.97 * channel.initial_density()
    channel.step(density, depolarizing_probability=1)
    assert density == pytest.approx(np.eye(4) * (0.97 / 4), abs=1e-14)


def test_complex_initial_density_remains_a_physical_state():
    channel = build_density_channel([[2, 0.2], [-0.1, 1.5]], [1, -0.5])
    state = np.array([1, 1j, -0.5, 0.2j])
    state /= np.linalg.norm(state)
    coordinates = channel.basis.T @ state
    density = np.outer(coordinates, coordinates.conj())
    for _ in range(8):
        channel.step(density, depolarizing_probability=0.001)
    assert density == pytest.approx(density.conj().T, abs=1e-13)
    assert np.linalg.eigvalsh(density)[0] > -1e-13
    assert np.trace(density).real == pytest.approx(1, abs=1e-13)


@pytest.mark.parametrize("probability", [-0.1, 1.1, np.nan])
def test_invalid_noise_is_rejected(probability):
    channel = build_density_channel(np.eye(2), [1, 0])
    with pytest.raises(ValueError):
        channel.step(channel.initial_density(), depolarizing_probability=probability)


def test_scaling_tables_cannot_be_forged_by_replacing_only_their_digest(tmp_path):
    import json

    payload = {
        "cases": [
            {
                "id": "test",
                "size": 10,
                "final": {"relative_residual": 0.02},
                "trajectory": [{"step": 0, "x1": 0.0}],
            }
        ],
        "dynamic_cases": [],
    }
    (tmp_path / "result.json").write_text(json.dumps(payload))
    for name, content in tables(payload).items():
        (tmp_path / name).write_text(content)
    manifest = {"files": {p.name: digest(p) for p in tmp_path.iterdir()}}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    assert verify(tmp_path)
    table = tmp_path / "static_summary.csv"
    table.write_text(table.read_text().replace("0.02", "0.00000001"))
    manifest["files"][table.name] = digest(table)
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Table disagrees"):
        verify(tmp_path)
