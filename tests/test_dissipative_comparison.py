import json

import numpy as np
import pytest
from scipy.linalg import expm

from qcenn_mi.dissipative_benchmark import convergence, spectrum_schedule
from qcenn_mi.dissipative_comparison import (
    change_channel_basis,
    continuous_trajectory,
    lindblad_rhs,
    shang_channel,
    stationary_depolarized,
)
from qcenn_mi.dissipative_report import digest, tables, verify
from qcenn_mi.recurrent_density import build_density_channel


@pytest.mark.parametrize("baseline", ["matched", "shang"])
def test_continuous_integrator_matches_independent_physical_jump_liouvillian(baseline):
    matrix = np.array([[1.2, 0.3], [-0.2, 0.8]])
    rhs = np.array([1.0, -0.7])
    channel = (shang_channel if baseline == "shang" else build_density_channel)(matrix, rhs)
    d = channel.dimension
    residual = np.eye(d)
    residual[:2, :2] = matrix / np.linalg.norm(matrix, 2)
    residual[:2, 2] = -channel.singular_ratio * rhs / np.linalg.norm(rhs)
    residual[2, :] = 0
    residual /= np.sqrt(2 if baseline == "shang" else 1 + channel.singular_ratio**2)
    reference = np.eye(d)[:, 2]
    jumps = [np.outer(reference, residual[j]) for j in range(d)]

    def physical_generator(density):
        return sum(
            jump @ density @ jump.T - 0.5 * (jump.T @ jump @ density + density @ jump.T @ jump)
            for jump in jumps
        )

    liouvillian = np.column_stack(
        [physical_generator(np.eye(d * d)[i].reshape(d, d)).ravel() for i in range(d * d)]
    )
    initial = np.outer(reference, reference)
    densities, _ = continuous_trajectory(channel, [0, 0.2, 3.0], rtol=2e-12, atol=2e-14)
    for time, density in zip([0, 0.2, 3.0], densities, strict=True):
        expected = (expm(time * liouvillian) @ initial.ravel()).reshape(d, d)
        assert channel.physical_density(density) == pytest.approx(expected, abs=5e-12)


def test_discrete_update_has_the_stated_lindblad_small_step_limit():
    channel = build_density_channel([[1.2, 0.3], [-0.2, 0.8]], [1, -0.7])
    density = channel.initial_density()
    h = 1e-6
    c = np.sqrt(1 - h * channel.residual_eigenvalues)
    changed = c[:, None] * density * c[None, :]
    changed += (
        h
        * np.dot(channel.residual_eigenvalues, density.diagonal())
        * np.outer(channel.reference, channel.reference)
    )
    assert (changed - density) / h == pytest.approx(lindblad_rhs(channel, density), abs=1e-7)


@pytest.mark.parametrize("probability", [1e-7, 1e-4, 0.2, 1])
def test_exact_logical_noise_stationary_density_satisfies_channel_equation(probability):
    channel = build_density_channel([[1, 0.002], [0, 0.01]], [1, -0.7])
    stationary = stationary_depolarized(channel, probability)
    changed = stationary.copy()
    channel.step(changed, depolarizing_probability=probability)
    assert changed == pytest.approx(stationary, abs=1e-12)
    assert np.trace(stationary) == pytest.approx(1, abs=2e-10)
    assert np.linalg.eigvalsh(stationary)[0] >= -1e-14


def test_spectral_change_preserves_the_entire_mixed_density_state():
    first = build_density_channel([[1, 0.3], [-0.2, 0.8]], [1, -0.7])
    second = build_density_channel([[1.7, -0.6], [0.2, 0.03]], [1, -0.7])
    density = first.initial_density()
    for _ in range(6):
        first.step(density, depolarizing_probability=0.01)
    physical = first.physical_density(density)
    changed = change_channel_basis(first, second, density)
    assert second.physical_density(changed) == pytest.approx(physical, abs=2e-14)
    assert np.trace(changed) == pytest.approx(np.trace(density), abs=2e-14)


def test_simulation_updates_do_not_consult_a_classical_solution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A reference solve entered quantum evolution")

    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    channel = shang_channel([[1, 0.2], [-0.1, 0.7]], [1, 0.3])
    continuous_trajectory(channel, [0, 0.1, 1])
    lindblad_rhs(channel, channel.initial_density())
    stationary_depolarized(channel, 0.001)


def test_spectrum_experiment_changes_condition_scale_and_interior_singular_values():
    singular = spectrum_schedule(10, [2, 10, 100, 10, 2])
    assert singular[:, 0] / singular[:, -1] == pytest.approx([2, 10, 100, 10, 2])
    assert np.ptp(singular[:, 0]) > 0.2
    assert np.ptp(singular[:, 4]) > 0.1
    assert singular[0] == pytest.approx(singular[-1])


def test_comparison_requires_both_metrics_and_retains_late_failure():
    settings = {"residual_target": 1e-6, "conditional_infidelity_target": 1e-6}
    rows = [
        {"time": 0, "relative_residual": 1, "conditional_solution_fidelity": None},
        {"time": 1, "relative_residual": 1e-7, "conditional_solution_fidelity": 0.99},
        {"time": 2, "relative_residual": 1e-7, "conditional_solution_fidelity": 1 - 1e-8},
    ]
    assert convergence(rows, settings)["first_sustained_sampled_time"] == 2
    rows.append({"time": 3, "relative_residual": 0.01, "conditional_solution_fidelity": 1})
    assert convergence(rows, settings) == {
        "accuracy_passed": False,
        "first_sustained_sampled_time": None,
    }


def test_rehashed_table_still_must_match_raw_experiment_data(tmp_path):
    data = {"cases": [], "dynamic_cases": [], "condition_100_noise": [], "dynamic_inputs": []}
    (tmp_path / "result.json").write_text(json.dumps(data))
    for name, content in tables(data).items():
        (tmp_path / name).write_text(content)
    manifest = {"files": {p.name: digest(p) for p in tmp_path.iterdir()}}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    assert verify(tmp_path)
    (tmp_path / "matched_comparison.csv").write_text("invented\n")
    manifest["files"]["matched_comparison.csv"] = digest(tmp_path / "matched_comparison.csv")
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Table disagrees"):
        verify(tmp_path)
