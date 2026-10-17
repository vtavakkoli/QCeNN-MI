import numpy as np
import pytest

from qcenn_mi.scaling import fixture, solve_block_hybrid
from qcenn_mi.tracking import InverseTracker


@pytest.mark.parametrize("shots", [None, 2048])
@pytest.mark.parametrize("size", [10, 100])
@pytest.mark.parametrize("symmetric", [True, False])
def test_large_block_hybrid_full_matrix_residual(size, symmetric, shots):
    m = fixture(size, symmetric=symmetric)
    result = solve_block_hybrid(m, shots=shots)
    assert result["converged"]
    assert result["quantum_blocks"] == size // 2
    assert np.linalg.norm(m @ result["estimate"] - np.eye(size)) < 1e-8
    assert np.all(np.diff(result["residual_history"]) <= 0)
    assert result["estimate"] == pytest.approx(np.linalg.inv(m), abs=1e-8)


def test_dynamic_2x2_noncommuting_matrices():
    tracker = InverseTracker()
    for t in np.linspace(0, 2 * np.pi, 11):
        m = np.array(
            [
                [2 + 0.2 * np.sin(t), 0.3 * np.cos(t)],
                [0.2 * np.sin(t + 0.4), 1.5 + 0.15 * np.cos(t)],
            ]
        )
        record = tracker.update(m, time=t)
        assert record["quantum_residual"] < 1e-7
        assert np.linalg.norm(m @ tracker.estimate - np.eye(2)) < 1e-8


def test_large_solver_does_not_hide_failed_convergence():
    result = solve_block_hybrid(fixture(10), max_steps=0)
    assert not result["converged"]
    assert len(result["residual_history"]) == 1


def test_shot_based_quantum_seeds():
    result = solve_block_hybrid(fixture(10), shots=2048)
    assert result["converged"]
    assert max(result["block_residuals"]) < 0.15


def test_singular_seed_block_is_explicit_failure():
    m = np.array([[0.0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    with pytest.raises(ValueError, match="nonsingular"):
        solve_block_hybrid(m)
