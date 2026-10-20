import numpy as np
import pytest

from qcenn_mi.polynomial_recurrent import build_polynomial_density_channel
from qcenn_mi.polynomial_renewal import renewal_observables
from qcenn_mi.recurrent_density import DensityObserver


@pytest.mark.parametrize("degree", [0, 2, 16])
@pytest.mark.parametrize("probability", [0, 0.003])
@pytest.mark.parametrize("initial", ["reference", "basis", "random", "mixed"])
def test_fft_renewal_matches_entire_density_evolution(degree, probability, initial):
    matrix = np.array([[1, 0.3, -0.2], [0.2, 0.7, 0.1], [0, 0.2, 0.4]])
    rhs = np.array([1.0, -0.3, 0.2])
    channel = build_polynomial_density_channel(matrix, rhs, degree)
    reference = np.linalg.solve(matrix, rhs)
    times = [0, 1, 3, 17, 64]
    rows = renewal_observables(channel, 64, times, initial, reference, probability)
    observer = DensityObserver.create(channel, reference)
    if initial == "reference":
        density = channel.initial_density()
    elif initial == "mixed":
        density = np.eye(channel.dimension) / channel.dimension
    else:
        physical = np.zeros(channel.dimension)
        if initial == "basis":
            physical[0] = 1
        else:
            physical[:4] = np.random.default_rng(831 + channel.size).normal(size=4)
            physical /= np.linalg.norm(physical)
        v = channel.basis.T @ physical
        density = np.outer(v, v)
    for k in range(65):
        if k in times:
            if k == 0 and initial == "basis":
                assert rows[0]["relative_residual"] is None
                assert not rows[0]["coherence_readout_defined"]
                channel.step(density, depolarizing_probability=probability)
                continue
            actual = observer.read(density)
            found = rows[times.index(k)]
            for key in [
                "relative_residual",
                "x1",
                "reference_probability",
                "padding_probability",
                "augmented_fidelity",
            ]:
                assert abs(actual[key] - found[key]) < 2e-11
            if actual["conditional_solution_fidelity"] is not None:
                assert (
                    abs(
                        actual["conditional_solution_fidelity"]
                        - found["conditional_solution_fidelity"]
                    )
                    < 2e-11
                )
        if k < 64:
            channel.step(density, depolarizing_probability=probability)
