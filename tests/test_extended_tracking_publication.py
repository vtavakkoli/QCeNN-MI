"""Formal-bound unit tests, family controls and native limitation sanity checks."""
from __future__ import annotations

import numpy as np
import pytest

from qcenn_mi.extended_tracking_benchmark import (
    extended_matrix_family,
    evaluate_case,
    percentile_bootstrap,
    run_extended,
)
from qcenn_mi.native_limitation_audit import gate_local_survival
from qcenn_mi.tracking_certificate import (
    certify_step,
    chebyshev_suppression,
    noise_budget_per_macro,
    projector_drift,
)


def test_chebyshev_suppression_shrinks_with_degree():
    values = [chebyshev_suppression(1 / 101, degree) for degree in (2, 6, 16)]
    assert 1 > values[0] > values[1] > values[2] > 0


def test_moving_target_projector_drift_is_exact():
    target = np.array([1., 0.])
    rotated = np.array([np.cos(.2), np.sin(.2)])
    assert projector_drift(target, rotated) == pytest.approx(np.sin(.2))
    assert projector_drift(None, target) == 0


def test_certificate_noise_and_drift_are_monotone():
    s = dict(degree=5, gap=.2, updates=3, reference_overlap=.5)
    baseline = certify_step(**s, drift=0, per_macro_error=0, previous_bound=.04)
    changed = certify_step(**s, drift=.06, per_macro_error=.001, previous_bound=.04)
    assert baseline.resident_bound < changed.resident_bound
    assert baseline.certification_advantage
    assert changed.certification_advantage
    assert changed.resident_bound <= 1
    assert changed.resident_bound < changed.fresh_bound


def test_no_cloning_fresh_baseline_bound_when_movement_large():
    s = certify_step(degree=5, gap=.2, updates=3, reference_overlap=.5,
                     drift=.8, per_macro_error=.01, previous_bound=.04)
    assert not s.certification_advantage
    assert s.resident_bound >= s.fresh_bound


def test_exact_cptp_tracking_inequality_random_mixed():
    from qcenn_mi.temporal_chebyshev_publication import apply_recurrent, response

    rng = np.random.default_rng(713)
    for _ in range(12):
        d = 4
        z = rng.normal(size=d)
        z /= np.linalg.norm(z)
        r = np.eye(d)[:, 0]
        q = abs(r @ z) ** 2
        if q < 1e-3:
            continue
        complement = rng.normal(size=(d, d))
        basis, _ = np.linalg.qr(np.column_stack((z, complement[:, : d - 1])))
        if np.vdot(basis[:, 0], z).real < 0:
            basis[:, 0] *= -1
        gap, degree = .2, 5
        eigenvalues = np.array([0., .2, .5, 1.0])
        b = (basis * response(eigenvalues, gap, degree)) @ basis.T
        random_state = rng.normal(size=(d, d))
        rho = random_state @ random_state.T
        rho /= np.trace(rho)
        ref = np.outer(r, r)
        before = 1 - z @ rho @ z
        after_rho = apply_recurrent(rho, b, ref, noise_per_query=1e-4, queries=10)
        actual = 1 - z @ after_rho @ z
        certificate = certify_step(
            degree=degree, gap=gap, updates=1, reference_overlap=q,
            drift=0, per_macro_error=noise_budget_per_macro(1e-4, 10, d),
            previous_bound=float(before),
        )
        assert actual <= certificate.resident_bound + 1e-9


@pytest.mark.parametrize("family", [
    "rotating", "rhs_only", "spectral_drift", "rank_two_shock", "periodic",
])
def test_extended_families_preserve_certified_singular_gap(family):
    for t in range(6):
        matrix, rhs = extended_matrix_family(
            4, 10, 23, t, 6, "abrupt", family=family
        )
        assert np.linalg.cond(matrix) <= 10 + 1e-8
        assert np.isclose(np.linalg.norm(rhs), 1)


def test_extended_pairing_verifies_actual_infidelity_below_bound():
    record = evaluate_case(
        family="rank_two_shock", size=2, condition=2, seed=23,
        snapshots=6, drift="abrupt", budget=32, noise=1e-4,
    )
    assert record["tracking_theorem"]["all_certificates_verified"]
    assert len(record["trajectory"]) == 6
    assert record["whole_algorithm_costs"][
        "resident_queries_for_one_destructive_readout_at_every_snapshot"
    ] > record["whole_algorithm_costs"][
        "fresh_queries_for_one_destructive_readout_at_every_snapshot"
    ]


def test_bootstrap_is_trajectory_clustered():
    ci = percentile_bootstrap([1, 0, 2, -1])
    assert ci["paired_mean_passed_snapshot_difference"] == .5
    assert ci["bootstrap_95_percentile_ci"][0] <= .5
    assert ci["bootstrap_95_percentile_ci"][1] >= .5


def test_survival_is_not_output_fidelity_estimate():
    report = gate_local_survival(
        {"cx": 12792, "sx": 20000}, p1=1e-4,
        p2=1e-3, readout=.02, readout_bits=4,
    )
    assert report["not_a_state_fidelity_bound"]
    assert report["synthetic_no_error_gate_event_probability"] < 1e-4


def test_extended_smoke_writes_complete_checksum_manifest(tmp_path):
    report = run_extended(output_dir=tmp_path / "extended", smoke=True)
    assert report["metrics"]["cases"] >= 10
    assert (tmp_path / "extended" / "MANIFEST.json").is_file()
