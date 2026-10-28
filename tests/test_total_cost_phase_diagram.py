"""Regression checks for explicit total-cost quantum state-reuse frontier."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from qcenn_mi.total_cost import (
    CostWeights,
    checkpoints_for,
    native_macro_lookup,
    resource_cost,
)
from qcenn_mi.total_cost_phase_diagram import (
    Study,
    candidates,
    controlled_fixture,
    run_setting,
    run_study,
    simulate,
    snapshot_data,
)


def test_readout_schedule_always_contains_final_snapshot():
    assert checkpoints_for(12, 0) == (11,)
    assert checkpoints_for(12, 3) == (2, 5, 8, 11)
    assert checkpoints_for(10, 3) == (2, 5, 8, 9)
    assert checkpoints_for(4, 1) == (0, 1, 2, 3)
    with pytest.raises(ValueError):
        checkpoints_for(4, -1)


def test_resident_replay_copies_and_fresh_only_current_snapshots():
    weights = CostWeights(oracle_call=1, reference_preparation=5,
                          macro_reset=2, measurement=.1, compile_snapshot=10)
    checkpoints = (2, 5)
    resident = resource_cost(
        "resident", degree=4, updates=2, checkpoints=checkpoints,
        copies_per_checkpoint=3, weights=weights,
    )
    fresh = resource_cost(
        "fresh", degree=4, updates=2, checkpoints=checkpoints,
        copies_per_checkpoint=3, weights=weights,
    )
    assert resident["macro_updates"] == 54
    assert fresh["macro_updates"] == 12
    assert resident["primitive_oracle_calls"] == 432
    assert fresh["primitive_oracle_calls"] == 96
    assert resident["expected_reference_preparations"] == 6
    assert fresh["expected_reference_preparations"] == 6
    assert resident["compiled_input_snapshots"] == 6
    assert fresh["compiled_input_snapshots"] == 2
    assert resident["total_equivalent_units"] > fresh["total_equivalent_units"]


def test_direct_postselection_charges_expected_retries_for_each_copy():
    weights = CostWeights(
        oracle_call=1, reference_preparation=5, macro_reset=0,
        measurement=0, compile_snapshot=0,
    )
    cost = resource_cost(
        "direct", degree=8, updates=1, checkpoints=(2, 5),
        copies_per_checkpoint=3, weights=weights,
        accepted_probabilities={2: .5, 5: .25},
    )
    assert cost["expected_reference_preparations"] == 18
    assert cost["primitive_oracle_calls"] == 288
    assert cost["total_equivalent_units"] == 378


def test_gate_proxy_requires_matching_exact_degree_and_disables_direct():
    data = {"records": [{
        "size": 2, "condition": 10.0, "filter_degree": 15,
        "status": "compiled_small_complete_circuit",
        "native_counts_line_topology": {"cx": 1238, "sx": 400, "x": 12},
    }]}
    assert native_macro_lookup(
        data, size=2, condition=10, degree=15,
    ) == pytest.approx(1238 + 41.2)
    assert native_macro_lookup(data, size=2, condition=10, degree=16) is None
    with pytest.raises(ValueError, match="recurrent"):
        resource_cost(
            "direct", degree=15, updates=1, checkpoints=(0,),
            copies_per_checkpoint=1, weights=CostWeights(),
            accepted_probabilities={0: .5}, native_macro_equivalent=100,
        )


def test_controllable_matrix_family_zero_drift_and_singular_gap():
    for family in ("rotating", "rhs_only", "spectral_drift", "abrupt", "periodic"):
        matrix0, rhs0 = controlled_fixture(
            4, 10, 23, 0, 8, family=family, drift=0
        )
        for t in (1, 4, 7):
            matrix, rhs = controlled_fixture(
                4, 10, 23, t, 8, family=family, drift=0
            )
            assert np.allclose(matrix, matrix0)
            assert np.allclose(rhs, rhs0)
            assert np.linalg.cond(matrix) <= 10 + 1e-8
    first, _ = controlled_fixture(4, 10, 23, 0, 8, family="rotating", drift=.2)
    last, _ = controlled_fixture(4, 10, 23, 7, 8, family="rotating", drift=.2)
    assert not np.allclose(first, last)


def test_candidate_budget_is_charged_per_snapshot():
    study = Study(query_budget_per_snapshot=32, condition=2)
    for method in ("resident", "fresh", "direct"):
        for degree, updates in candidates(study, method):
            assert 2 * degree * updates <= 32


def test_simulated_persistent_states_respect_tracking_certificate():
    study = Study(horizon=4, query_budget_per_snapshot=32)
    data = snapshot_data(study, family="abrupt", drift=.12, seed=23)
    checkpoints = checkpoints_for(4, 2)
    outcome = simulate(
        study, data, method="resident", degree=3, updates=2,
        noise=1e-5, checkpoints=checkpoints,
    )
    assert len(outcome) == 2
    for row in outcome:
        assert row["augmented_infidelity"] <= row["resident_tracking_bound"] + 1e-8


def test_holdout_seeds_never_select_validation_policy():
    options = dict(
        horizon=4, query_budget_per_snapshot=32, shots_per_setting=1,
        required_coverage=.5, train_seeds=(23, 29),
    )
    a = run_setting(
        Study(**options, test_seeds=(41, 43)),
        family="rotating", drift=.05, noise=0, readout_every=0,
    )
    b = run_setting(
        Study(**options, test_seeds=(47, 53)),
        family="rotating", drift=.05, noise=0, readout_every=0,
    )
    for method in ("resident", "fresh", "direct"):
        assert a["methods"][method]["policy"] == b["methods"][method]["policy"]
        assert a["methods"][method]["status"] == b["methods"][method]["status"]


def test_strict_unattainable_accuracy_never_claims_advantage():
    study = Study(
        horizon=4, query_budget_per_snapshot=8,
        target_residual=1e-13, target_infidelity=1e-13,
        required_coverage=1.0, shots_per_setting=1,
    )
    r = run_setting(
        study, family="rotating", drift=.3, noise=.001, readout_every=1,
    )
    assert r["resident_vs_fresh_region"] == (
        "neither_recurrent_method_heldout_feasible"
    )
    assert r["resident_to_fresh_total_cost_ratio_if_both_qualified"] is None


def test_smoke_writes_phase_diagram_csv_and_checksum_manifest(tmp_path):
    report = run_study(
        output_dir=tmp_path / "cost", smoke=True, horizon=4,
        query_budget=32, shots_per_setting=1,
    )
    assert report["settings"] == 8
    assert sum(report["region_counts"].values()) == 8
    root = tmp_path / "cost"
    checksum = json.loads((root / "MANIFEST.json").read_text())["sha256"]
    assert set(checksum) == {"result.json", "summary.csv", "REPORT.md"}
    for filename, expected in checksum.items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == expected
    assert report["training_and_heldout_seeds_disjoint"]
    assert not report["hardware_tested"]
