"""Expanded paired temporal channel benchmark with theorem certificates and costs.

Exact classical density-channel simulations; no native-QSP scaling, hardware
demonstration, or fair postselection-vs-unconditional advantage is claimed.
The original 96-case manuscript experiment remains untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from .recurrent_density import build_density_channel
from .temporal_chebyshev_publication import (
    matrix_family,
    run_case,
    target_state,
)
from .tracking_certificate import (
    certify_step,
    noise_budget_per_macro,
    projector_drift,
)

FAMILIES = ("rotating", "rhs_only", "spectral_drift", "rank_two_shock", "periodic")
METHODS = ("resident", "fresh", "one_shot")


def extended_matrix_family(size, condition, seed, snapshot, snapshots, drift, *, family):
    """Reproducible dense fixtures with controlled conditioning and no target leakage."""
    if family not in FAMILIES:
        raise ValueError("unknown matrix family")
    if family == "rotating":
        return matrix_family(size, condition, seed, snapshot, snapshots, drift)
    rng = np.random.default_rng(seed + 7 * size)
    left, _ = np.linalg.qr(rng.standard_normal((size, size)))
    right, _ = np.linalg.qr(rng.standard_normal((size, size)))
    base_rhs = rng.standard_normal(size)
    direction = rng.standard_normal(size)
    progress = snapshot / (snapshots - 1)
    jumped = int(snapshot >= snapshots // 2)
    jump = 1.1 * jumped if drift == "abrupt" else 0.0
    strength = 0.10 if drift == "slow" else 0.75
    theta = strength * progress + jump
    if family == "periodic":
        theta = (0.15 if drift == "slow" else 0.9) * np.sin(
            4 * np.pi * progress
        ) + jump
    rotation = np.eye(size)
    rotation[:2, :2] = (
        (np.cos(theta), -np.sin(theta)),
        (np.sin(theta), np.cos(theta)),
    )
    effective = condition
    if family == "spectral_drift":
        # Always <= the declared gap condition; track a nonconstant spectrum.
        effective = 1 + (condition - 1) * (0.6 + 0.4 * (
            1 + np.cos(2 * np.pi * progress + jump)
        ) / 2)
    singular = np.geomspace(1, 1 / effective, size)
    if family == "rhs_only":
        matrix = left @ np.diag(singular) @ right.T
        rhs = base_rhs + (0.05 if drift == "slow" else 0.9) * (
            progress + jumped
        ) * direction
    elif family == "rank_two_shock":
        # Left rotation changes the matrix by rank at most two.
        matrix = (left @ rotation) @ np.diag(singular) @ right.T
        rhs = base_rhs + 0.08 * np.sin(3 * progress) * direction
    else:
        matrix = left @ np.diag(singular) @ (right @ rotation).T
        rhs = base_rhs + 0.12 * np.sin(5 * progress + jump) * direction
    rhs /= np.linalg.norm(rhs)
    return matrix, rhs


def evaluate_case(*, family, size, condition, seed, snapshots, drift, budget, noise):
    def matrix_generator(n, kappa, random_seed, step, horizon, change):
        return extended_matrix_family(
            n, kappa, random_seed, step, horizon, change, family=family
        )

    result = run_case(
        size=size, condition=condition, seed=seed, snapshots=snapshots,
        drift=drift, query_budget=budget, noise_per_query=noise,
        matrix_generator=matrix_generator,
    )
    spec = result["case"]
    spec["family"] = family
    previous_target = None
    previous_bound = None
    certificates = []
    advantage = 0
    for t, record in enumerate(result["trajectory"]):
        matrix, rhs = matrix_generator(size, condition, seed, t, snapshots, drift)
        channel = build_density_channel(matrix, rhs)
        _, target = target_state(matrix, rhs, channel)
        drift_norm = projector_drift(previous_target, target)
        q = float(abs(target[size]) ** 2)
        tau = noise_budget_per_macro(
            noise, 2 * spec["degree"], channel.dimension
        )
        certificate = certify_step(
            degree=spec["degree"], gap=1 / (1 + condition**2),
            updates=spec["macro_updates"],
            reference_overlap=q, drift=drift_norm,
            per_macro_error=tau, previous_bound=previous_bound,
        )
        bound = certificate.resident_bound
        measured = float(record["observations"]["resident"]["purified_distance"] ** 2)
        if measured > bound + 3e-8:
            raise ArithmeticError(
                f"Tracking theorem violated: observed {measured}, certificate {bound}"
            )
        fresh_observed = float(
            record["observations"]["fresh"]["purified_distance"] ** 2
        )
        if fresh_observed > certificate.fresh_bound + 3e-8:
            raise ArithmeticError("Fresh-channel certificate violated")
        row = {
            "snapshot": t,
            "target_projector_drift": drift_norm,
            "reference_target_overlap": q,
            "contraction": certificate.alpha,
            "spectral_suppression": certificate.spectral_suppression,
            "per_macro_half_trace_noise_bound": tau,
            "resident_infidelity_upper_bound": bound,
            "fresh_infidelity_upper_bound": certificate.fresh_bound,
            "resident_observed_infidelity": measured,
            "fresh_observed_infidelity": fresh_observed,
            "resident_certificate_tighter": certificate.certification_advantage,
            "verified": True,
        }
        record["tracking_certificate"] = row
        certificates.append(row)
        advantage += certificate.certification_advantage
        previous_target = target
        previous_bound = bound
    # Charging single-copy state preparation: new reference state each fresh
    # snapshot, one resident trajectory preparation; no state cloning assumed.
    # Measuring the resident state each time consumes it: repeated snapshots
    # require full path replay or an independent hardware trajectory per copy.
    per = spec["recurrent_queries_charged"]
    expected_direct = [
        row["observations"]["one_shot"][
            "expected_queries_one_accepted_preparation"
        ] for row in result["trajectory"]
    ]
    costs = {
        "logical_oracle_calls_per_snapshot": per,
        "resident_unmeasured_path_queries": snapshots * per,
        "fresh_one_copy_each_snapshot_queries": snapshots * per,
        "resident_initial_reference_preparations_unmeasured": 1,
        "fresh_reference_preparations_one_copy_each_snapshot": snapshots,
        "direct_expected_queries_one_accepted_each_snapshot": float(
            sum(expected_direct)
        ),
        "direct_expected_initial_preparations_one_accepted_each_snapshot": float(
            sum(per_row["observations"]["one_shot"][
                "expected_queries_one_accepted_preparation"
            ] / spec["one_shot_queries_charged"] for per_row in result["trajectory"])
        ),
        "resident_queries_for_one_destructive_readout_at_every_snapshot": int(
            per * snapshots * (snapshots + 1) // 2
        ),
        "fresh_queries_for_one_destructive_readout_at_every_snapshot": int(
            per * snapshots
        ),
        "resident_readout_replay_assumption": (
            "Replay of earlier time-dependent inputs or N independent "
            "trajectory copies; impossible to clone unknown quantum states. "
            "The unmeasured path is not itself readable at every snapshot."
        ),
        "state_preparation_and_compilation_not_counted_in_oracle_queries": True,
        "full_vector_readout_not_implemented": True,
    }
    result["tracking_theorem"] = {
        "model": "ideal rank-one fixed-point CPTP map plus bounded logical noise",
        "certification_advantage_snapshots": int(advantage),
        "all_certificates_verified": True,
        "per_snapshot": certificates,
        "benefit_is_bound_comparison_not_quantum_speedup": True,
    }
    result["whole_algorithm_costs"] = costs
    return result


def percentile_bootstrap(differences, *, seed=1729, repetitions=1200):
    # Resample independent trajectory configurations, not correlated snapshots.
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True)
    means = samples.mean(axis=1)
    return {
        "paired_mean_passed_snapshot_difference": float(values.mean()),
        "bootstrap_95_percentile_ci": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "resampling_unit": "complete matched trajectory",
        "confidence_interpretation": "descriptive for fixed synthetic family grid",
    }


def run_extended(*, output_dir, smoke=False):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Extended results must use a fresh directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    if smoke:
        sizes, conditions, budgets, horizons = (2,), (2,), (32,), (4, 6)
        noises, seeds = (0.0, 0.0001), (23, 29)
        families = ("rotating", "rhs_only", "spectral_drift")
    else:
        sizes, conditions, budgets, horizons = (2, 4), (2, 10), (32, 64, 128), (6, 12)
        noises, seeds = (0.0, 1e-5, 1e-4, 1e-3), (23, 29, 37)
        families = FAMILIES
    records = []
    rows = []
    for family in families:
        for size in sizes:
            for condition in conditions:
                for budget in budgets:
                    for horizon in horizons:
                        for drift in ("slow", "abrupt"):
                            for noise in noises:
                                for seed in seeds:
                                    from .temporal_chebyshev_publication import certified_degree

                                    degree = certified_degree(1 / (1 + condition**2))
                                    if 2 * degree > budget:
                                        continue
                                    result = evaluate_case(
                                        family=family, size=size, condition=condition,
                                        seed=seed, snapshots=horizon, drift=drift,
                                        budget=budget, noise=noise,
                                    )
                                    counts = {
                                        name: result["summary"][name]["passed"]
                                        for name in METHODS
                                    }
                                    row = {
                                        **result["case"],
                                        "resident_passed": counts["resident"],
                                        "fresh_passed": counts["fresh"],
                                        "one_shot_passed": counts["one_shot"],
                                        "paired_delta": counts["resident"] - counts["fresh"],
                                        "certification_advantage_snapshots":
                                            result["tracking_theorem"][
                                                "certification_advantage_snapshots"
                                            ],
                                        "one_shot_expected_accepted_queries": result[
                                            "whole_algorithm_costs"
                                        ]["direct_expected_queries_one_accepted_each_snapshot"],
                                    }
                                    records.append(result)
                                    rows.append(row)
        print(f"Completed extended family: {family}", flush=True)
    if not rows:
        raise ValueError("No affordable extended cases")
    differences = [r["paired_delta"] for r in rows]
    slices = defaultdict(list)
    for row in rows:
        for key in ("family", "drift", "size", "condition",
                    "logical_noise_per_oracle_call", "snapshots"):
            slices[f"{key}:{row[key]}"].append(row["paired_delta"])
    metrics = {
        "cases": len(rows),
        "resident_better_cases": sum(v > 0 for v in differences),
        "ties": sum(v == 0 for v in differences),
        "fresh_better_cases": sum(v < 0 for v in differences),
        "resident_passing_snapshots": sum(r["resident_passed"] for r in rows),
        "fresh_passing_snapshots": sum(r["fresh_passed"] for r in rows),
        "one_shot_passing_snapshots": sum(r["one_shot_passed"] for r in rows),
        "total_snapshots_per_method": sum(r["snapshots"] for r in rows),
        "paired_bootstrap": percentile_bootstrap(differences),
        "by_slice": {
            key: {"cases": len(values), "paired_mean": float(statistics.mean(values)),
                  "resident_wins": sum(x > 0 for x in values),
                  "fresh_wins": sum(x < 0 for x in values)}
            for key, values in sorted(slices.items())
        },
    }
    report = {
        "schema_version": 1,
        "benchmark": "expanded independent paired time-varying channel experiment",
        "profile": "smoke" if smoke else "full",
        "results": records,
        "metrics": metrics,
        "physical_hardware_tested": False,
        "native_gate_noise_simulated_in_this_module": False,
        "logical_noise_only": True,
        "statistical_scope": (
            "Bootstrap CI is over synthetic trajectory configurations; "
            "not evidence of real-world generalization."
        ),
        "canonical_publication_table_modified": False,
    }
    (output_dir / "result.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Extended matched Chebyshev tracking evidence",
        "",
        "Classical density-channel simulator; native-gate benchmarking is separate.",
        "No quantum speedup, large native circuit, or calibrated hardware claim.",
        "",
        f"Cases: {metrics['cases']}, snapshots per method: "
        f"{metrics['total_snapshots_per_method']}.",
        f"Resident vs fresh: {metrics['resident_better_cases']} wins, "
        f"{metrics['ties']} ties, {metrics['fresh_better_cases']} losses.",
        f"Resident/fresh/one-shot pass totals: "
        f"{metrics['resident_passing_snapshots']} / "
        f"{metrics['fresh_passing_snapshots']} / "
        f"{metrics['one_shot_passing_snapshots']}.",
        "One-shot postselection expected retries and preparation costs are charged",
        "separately. Destructive readout removes any free persistent-state reuse.",
        "The theorem certificates are sufficient upper bounds, not a theorem of",
        "uniform resident superiority. See result.json and summary.csv.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    checksums = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output_dir.iterdir()) if p.is_file()
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps({"sha256": checksums}, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_extended(output_dir=args.output_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
