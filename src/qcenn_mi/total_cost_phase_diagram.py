"""Conditional total-cost frontier for resident vs fresh Chebyshev state preparation.

This benchmark answers *whether and under what modeled conditions* a resident
state uses fewer total resources for the SAME accuracy/coverage goal. It
performs deterministic validation-only parameter selection and independent
held-out evaluation, with explicit destructive-readout replay, postselection,
preparation, compilation and logical-noise charges. It is a small classical
density-channel simulation, not measured hardware, sampled readout, or a
quantum-speedup demonstration.

Each scenario sweeps per-snapshot matrix drift, logical per-oracle noise and
measurement interval. A single fixed (degree, updates) policy is selected
for each method on VALIDATION seeds only. A method may be classified as
feasible on heldout seeds only when it reaches the required same fraction
of strict residual+infidelity passing checkpoints. If nobody qualifies, no
winner is declared. The direct postselected control has a distinct expected
retry cost; gate-proxy mode excludes it without a matching circuit audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .recurrent_density import build_density_channel
from .temporal_chebyshev_publication import (
    apply_recurrent,
    certified_degree,
    direct_postselect,
    observe,
    physical_filter,
    target_state,
)
from .total_cost import (
    CostWeights,
    checkpoints_for,
    native_macro_lookup,
    resource_cost,
)
from .tracking_certificate import certify_step, noise_budget_per_macro, projector_drift

FAMILIES = ("rotating", "rhs_only", "spectral_drift", "abrupt", "periodic")
METHODS = ("resident", "fresh", "direct")


@dataclass(frozen=True)
class Study:
    size: int = 2
    condition: float = 2.0
    horizon: int = 12
    query_budget_per_snapshot: int = 128
    target_residual: float = 0.05
    target_infidelity: float = 0.01
    required_coverage: float = 0.90
    shots_per_setting: int = 4
    suppressions: tuple[float, ...] = (0.3, 0.1, 0.03)
    macro_counts: tuple[int, ...] = (1, 2, 3, 4, 6, 8)
    train_seeds: tuple[int, ...] = (23, 29)
    test_seeds: tuple[int, ...] = (41, 43)
    weights: CostWeights = field(default_factory=CostWeights)

    def __post_init__(self):
        if self.size < 2 or self.condition <= 1 or self.horizon < 2:
            raise ValueError("size>=2, condition>1, horizon>=2 required")
        if self.query_budget_per_snapshot < 2:
            raise ValueError("query budget too small")
        if not 0 < self.target_residual < 1 or not 0 < self.target_infidelity < 1:
            raise ValueError("accuracy thresholds must be in (0,1)")
        if not 0 < self.required_coverage <= 1:
            raise ValueError("coverage threshold must be in (0,1]")
        if self.shots_per_setting < 1:
            raise ValueError("shots per setting must be positive")
        if not self.train_seeds or not self.test_seeds or set(
            self.train_seeds
        ) & set(self.test_seeds):
            raise ValueError("train/test seeds must be nonempty and disjoint")
        if any(not 0 < s < 1 for s in self.suppressions):
            raise ValueError("suppression values must lie in (0,1)")
        if any(isinstance(k, bool) or k < 1 for k in self.macro_counts):
            raise ValueError("macro counts must be positive integers")


def controlled_fixture(size, condition, seed, snapshot, horizon, *, family, drift):
    """Fixed-conditioning generated family with a real drift-per-step control.

    Singular-value conditioning stays <= declared 'condition'; outputs
    scored only with classical solutions and never fed into a channel update.
    drift is an angle increment (radians per snapshot), not projector drift.
    The *observed projector drift* is saved separately for interpretation.
    """
    if family not in FAMILIES or not math.isfinite(drift) or drift < 0:
        raise ValueError("unknown family or invalid drift parameter")
    if snapshot < 0 or snapshot >= horizon:
        raise ValueError("snapshot out of range")
    rng = np.random.default_rng(seed + 17 * size)
    left, _ = np.linalg.qr(rng.standard_normal((size, size)))
    right, _ = np.linalg.qr(rng.standard_normal((size, size)))
    rhs0 = rng.standard_normal(size)
    rhs_delta = rng.standard_normal(size)
    phase = drift * snapshot
    theta = phase
    if family == "periodic":
        theta = drift * np.sin(2 * np.pi * snapshot / 6)
    if family == "abrupt" and snapshot >= horizon // 2:
        theta += 3 * drift
    rotation = np.eye(size)
    rotation[:2, :2] = [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ]
    effective_condition = condition
    if family == "spectral_drift":
        effective_condition = 1 + (condition - 1) * (
            0.65 + 0.35 * (1 + np.cos(phase)) / 2
        )
    singular = np.geomspace(1, 1 / effective_condition, size)
    if family == "rhs_only":
        matrix = left @ np.diag(singular) @ right.T
        rhs = rhs0 + 0.6 * np.sin(phase) * rhs_delta
    else:
        matrix = left @ np.diag(singular) @ (right @ rotation).T
        rhs = rhs0 + 0.15 * np.sin(phase) * rhs_delta
    rhs /= np.linalg.norm(rhs)
    return matrix, rhs


def snapshot_data(study: Study, *, family, drift, seed):
    """Precompute independently scored snapshot ingredients once per seed."""
    data = []
    previous = None
    for t in range(study.horizon):
        matrix, rhs = controlled_fixture(
            study.size, study.condition, seed, t, study.horizon,
            family=family, drift=drift,
        )
        channel = build_density_channel(matrix, rhs)
        solution, target = target_state(matrix, rhs, channel)
        ref = np.zeros((channel.dimension, channel.dimension))
        ref[study.size, study.size] = 1.0
        target_movement = projector_drift(previous, target)
        data.append((matrix, rhs, channel, solution, target, ref, target_movement))
        previous = target
    return data


def candidates(study: Study, method: str) -> tuple[tuple[int, int], ...]:
    gap = 1 / (1 + study.condition**2)
    degrees = sorted({
        certified_degree(gap, suppression=s)
        for s in study.suppressions
    })
    if method == "direct":
        degrees = sorted(set(degrees) | {study.query_budget_per_snapshot // 2})
        return tuple(
            (d, 1) for d in degrees
            if d >= 1 and 2 * d <= study.query_budget_per_snapshot
        )
    return tuple(
        (d, k) for d in degrees for k in study.macro_counts
        if 2 * d * k <= study.query_budget_per_snapshot
    )


def simulate(study: Study, data, *, method, degree, updates, noise, checkpoints):
    """One trajectory, exact density scoring; never measure within recurrence.

    The trajectory may be used to score arbitrary checkpoints for validation.
    The *cost accounting*, not this numerical state object, charges an
    independent complete replay for each destructive readout copy.
    """
    if not 0 <= noise < 1:
        raise ValueError("noise must be in [0,1)")
    gap = 1 / (1 + study.condition**2)
    resident = None
    records = []
    previous_target = None
    bound = None
    for t, (matrix, rhs, channel, solution, target, ref, movement) in enumerate(data):
        if method == "resident" or t in checkpoints:
            operator = physical_filter(channel, gap, degree)
            if method == "resident":
                rho = ref.copy() if resident is None else resident
                for _ in range(updates):
                    rho = apply_recurrent(rho, operator, ref, noise, 2 * degree)
                resident = rho
                accepted_probability = 1.0
            elif method == "fresh":
                rho = ref.copy()
                for _ in range(updates):
                    rho = apply_recurrent(rho, operator, ref, noise, 2 * degree)
                accepted_probability = 1.0
            else:
                rho, accepted_probability = direct_postselect(
                    operator, ref, noise, 2 * degree
                )
        else:
            rho = None
            accepted_probability = 1.0

        if t in checkpoints:
            observation = observe(
                rho, matrix, rhs, channel, solution, target,
                study.target_residual, study.target_infidelity,
            )
            records.append({
                "snapshot": t,
                "passed": bool(observation["accuracy_passed"]),
                "relative_residual": observation["relative_residual"],
                "conditional_infidelity": observation["conditional_infidelity"],
                "augmented_infidelity": (
                    observation["purified_distance"] ** 2
                    if observation["purified_distance"] is not None else None
                ),
                "postselection_success_probability": accepted_probability,
                "target_projector_drift": movement,
            })
        if method == "resident":
            # The bound uses *all* intermediate steps, not only readouts.
            cert = certify_step(
                degree=degree, gap=gap, updates=updates,
                reference_overlap=float(abs(target[study.size]) ** 2),
                drift=movement,
                per_macro_error=noise_budget_per_macro(
                    noise, 2 * degree, channel.dimension
                ),
                previous_bound=bound,
            )
            bound = cert.resident_bound
            if t in checkpoints and observation["purified_distance"] is not None:
                if observation["purified_distance"] ** 2 > bound + 5e-8:
                    raise ArithmeticError("Tracking bound violated by exact simulator")
                records[-1]["resident_tracking_bound"] = bound
                records[-1]["resident_certificate_tighter"] = (
                    cert.certification_advantage
                )
        previous_target = target
    return records


def evaluated_policy(study, data_by_seed, *, method, degree, updates, noise,
                     checkpoints, native_macro_cost=None):
    """Aggregate paired seed trajectories and exact resource charges."""
    copies = (study.size + 1) * study.shots_per_setting
    outcomes = []
    costs = []
    success_probabilities = []
    for seed, data in data_by_seed.items():
        observed = simulate(
            study, data, method=method, degree=degree, updates=updates,
            noise=noise, checkpoints=checkpoints,
        )
        accept = {
            row["snapshot"]: row["postselection_success_probability"]
            for row in observed
        } if method == "direct" else None
        cost = resource_cost(
            method, degree=degree, updates=updates,
            checkpoints=checkpoints, copies_per_checkpoint=copies,
            weights=study.weights, accepted_probabilities=accept,
            native_macro_equivalent=native_macro_cost,
        )
        costs.append(cost)
        outcomes.extend({"seed": seed, **row} for row in observed)
        if method == "direct":
            success_probabilities.extend(accept.values())
    fraction = sum(row["passed"] for row in outcomes) / len(outcomes)
    return {
        "coverage": float(fraction),
        "passed_checkpoints": sum(row["passed"] for row in outcomes),
        "evaluated_checkpoints": len(outcomes),
        "per_seed_coverage": {
            str(seed): sum(row["passed"] for row in outcomes if row["seed"] == seed)
            / len(checkpoints) for seed in data_by_seed
        },
        "mean_total_cost_per_seed": float(np.mean([
            row["total_equivalent_units"] for row in costs
        ])),
        "minimum_postselection_success": (
            float(min(success_probabilities)) if success_probabilities else None
        ),
        "mean_observed_target_drift": float(np.mean([
            row["target_projector_drift"] for row in outcomes
        ])),
        "max_observed_target_drift": float(max(
            row["target_projector_drift"] for row in outcomes
        )),
        "theorem_tighter_checkpoints": sum(
            bool(row.get("resident_certificate_tighter", False))
            for row in outcomes
        ) if method == "resident" else None,
        "mean_reference_preparations": float(np.mean([
            row["expected_reference_preparations"] for row in costs
        ])),
        "mean_oracle_calls": float(np.mean([
            row["primitive_oracle_calls"] for row in costs
        ])),
        "resource_breakdown_for_first_seed": costs[0],
        "outcomes": outcomes,
    }


def select_policy(study, *, method, train, checkpoints, noise,
                  native_evidence=None):
    """Choose minimum cost among validation-feasible policies. Never use test."""
    eligible = []
    for degree, updates in candidates(study, method):
        native_equivalent = None
        if native_evidence is not None:
            if method == "direct":
                continue  # No native compiled direct-postselect gate evidence.
            native_equivalent = native_macro_lookup(
                native_evidence, size=study.size,
                condition=study.condition, degree=degree,
            )
            if native_equivalent is None:
                continue  # Never extrapolate unsupported degree/condition.
        trial = evaluated_policy(
            study, train, method=method, degree=degree, updates=updates,
            noise=noise, checkpoints=checkpoints,
            native_macro_cost=native_equivalent,
        )
        if trial["coverage"] + 1e-12 >= study.required_coverage:
            eligible.append((trial["mean_total_cost_per_seed"], degree,
                             updates, native_equivalent, trial))
    if not eligible:
        return {"status": "no_validation_feasible_candidate", "policy": None}
    _, degree, updates, native_equivalent, trial = min(
        eligible, key=lambda r: (r[0], r[1], r[2])
    )
    return {
        "status": "validation_feasible",
        "policy": {"degree": degree, "updates_per_snapshot": updates,
                   "calls_per_snapshot": 2 * degree * updates},
        "native_macro_proxy": native_equivalent,
        "validation": {key: val for key, val in trial.items() if key != "outcomes"},
    }


def run_setting(study: Study, *, family: str, drift: float, noise: float,
                readout_every: int, native_evidence=None):
    """One independent validation/test condition with no oracle optimization leakage."""
    if family not in FAMILIES or not math.isfinite(drift) or drift < 0:
        raise ValueError("invalid matrix family/drift")
    if not 0 <= noise < 1:
        raise ValueError("invalid noise")
    checkpoints = checkpoints_for(study.horizon, readout_every)
    train = {
        seed: snapshot_data(study, family=family, drift=drift, seed=seed)
        for seed in study.train_seeds
    }
    test = {
        seed: snapshot_data(study, family=family, drift=drift, seed=seed)
        for seed in study.test_seeds
    }
    methods = {}
    for method in METHODS:
        selected = select_policy(
            study, method=method, train=train, checkpoints=checkpoints,
            noise=noise, native_evidence=native_evidence,
        )
        if selected["policy"] is None:
            methods[method] = selected
            continue
        candidate = selected["policy"]
        eval_result = evaluated_policy(
            study, test, method=method,
            degree=candidate["degree"], updates=candidate["updates_per_snapshot"],
            noise=noise, checkpoints=checkpoints,
            native_macro_cost=selected["native_macro_proxy"],
        )
        selected["heldout"] = {
            k: v for k, v in eval_result.items() if k != "outcomes"
        }
        selected["heldout_feasible"] = (
            eval_result["coverage"] + 1e-12 >= study.required_coverage
        )
        selected["heldout_observations"] = eval_result["outcomes"]
        methods[method] = selected

    viable = {
        m: methods[m] for m in METHODS if methods[m].get("heldout_feasible")
    }
    recurrent = {m: viable[m] for m in ("resident", "fresh") if m in viable}
    cost_ratio = None
    if len(recurrent) == 2:
        resident = recurrent["resident"]["heldout"]["mean_total_cost_per_seed"]
        fresh = recurrent["fresh"]["heldout"]["mean_total_cost_per_seed"]
        cost_ratio = resident / fresh if fresh > 0 else None
        if resident < fresh - 1e-10:
            region = "resident_lower_total_cost"
        elif fresh < resident - 1e-10:
            region = "fresh_lower_total_cost"
        else:
            region = "equal_total_cost"
    elif "resident" in recurrent:
        region = "resident_only_heldout_feasible"
    elif "fresh" in recurrent:
        region = "fresh_only_heldout_feasible"
    else:
        region = "neither_recurrent_method_heldout_feasible"

    all_viable_costs = {
        m: row["heldout"]["mean_total_cost_per_seed"]
        for m, row in viable.items()
    }
    cheapest = min(all_viable_costs, key=all_viable_costs.get) if all_viable_costs else None
    recorded = [
        row["target_projector_drift"]
        for data in test.values()
        for row in (
            {"target_projector_drift": item[-1]} for item in data[1:]
        )
    ]
    return {
        "scenario": {
            "family": family, "size": study.size,
            "condition": study.condition,
            "drift_radians_per_step": drift,
            "logical_noise_per_oracle_call": noise,
            "readout_every_snapshots": readout_every,
            "readout_checkpoints": list(checkpoints),
            "shots_per_setting": study.shots_per_setting,
            "independent_readout_copies_per_checkpoint":
                (study.size + 1) * study.shots_per_setting,
            "mean_heldout_target_projector_drift": float(np.mean(recorded)),
            "maximum_heldout_target_projector_drift": float(max(recorded)),
        },
        "methods": methods,
        "resident_vs_fresh_region": region,
        "resident_to_fresh_total_cost_ratio_if_both_qualified": cost_ratio,
        "least_cost_qualified_method_including_postselection": cheapest,
        "qualified_methods": sorted(viable),
        "winner_claim_scope": (
            "Conditional held-out simulator comparison under declared "
            "readout copies, gate or oracle weight model, and pass coverage. "
            "No measured device advantage or shot-based accuracy certificate."
        ),
    }


def run_study(
    *, output_dir: Path, smoke=False, size=2, condition=2.0,
    horizon=12, query_budget=128, target_residual=0.05,
    target_infidelity=0.01, required_coverage=0.90,
    shots_per_setting=4, families=None, drifts=None, noises=None,
    intervals=None, train_seeds=(23, 29), test_seeds=(41, 43),
    weights=None, native_evidence_path=None,
):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Never overwrite an existing total-cost study")
    output.mkdir(parents=True, exist_ok=True)
    study = Study(
        size=size, condition=condition, horizon=horizon,
        query_budget_per_snapshot=query_budget, target_residual=target_residual,
        target_infidelity=target_infidelity, required_coverage=required_coverage,
        shots_per_setting=shots_per_setting,
        train_seeds=tuple(train_seeds), test_seeds=tuple(test_seeds),
        weights=weights if weights is not None else CostWeights(),
    )
    families = tuple(families if families is not None else (
        ("rotating",) if smoke else ("rotating", "rhs_only", "abrupt")
    ))
    drifts = tuple(drifts if drifts is not None else (
        (0.0, 0.05) if smoke else (0, 0.01, 0.05, 0.2, 0.6)
    ))
    noises = tuple(noises if noises is not None else (
        (0.0, 2e-5) if smoke else (0, 1e-6, 1e-5, 2e-5, 5e-5, 1e-4)
    ))
    intervals = tuple(intervals if intervals is not None else (
        (0, 2) if smoke else (0, 1, 3, 6)
    ))
    if (not families or not drifts or not noises or not intervals
            or any(f not in FAMILIES for f in families)
            or any(d < 0 or not math.isfinite(d) for d in drifts)
            or any(not 0 <= n < 1 for n in noises)
            or any(i < 0 for i in intervals)):
        raise ValueError("invalid phase diagram grid")
    if native_evidence_path is None:
        evidence = None
    else:
        evidence = json.loads(Path(native_evidence_path).read_text())
        if not evidence.get("native_gates_compiled"):
            raise ValueError("Native cost proxy requires a native compilation audit")
    rows = []
    for family in families:
        for drift in drifts:
            for noise in noises:
                for every in intervals:
                    rows.append(run_setting(
                        study, family=family, drift=drift, noise=noise,
                        readout_every=every, native_evidence=evidence,
                    ))
        print(f"Total-cost study complete for family {family}", flush=True)
    counts = {
        status: sum(r["resident_vs_fresh_region"] == status for r in rows)
        for status in (
            "resident_lower_total_cost", "fresh_lower_total_cost",
            "equal_total_cost", "resident_only_heldout_feasible",
            "fresh_only_heldout_feasible",
            "neither_recurrent_method_heldout_feasible",
        )
    }
    summary = {
        "schema_version": 1,
        "profile": "smoke" if smoke else "full",
        "experiment": "validation-selected heldout total-cost frontier",
        "settings": len(rows),
        "cost_model": (
            "native gate-weighted small-diagonal-circuit proxy, not varying-input"
            if evidence is not None else "weighted logical primitive queries"
        ),
        "input_parameters": {
            **asdict(study),
            "families": families, "drifts_radians_per_step": drifts,
            "logical_noise_per_oracle_call": noises,
            "readout_intervals": intervals,
            "native_evidence_used": native_evidence_path is not None,
        },
        "region_counts": counts,
        "scenarios": rows,
        "source_input_leakage": False,
        "training_and_heldout_seeds_disjoint": True,
        "noise_model_is_logical_depolarization": True,
        "hardware_tested": False,
        "shot_based_readout_implemented": False,
        "classical_density_scorer_used_only_for_evaluation": True,
        "costs_are_configurable_assumptions_not_measured_runtime": True,
    }
    (output / "result.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    csv_rows = []
    for row in rows:
        d = dict(row["scenario"])
        d["readout_checkpoints"] = json.dumps(d["readout_checkpoints"])
        d.update({
            "resident_vs_fresh_region": row["resident_vs_fresh_region"],
            "cost_ratio_resident_over_fresh":
                row["resident_to_fresh_total_cost_ratio_if_both_qualified"],
            "cheapest_qualified_method":
                row["least_cost_qualified_method_including_postselection"],
        })
        for method in METHODS:
            item = row["methods"][method]
            d[f"{method}_status"] = item["status"]
            d[f"{method}_degree"] = (
                item["policy"]["degree"] if item["policy"] else None
            )
            d[f"{method}_updates"] = (
                item["policy"]["updates_per_snapshot"] if item["policy"] else None
            )
            d[f"{method}_heldout_coverage"] = (
                item["heldout"]["coverage"] if "heldout" in item else None
            )
            d[f"{method}_heldout_cost"] = (
                item["heldout"]["mean_total_cost_per_seed"]
                if "heldout" in item else None
            )
        csv_rows.append(d)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = [
        "# QCeNN-MI: when does resident Chebyshev tracking cost less?",
        "",
        "Exactly simulated density states and validation-only policy selection;",
        "no sampled readout accuracy, no measured native latency, no QPU.",
        "Both resident and fresh must satisfy the same heldout joint residual",
        "and infidelity coverage requirement before cost can decide.",
        "Every destructive resident readout charges a full path replay for each",
        "measurement setting/shot. Fresh may compile only its readout snapshots;",
        "both cache input gates across independent copies.",
        "",
        f"Configurations: {len(rows)}; validation seeds {list(train_seeds)};",
        f"heldout seeds {list(test_seeds)}.",
        f"Accuracy: residual <= {target_residual}, infidelity <= "
        f"{target_infidelity}, coverage >= {required_coverage}.",
        "",
        "| Region | Scenarios |", "|---|---:|",
        *[f"| {key} | {value} |" for key, value in counts.items()],
        "",
        "Use summary.csv for the full drift/noise/readout frontier and exact",
        "target-projector drift; result.json retains parameters and trajectories.",
        "An absent or negative advantage is a legitimate outcome; it must not",
        "be converted into a claimed end-to-end speedup.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    checksum = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file()
    }
    (output / "MANIFEST.json").write_text(
        json.dumps({"sha256": checksum}, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--size", type=int, default=2)
    p.add_argument("--condition", type=float, default=2)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--query-budget", type=int, default=128)
    p.add_argument("--target-residual", type=float, default=.05)
    p.add_argument("--target-infidelity", type=float, default=.01)
    p.add_argument("--required-coverage", type=float, default=.9)
    p.add_argument("--shots-per-setting", type=int, default=4)
    p.add_argument("--families", nargs="+", choices=FAMILIES)
    p.add_argument("--drifts", nargs="+", type=float)
    p.add_argument("--noises", nargs="+", type=float)
    p.add_argument("--intervals", nargs="+", type=int)
    p.add_argument("--train-seeds", nargs="+", type=int, default=[23, 29])
    p.add_argument("--test-seeds", nargs="+", type=int, default=[41, 43])
    p.add_argument("--oracle-cost", type=float, default=1)
    p.add_argument("--prep-cost", type=float, default=5)
    p.add_argument("--reset-cost", type=float, default=2)
    p.add_argument("--measurement-cost", type=float, default=.1)
    p.add_argument("--compile-cost", type=float, default=10)
    p.add_argument("--native-evidence", type=Path)
    args = p.parse_args()
    result = run_study(
        output_dir=args.output_dir, smoke=args.smoke,
        size=args.size, condition=args.condition, horizon=args.horizon,
        query_budget=args.query_budget, target_residual=args.target_residual,
        target_infidelity=args.target_infidelity,
        required_coverage=args.required_coverage,
        shots_per_setting=args.shots_per_setting,
        families=args.families, drifts=args.drifts, noises=args.noises,
        intervals=args.intervals, train_seeds=args.train_seeds,
        test_seeds=args.test_seeds,
        weights=CostWeights(
            oracle_call=args.oracle_cost,
            reference_preparation=args.prep_cost,
            macro_reset=args.reset_cost,
            measurement=args.measurement_cost,
            compile_snapshot=args.compile_cost,
        ),
        native_evidence_path=args.native_evidence,
    )
    print(json.dumps({
        "settings": result["settings"],
        "region_counts": result["region_counts"],
        "output": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
