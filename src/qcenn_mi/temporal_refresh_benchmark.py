"""Measure useful refresh intervals, holding age, acquisition error, and failures.

Model time is dimensionless. Wall-clock compilation/simulation timings are
reported separately and cannot establish a physical real-time deadline.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.linalg import solve

from .compact import build_compact_inverse
from .experiment_io import configuration, save_result
from .time_varying_benchmark import simulated_operator, tracking_bound, trajectory_bases

DEFAULTS = {
    "sizes": [2, 10, 100],
    "conditions": [10, 100, 1000],
    "families": ["slow_spd", "slow_nonsymmetric", "fast_nonsymmetric"],
    "ages": [0.0, *np.geomspace(1e-9, 0.25, 19).tolist()],
    "targets": [1e-2, 1e-3, 1e-4],
    "noise_to_gap_ratios": [0.0, 1e-5],
    "update_times": [0.0, 0.5],
    "seed": 1729,
    "polynomial_tolerance": 1e-10,
    "native_timing_max_size": 10,
    "native_timing_max_condition": 10,
}


def refresh_matrix(t, family, bases, condition):
    """Known finite-horizon gap >= exp(-0.5)/condition and Lipschitz bound."""
    if family not in DEFAULTS["families"] or not 0 <= t <= 1 or condition < 1:
        raise ValueError("Unsupported trajectory or parameters")
    left, right = (b.copy() for b in bases)
    rate = 4.0 if family == "fast_nonsymmetric" else 0.2

    def rotated(q, angle):
        c, s = np.cos(angle), np.sin(angle)
        q[:, [0, -1]] = q[:, [0, -1]] @ np.array([[c, -s], [s, c]])
        return q

    left = rotated(left, rate * t)
    if family == "slow_spd":
        right = left
        lipschitz = 2 * rate + 0.5
    else:
        right = rotated(right, -0.7 * rate * t)
        lipschitz = 1.7 * rate + 0.5
    sigma = np.geomspace(1.0, 1.0 / condition, len(left)) * np.exp(-0.5 * t)
    return (left * sigma) @ right.T, float(np.exp(-0.5) / condition), lipschitz


def sufficient_age(target, delta, gap, lipschitz, noise):
    if target <= delta or noise >= gap or lipschitz <= 0:
        return None
    budget = (target - delta) * (gap - noise) / (1 + delta) - noise
    return float(budget / lipschitz) if budget >= 0 else None


def campaign(output, **settings):
    cfg = {**DEFAULTS, **settings}
    if any(not 0 <= age <= 0.5 for age in cfg["ages"]):
        raise ValueError("Refresh ages must be in [0,0.5]")
    if any(not 0 < target < 1 for target in cfg["targets"]):
        raise ValueError("Targets must be between zero and one")
    rows, intervals, rejected, timing = [], [], [], []
    for n in cfg["sizes"]:
        bases = trajectory_bases(n, seed=cfg["seed"])
        rng = np.random.default_rng(cfg["seed"] + n)
        rhs_set = [np.eye(n)[0], np.ones(n) / np.sqrt(n), rng.normal(size=n)]
        for condition in cfg["conditions"]:
            for family in cfg["families"]:
                for update in cfg["update_times"]:
                    old, gap, lipschitz = refresh_matrix(update, family, bases, condition)
                    for ratio in cfg["noise_to_gap_ratios"]:
                        if not 0 <= ratio < 1:
                            raise ValueError("Acquisition error must be smaller than the gap")
                        direction = rng.normal(size=(n, n))
                        direction /= np.linalg.norm(direction, ord=2)
                        acquired = old + ratio * gap * direction
                        noise = float(np.linalg.norm(acquired - old, ord=2))
                        start = time.perf_counter()
                        op = build_compact_inverse(acquired, tolerance=cfg["polynomial_tolerance"])
                        encode_seconds = time.perf_counter() - start
                        start = time.perf_counter()
                        inverse = simulated_operator(op)
                        simulation_seconds = time.perf_counter() - start
                        key = {
                            "size": n,
                            "condition": condition,
                            "family": family,
                            "update_time": update,
                            "noise_to_gap_ratio": ratio,
                        }
                        start = time.perf_counter()
                        np.linalg.svd(acquired, compute_uv=False)
                        isolated_svd_seconds = time.perf_counter() - start
                        native_timing = {
                            "isolated_svd_seconds_not_added_twice": isolated_svd_seconds,
                            "native_synthesis_seconds": None,
                            "native_transpile_seconds": None,
                            "native_gate_simulation_seconds": None,
                        }
                        if (
                            n <= cfg["native_timing_max_size"]
                            and condition <= cfg["native_timing_max_condition"]
                            and ratio == 0
                            and update == 0
                        ):
                            from qiskit_aer import AerSimulator

                            from .hardware import compile_native

                            start = time.perf_counter()
                            circuit = op.circuit(rhs_set[0], fuse_input=False)
                            native_timing["native_synthesis_seconds"] = time.perf_counter() - start
                            start = time.perf_counter()
                            compiled = compile_native(circuit, optimization_level=1)
                            native_timing["native_transpile_seconds"] = time.perf_counter() - start
                            compiled.save_statevector()
                            start = time.perf_counter()
                            AerSimulator(method="statevector", max_parallel_threads=1).run(
                                compiled
                            ).result()
                            native_timing["native_gate_simulation_seconds"] = (
                                time.perf_counter() - start
                            )
                            native_timing["native_timing_status"] = (
                                "measured on this trajectory; simulator wall seconds, "
                                "not physical time"
                            )
                        else:
                            native_timing["native_timing_status"] = (
                                "not run under the explicit native timing budget"
                            )
                        timing.append(
                            {
                                **key,
                                **native_timing,
                                "encoding_kind": op.encoding.encoding_kind,
                                "encoding_and_polynomial_seconds": encode_seconds,
                                "structured_operator_simulation_seconds": simulation_seconds,
                                "physical_acquisition_seconds": None,
                                "physical_acquisition_status": "not measured; generated input",
                            }
                        )
                        case_rows = []
                        for age in cfg["ages"]:
                            current, _, _ = refresh_matrix(update + age, family, bases, condition)
                            drift = float(np.linalg.norm(current - old, ord=2))
                            left = float(np.linalg.norm(current @ inverse - np.eye(n), ord=2))
                            right = float(np.linalg.norm(inverse @ current - np.eye(n), ord=2))
                            bound = tracking_bound(op.residual_bound, gap, drift, noise)
                            samples = []
                            for index, b in enumerate(rhs_set):
                                x = inverse @ b
                                reference = solve(current, b, assume_a="gen")
                                p = float(
                                    np.linalg.norm(x) ** 2
                                    / (op.normalization * np.linalg.norm(b)) ** 2
                                )
                                samples.append(
                                    {
                                        "rhs_index": index,
                                        "relative_solution_error": float(
                                            np.linalg.norm(x - reference)
                                            / np.linalg.norm(reference)
                                        ),
                                        "relative_residual": float(
                                            np.linalg.norm(current @ x - b) / np.linalg.norm(b)
                                        ),
                                        "normalized_state_error": float(
                                            np.linalg.norm(
                                                x / np.linalg.norm(x)
                                                - reference / np.linalg.norm(reference)
                                            )
                                        ),
                                        "accepted_output_probability": p,
                                        "expected_attempts_per_accepted_output": 1 / p,
                                    }
                                )
                            residual = max(left, right)
                            row = {
                                **key,
                                "evaluation_time": update + age,
                                "refresh_age": age,
                                "gap_lower_bound": gap,
                                "lipschitz_upper_bound": lipschitz,
                                "acquisition_error_norm": noise,
                                "drift_norm": drift,
                                "left_residual": left,
                                "right_residual": right,
                                "tracking_bound_eq24": bound,
                                "samples": samples,
                                "targets_met": {str(t): residual <= t for t in cfg["targets"]},
                                "bound_passed": residual <= bound + 2e-9
                                and drift <= lipschitz * age + 1e-12,
                            }
                            row["bound_passed"] &= all(
                                s["relative_solution_error"] <= bound + 2e-9
                                and (bound >= 1 or s["normalized_state_error"] <= 2 * bound + 2e-9)
                                for s in samples
                            )
                            rows.append(row)
                            case_rows.append(row)
                        for target in cfg["targets"]:
                            sufficient = sufficient_age(
                                target, op.residual_bound, gap, lipschitz, noise
                            )
                            feasible = [
                                r["refresh_age"]
                                for r in case_rows
                                if max(r["left_residual"], r["right_residual"]) <= target
                            ]
                            guaranteed = [
                                r
                                for r in case_rows
                                if sufficient is not None and r["refresh_age"] <= sufficient
                            ]
                            intervals.append(
                                {
                                    **key,
                                    "target": target,
                                    "sufficient_refresh_age_eq25": sufficient,
                                    "empirical_maximum_sampled_feasible_age": max(feasible)
                                    if feasible
                                    else None,
                                    "sufficient_condition_validated": all(
                                        r["targets_met"][str(target)] for r in guaranteed
                                    ),
                                    "samples_inside_sufficient_interval": len(guaranteed),
                                    "maximum_is_grid_limited": bool(
                                        feasible and max(feasible) == max(cfg["ages"])
                                    ),
                                }
                            )
        for condition in (1e8, 1e16):
            matrix, _, _ = refresh_matrix(0, "fast_nonsymmetric", bases, condition)
            try:
                build_compact_inverse(matrix, tolerance=1e-12)
                rejected.append({"size": n, "condition": condition, "rejected": False})
            except (ValueError, MemoryError) as error:
                rejected.append(
                    {"size": n, "condition": condition, "rejected": True, "reason": str(error)}
                )
    positive = sum(r["targets_met"][str(cfg["targets"][0])] for r in rows)
    failures = len(rows) - positive
    payload = {
        "passed": all(r["bound_passed"] for r in rows)
        and all(r["sufficient_condition_validated"] for r in intervals)
        and all(r["rejected"] for r in rejected)
        and positive > 0
        and failures > 0,
        "configuration": cfg,
        "samples": rows,
        "refresh_intervals": intervals,
        "positive_accuracy_cases": positive,
        "intentional_accuracy_failures": failures,
        "near_rank_loss_rejections": rejected,
        "timings": timing,
        "model_time_unit": "dimensionless; not physical seconds",
        "native_synthesis_timing": "reported separately in baselines/result.json",
        "physical_hardware_tested": False,
        "real_time_deadline_tested": False,
    }
    return save_result(output, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/temporal-refresh")
    parser.add_argument("--config")
    for name, typ in [
        ("sizes", int),
        ("conditions", float),
        ("ages", float),
        ("targets", float),
        ("update-times", float),
        ("noise-to-gap-ratios", float),
    ]:
        parser.add_argument("--" + name, type=typ, nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--native-timing-max-size", type=int)
    args = parser.parse_args()
    cfg = configuration(args.config, "temporal-refresh", DEFAULTS)
    cfg.update({k: v for k, v in vars(args).items() if k in DEFAULTS and v is not None})
    result = campaign(args.output_dir, **cfg)
    print(
        "Refresh study: "
        f"{len(result['samples'])}"
        " samples; positive="
        f"{result['positive_accuracy_cases']}"
        "; intentional failures="
        f"{result['intentional_accuracy_failures']}"
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
