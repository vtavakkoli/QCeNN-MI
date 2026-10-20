"""Dissipative baseline, variable-spectrum tracking, and condition-100 diagnosis.

All large experiments simulate reduced density matrices. No gate-level or
physical noise evidence is inferred from these logical simulations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .dissipative_comparison import (
    change_channel_basis,
    continuous_trajectory,
    shang_channel,
    stationary_depolarized,
)
from .experiment_io import configuration, save_npz_atomic, save_result
from .precision_benchmark import controlled_fixture
from .recurrent_density import DensityObserver, build_density_channel
from .validation import provenance

DEFAULTS = {
    "sizes": [10, 100, 1000],
    "conditions": [2, 10, 100],
    "steps": {"2": 512, "10": 4096, "100": 4096},
    "record_points": 33,
    "residual_target": 1e-6,
    "conditional_infidelity_target": 1e-6,
    "ode_rtol": 2e-10,
    "ode_atol": 2e-12,
    "noise_probabilities": [1e-6, 1e-5, 1e-4, 1e-3],
    "dynamic_conditions": [2, 4, 10, 30, 100, 100, 30, 10, 4, 2],
    "dynamic_dwells": [32, 128],
    "dynamic_hold": 1024,
    "dynamic_probabilities": [0, 1e-4, 1e-3],
    "dynamic_stride": 16,
    "seed": 1729,
    "extended_steps": 1048576,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sampled_times(budget, points):
    return np.unique(
        np.rint(np.r_[np.linspace(0, budget, points), np.geomspace(1, budget, points)]).astype(int)
    )


def convergence(rows, settings):
    """First recorded time after which every recorded point meets both targets."""
    good = [
        row["relative_residual"] <= settings["residual_target"]
        and row["conditional_solution_fidelity"] is not None
        and 1 - row["conditional_solution_fidelity"] <= settings["conditional_infidelity_target"]
        for row in rows
    ]
    first = None
    for row, passed in zip(reversed(rows), reversed(good), strict=True):
        if not passed:
            break
        first = row["time"]
    return {"accuracy_passed": bool(good[-1]), "first_sustained_sampled_time": first}


def integrity(density, rows):
    minimum = float(np.linalg.eigvalsh(density)[0])
    drift = max(abs(row["trace"] - 1) for row in rows)
    return {
        "minimum_final_density_eigenvalue": minimum,
        "maximum_trace_drift": drift,
        "numerical_integrity_passed": bool(minimum > -1e-8 and drift < 1e-8),
    }


def matched_fixture(output, size, condition, settings):
    matrix = controlled_fixture(size, "controlled_nonsymmetric", condition, seed=settings["seed"])
    rhs = np.random.default_rng(settings["seed"] + size).normal(size=size)
    reference = np.linalg.solve(matrix, rhs)  # Read-only independent diagnostic.
    fixture = output / "inputs" / f"n{size}-k{condition}.npz"
    save_npz_atomic(fixture, matrix=matrix, rhs=rhs, reference_solution=reference)
    channel = build_density_channel(matrix, rhs)
    times = sampled_times(settings["steps"][str(condition)], settings["record_points"])
    observer = DensityObserver.create(channel, reference)
    density = channel.initial_density()
    rows, records = [], []
    start = time.perf_counter()
    wanted = set(times)
    work = np.empty_like(density)
    for step in range(int(times[-1]) + 1):
        if step in wanted:
            rows.append({"time": step, **observer.read(density)})
        if step < times[-1]:
            channel.step(density, work=work)
    records.append(
        {
            "method": "discrete",
            "trajectory": rows,
            "final": rows[-1],
            "wall_seconds": time.perf_counter() - start,
            **convergence(rows, settings),
            **integrity(density, rows),
        }
    )
    short_density = density.copy()
    for name, compiled in [("matched_continuous", channel), ("shang_rate_normalized", None)]:
        start = time.perf_counter()
        if compiled is None:
            compiled = shang_channel(matrix, rhs)
        readout = DensityObserver.create(compiled, reference)
        densities, evaluations = continuous_trajectory(
            compiled, times, rtol=settings["ode_rtol"], atol=settings["ode_atol"]
        )
        rows = [
            {"time": int(t), **readout.read(rho)} for t, rho in zip(times, densities, strict=True)
        ]
        records.append(
            {
                "method": name,
                "trajectory": rows,
                "final": rows[-1],
                "ode_evaluations": evaluations,
                "wall_seconds": time.perf_counter() - start,
                **convergence(rows, settings),
                **integrity(densities[-1], rows),
            }
        )
        if size <= 100:
            refined, _ = continuous_trajectory(
                compiled,
                [0, times[-1]],
                rtol=settings["ode_rtol"] / 10,
                atol=settings["ode_atol"] / 10,
            )
            records[-1]["tolerance_refinement_maximum_density_error"] = float(
                np.max(abs(refined[-1] - densities[-1]))
            )
            records[-1]["tolerance_refinement_residual_difference"] = float(
                abs(readout.read(refined[-1])["relative_residual"] - rows[-1]["relative_residual"])
            )
        del densities
    for row in records:
        row.update(
            size=size,
            condition=condition,
            input_file=str(fixture.relative_to(output)),
            input_sha256=sha(fixture),
            logical_depolarizing_probability=0.0,
        )
    noise = []
    if condition == 100:
        for probability in settings["noise_probabilities"]:
            density = channel.initial_density()
            for _ in range(int(times[-1])):
                channel.step(density, depolarizing_probability=probability, work=work)
            stationary = stationary_depolarized(channel, probability)
            fixed_check = stationary.copy()
            channel.step(fixed_check, depolarizing_probability=probability, work=work)
            noisy = observer.read(density)
            fixed = observer.read(stationary)
            noise.append(
                {
                    "size": size,
                    "condition": condition,
                    "steps": int(times[-1]),
                    "logical_depolarizing_probability": probability,
                    "transient": noisy,
                    "stationary": fixed,
                    "stationary_equation_maximum_error": float(
                        np.max(abs(fixed_check - stationary))
                    ),
                    "stationary_trace": float(np.trace(stationary)),
                    "stationary_minimum_eigenvalue": float(np.linalg.eigvalsh(stationary)[0]),
                    "noisy_target": 0.01,
                    "transient_accuracy_passed": noisy["relative_residual"] <= 0.01,
                    "stationary_accuracy_passed": fixed["relative_residual"] <= 0.01,
                    "ideal_same_budget_residual": records[0]["final"]["relative_residual"],
                    "transient_distance_to_stationary": float(np.linalg.norm(density - stationary)),
                    "input_file": str(fixture.relative_to(output)),
                    "input_sha256": sha(fixture),
                }
            )
    extended = None
    if condition == 100 and size <= 10:
        # Independently exponentiate the *discrete* superoperator. This is a
        # small simulator shortcut, not a claim to power a quantum channel for free.
        d = channel.dimension
        superoperator = np.diag(np.outer(channel.no_jump_values, channel.no_jump_values).ravel())
        trace_g = np.zeros((d, d))
        np.fill_diagonal(trace_g, channel.residual_eigenvalues)
        superoperator += np.outer(
            np.outer(channel.reference, channel.reference).ravel(), trace_g.ravel()
        )
        short = (
            np.linalg.matrix_power(superoperator, int(times[-1]))
            @ channel.initial_density().ravel()
        ).reshape(d, d)
        crosscheck = float(np.max(abs(short - short_density)))
        density = (
            np.linalg.matrix_power(superoperator, settings["extended_steps"])
            @ channel.initial_density().ravel()
        ).reshape(d, d)
        extended = {
            "size": size,
            "condition": condition,
            "steps": settings["extended_steps"],
            "superoperator_vs_loop_maximum_error": crosscheck,
            "final": observer.read(density),
            **integrity(density, [observer.read(density)]),
        }
    return records, noise, extended


def spectrum_schedule(size, conditions):
    """Vary every singular value, norm and condition; final matrix returns to start."""
    grids = []
    for index, condition in enumerate(conditions):
        phase = index / (len(conditions) - 1)
        scale = 1 + 0.25 * np.sin(2 * np.pi * phase)
        exponent = 1 + 0.3 * np.sin(2 * np.pi * phase)
        grids.append(scale * condition ** (-(np.linspace(0, 1, size) ** exponent)))
    return np.asarray(grids)


def dynamic_fixture(output, size, settings):
    rng = np.random.default_rng(settings["seed"] + size + 7000)
    left, _ = np.linalg.qr(rng.normal(size=(size, size)))
    right, _ = np.linalg.qr(rng.normal(size=(size, size)))
    rhs = rng.normal(size=size)
    spectra = spectrum_schedule(size, settings["dynamic_conditions"])
    channels, observers, snapshots = [], [], []
    start_compile = time.perf_counter()
    for index, singular in enumerate(spectra):
        matrix = (left * singular) @ right.T
        reference = right @ ((left.T @ rhs) / singular)  # Diagnostics only.
        channel = build_density_channel(matrix, rhs)
        channels.append(channel)
        observers.append(DensityObserver.create(channel, reference))
        path = output / "inputs" / f"n{size}-spectrum-{index:02d}.npz"
        save_npz_atomic(
            path, matrix=matrix, rhs=rhs, singular_values=singular, reference_solution=reference
        )
        snapshots.append(
            {
                "index": index,
                "condition": float(singular[0] / singular[-1]),
                "sigma_max": float(singular[0]),
                "sigma_min": float(singular[-1]),
                "file": str(path.relative_to(output)),
                "sha256": sha(path),
            }
        )
    compile_seconds = time.perf_counter() - start_compile
    cases = []
    for dwell in settings["dynamic_dwells"]:
        budget = (len(channels) - 1) * dwell + settings["dynamic_hold"]
        for probability in settings["dynamic_probabilities"]:
            density = channels[0].initial_density()
            work = np.empty_like(density)
            rows, previous = [], 0
            purified_bound = np.sqrt(max(0, 1 - observers[0].read(density)["augmented_fidelity"]))
            previous_bound = purified_bound
            start = time.perf_counter()
            worst_bound_excess = 0.0
            for step in range(budget + 1):
                index = min(step // dwell, len(channels) - 1)
                if index != previous:
                    density = change_channel_basis(channels[previous], channels[index], density)
                    # Targets are used only to verify a sufficient certificate,
                    # and are never inputs to density evolution.
                    overlap = np.vdot(
                        channels[previous].basis @ observers[previous].target_coordinates,
                        channels[index].basis @ observers[index].target_coordinates,
                    )
                    target_drift = np.sqrt(max(0, 1 - abs(overlap) ** 2))
                    purified_bound = min(1, purified_bound + target_drift)
                channel = channels[index]
                if step:
                    channel.step(density, depolarizing_probability=probability, work=work)
                    theta = channel.singular_ratio**2 / (2 * (1 + channel.singular_ratio**2))
                    purified_bound = np.sqrt(
                        (1 - probability) * (1 - theta) * purified_bound**2
                        + probability * (1 - 1 / channel.dimension)
                    )
                if step % settings["dynamic_stride"] == 0 or step == budget or index != previous:
                    observed = observers[index].read(density)
                    worst_bound_excess = max(
                        worst_bound_excess, 1 - observed["augmented_fidelity"] - purified_bound**2
                    )
                    rows.append(
                        {
                            "time": step,
                            "snapshot": index,
                            "condition": snapshots[index]["condition"],
                            "sigma_min": snapshots[index]["sigma_min"],
                            "sigma_max": snapshots[index]["sigma_max"],
                            "purified_distance_upper_bound": float(purified_bound),
                            **observed,
                        }
                    )
                previous = index
                previous_bound = purified_bound
            cases.append(
                {
                    "size": size,
                    "dwell": dwell,
                    "steps": budget,
                    "logical_depolarizing_probability": probability,
                    "data_reinitialized": False,
                    "wall_seconds": time.perf_counter() - start,
                    "dense_channel_compilation_seconds_shared": compile_seconds,
                    "trajectory": rows,
                    "final": rows[-1],
                    "maximum_tracking_bound_excess": float(worst_bound_excess),
                    "final_purified_distance_bound": float(previous_bound),
                    **integrity(density, rows),
                }
            )
    return cases, {"size": size, "snapshots": snapshots}


def campaign(output, settings):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "inputs").mkdir(exist_ok=True)
    records, noise, extended, dynamics, inputs = [], [], [], [], []
    for size in settings["sizes"]:
        for condition in settings["conditions"]:
            print(f"Matched dissipative comparison N={size}, kappa={condition}", flush=True)
            found, noisy, long_run = matched_fixture(output, size, condition, settings)
            records.extend(found)
            noise.extend(noisy)
            if long_run:
                extended.append(long_run)
        print(f"Variable-spectrum dynamics N={size}", flush=True)
        found, fixture = dynamic_fixture(output, size, settings)
        dynamics.extend(found)
        inputs.append(fixture)
    passed = (
        all(row["numerical_integrity_passed"] for row in records + dynamics + extended)
        and all(row["maximum_tracking_bound_excess"] < 1e-8 for row in dynamics)
        and all(
            row["stationary_equation_maximum_error"] < 1e-9
            and abs(row["stationary_trace"] - 1) < 1e-8
            and row["stationary_minimum_eigenvalue"] > -1e-10
            for row in noise
        )
        and all(row["superoperator_vs_loop_maximum_error"] < 1e-9 for row in extended)
        and all(row.get("tolerance_refinement_maximum_density_error", 0) < 5e-8 for row in records)
    )
    result = save_result(
        output,
        {
            "schema_version": 1,
            "configuration": settings,
            "provenance": provenance(Path.cwd()),
            "comparison_contract": {
                "source": "Z.-X. Shang, arXiv:2609.12284v1, Eqs. (10), (20)",
                "normalization": "Shang L/2; all residual G norms <=1; model time, not gate count",
                "initial_state": "same physical reference state |r><r|",
                "inputs": "identical matrix and rhs per matched fixture",
                "criteria": "relative coherence residual and conditional infidelity each <=1e-6",
                "censoring": "first sustained recorded sample; null means budget exhausted",
                "comparison_noise": "ideal only; separate condition-100 logical-noise diagnosis",
                "no_hardware_speedup_claim": True,
            },
            "cases": records,
            "condition_100_noise": noise,
            "extended_ideal": extended,
            "dynamic_cases": dynamics,
            "dynamic_inputs": inputs,
            "passed": bool(passed),
            "physical_hardware_tested": False,
        },
    )
    from .dissipative_report import build_report

    build_report(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    settings = configuration(args.config, "dissipative-comparison", DEFAULTS)
    result = campaign(args.output_dir, settings)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "matched_cases": len(result["cases"]),
                "dynamic_cases": len(result["dynamic_cases"]),
            }
        ),
        flush=True,
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
