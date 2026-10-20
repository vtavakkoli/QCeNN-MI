"""Matched polynomial recurrence experiments; no hardware performance inferred."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np

from .dissipative_comparison import change_channel_basis
from .polynomial_recurrent import (
    accelerate_density_channel,
    accelerated_gap,
    build_polynomial_solver,
)
from .polynomial_renewal import renewal_observables
from .precision_benchmark import controlled_fixture
from .recurrent_density import DensityObserver, build_density_channel
from .validation import provenance

DEFAULTS = {
    "sizes": [2, 4, 10, 100, 1000],
    "conditions": [2, 10, 100, 1000],
    "families": ["controlled_spd", "controlled_nonsymmetric", "controlled_indefinite", "clustered"],
    "degrees": [0, 1, 2, 4, 8, 16],
    "seed": 1729,
    "budgets": {"2": 512, "10": 4096, "100": 16384, "1000": 16384},
    "record_points": 33,
    "residual_target": 1e-6,
    "infidelity_target": 1e-6,
    "dynamic_sizes": [10, 100, 1000],
    "dynamic_degrees": [0, 4, 16],
    "dynamic_conditions": [2, 10, 100, 1000, 1000, 100, 10, 2],
    "dynamic_probabilities": [0, 1e-4, 1e-3],
    "dynamic_dwell": 32,
    "dynamic_hold": 512,
    "dynamic_stride": 16,
}


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_input(path, **arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    content = buffer.getvalue()
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)
    return hashlib.sha256(content).hexdigest()


def fixture(size, condition, family, seed):
    if family != "clustered":
        return controlled_fixture(size, family, condition, seed=seed)
    rng = np.random.default_rng(seed + size + 917)
    left, _ = np.linalg.qr(rng.normal(size=(size, size)))
    right, _ = np.linalg.qr(rng.normal(size=(size, size)))
    singular = np.r_[1.0, np.geomspace(1 / condition, min(1, 1.01 / condition), size - 1)]
    return (left * singular) @ right.T


def sustained(rows, settings):
    found = None
    for point in reversed(rows):
        fidelity = point["conditional_solution_fidelity"]
        residual = point["relative_residual"]
        if (
            residual is None
            or residual > settings["residual_target"]
            or fidelity is None
            or 1 - fidelity > settings["infidelity_target"]
        ):
            break
        found = point["time"]
    return found


def static_study(output, settings):
    records = []
    for size in settings["sizes"]:
        for condition in settings["conditions"]:
            for family in settings["families"]:
                print(f"Static N={size} kappa={condition} family={family}", flush=True)
                start = time.perf_counter()
                matrix = fixture(size, condition, family, settings["seed"])
                rhs = np.random.default_rng(settings["seed"] + size).normal(size=size)
                reference = np.linalg.solve(matrix, rhs)
                base = build_density_channel(matrix, rhs)
                preprocessing = time.perf_counter() - start
                file = f"inputs/static-n{size}-k{condition}-{family}.npz"
                checksum = save_input(
                    output / file, matrix=matrix, rhs=rhs, reference_solution=reference
                )
                budget = settings["budgets"][str(condition)]
                times = np.unique(
                    np.r_[
                        0,
                        np.arange(1, 9),
                        np.rint(np.geomspace(9, budget, settings["record_points"])),
                    ]
                ).astype(int)
                varied_initials = size <= 10 or (
                    family == "controlled_nonsymmetric" and condition in [2, 100]
                )
                initials = (
                    ["reference", "basis", "random", "mixed"] if varied_initials else ["reference"]
                )
                for degree in settings["degrees"]:
                    start = time.perf_counter()
                    channel = accelerate_density_channel(base, degree)
                    transform = time.perf_counter() - start
                    for initial in initials:
                        start = time.perf_counter()
                        rows = renewal_observables(channel, budget, times, initial, reference)
                        measured = time.perf_counter() - start
                        first = sustained(rows, settings)
                        trace_error = max(abs(x["trace"] - 1) for x in rows)
                        theta = accelerated_gap(channel.singular_ratio, degree) / 2
                        initial_error = 1 - rows[0]["augmented_fidelity"]
                        certificate_excess = max(
                            1 - x["augmented_fidelity"] - (1 - theta) ** x["time"] * initial_error
                            for x in rows
                        )
                        record = {
                            "size": size,
                            "condition": condition,
                            "family": family,
                            "polynomial_degree": degree,
                            "initial_state": initial,
                            "input_file": file,
                            "input_sha256": checksum,
                            "steps": budget,
                            "first_sustained_sampled_update": first,
                            "accuracy_passed": first is not None,
                            "polynomial_order_times_first_update": None
                            if first is None
                            else (degree + 1) * first,
                            "shared_preprocessing_seconds": preprocessing,
                            "polynomial_response_seconds": transform,
                            "simulation_seconds": measured,
                            "trajectory": rows,
                            "final": rows[-1],
                            "maximum_trace_error": trace_error,
                            "maximum_static_certificate_excess": float(certificate_excess),
                            "simulation_method": "exact renewal observables with FFT convolution",
                            "native_gate_simulation": False,
                        }
                        records.append(record)
                        save_json(
                            output / "static-progress.json",
                            {
                                "records_completed": len(records),
                                "last_case": {
                                    k: record[k]
                                    for k in [
                                        "size",
                                        "condition",
                                        "family",
                                        "polynomial_degree",
                                        "initial_state",
                                    ]
                                },
                            },
                        )
    return records


def dynamic_study(output, settings):
    results, inputs = [], []
    for size in settings["dynamic_sizes"]:
        print(f"Dynamic N={size}", flush=True)
        rng = np.random.default_rng(settings["seed"] + size + 8000)
        u, _ = np.linalg.qr(rng.normal(size=(size, size)))
        v, _ = np.linalg.qr(rng.normal(size=(size, size)))
        b0, b1 = rng.normal(size=(2, size))
        channels, observers, snapshots = [], [], []
        conditions = settings["dynamic_conditions"]
        for j, condition in enumerate(conditions):
            phase = j / (len(conditions) - 1)
            singular = (1 + 0.2 * np.sin(2 * np.pi * phase)) * condition ** (
                -(np.linspace(0, 1, size) ** (1 + 0.2 * np.sin(2 * np.pi * phase)))
            )
            angle = 0.4 * np.sin(2 * np.pi * phase) + (0.5 if j >= 4 else 0)
            rotation = np.eye(size)
            rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            matrix = ((u * singular) @ v.T) @ rotation
            rhs = np.cos(phase) * b0 + np.sin(phase) * b1
            if j >= 4:
                rhs += 0.25 * b1
            reference = np.linalg.solve(matrix, rhs)
            base = build_density_channel(matrix, rhs)
            channels.append(base)
            observers.append(DensityObserver.create(base, reference))
            file = f"inputs/dynamic-n{size}-{j}.npz"
            checksum = save_input(
                output / file,
                matrix=matrix,
                rhs=rhs,
                reference_solution=reference,
                singular_values=singular,
            )
            snapshots.append({"index": j, "condition": condition, "file": file, "sha256": checksum})
        inputs.append({"size": size, "snapshots": snapshots})
        budget = (len(channels) - 1) * settings["dynamic_dwell"] + settings["dynamic_hold"]
        for degree in settings["dynamic_degrees"]:
            compiled = [accelerate_density_channel(x, degree) for x in channels]
            for probability in settings["dynamic_probabilities"]:
                start = time.perf_counter()
                density = compiled[0].initial_density()
                work = np.empty_like(density)
                old, rows = 0, []
                error_bound = np.sqrt(max(0, 1 - observers[0].read(density)["augmented_fidelity"]))
                maximum_bound_excess = 0
                for k in range(budget + 1):
                    index = min(k // settings["dynamic_dwell"], len(compiled) - 1)
                    if index != old:
                        density = change_channel_basis(compiled[old], compiled[index], density)
                        before = channels[old].basis @ observers[old].target_coordinates
                        after = channels[index].basis @ observers[index].target_coordinates
                        drift = np.sqrt(max(0, 1 - float(np.dot(before, after)) ** 2))
                        error_bound = min(1, error_bound + drift)
                    if k:
                        compiled[index].step(
                            density, depolarizing_probability=probability, work=work
                        )
                        theta = accelerated_gap(compiled[index].singular_ratio, degree) / 2
                        error_bound = np.sqrt(
                            (1 - probability) * (1 - theta) * error_bound**2
                            + probability * (1 - 1 / compiled[index].dimension)
                        )
                    if k % settings["dynamic_stride"] == 0 or k == budget or index != old:
                        observed = observers[index].read(density)
                        excess = 1 - observed["augmented_fidelity"] - error_bound**2
                        maximum_bound_excess = max(maximum_bound_excess, excess)
                        rows.append(
                            {
                                "time": k,
                                "snapshot": index,
                                "condition": conditions[index],
                                "purified_distance_bound": float(error_bound),
                                "augmented_purified_distance": float(
                                    np.sqrt(max(0, 1 - observed["augmented_fidelity"]))
                                ),
                                **observed,
                            }
                        )
                    old = index
                final_hold = [x for x in rows if x["snapshot"] == len(channels) - 1]
                first = sustained(final_hold, settings)
                results.append(
                    {
                        "size": size,
                        "polynomial_degree": degree,
                        "logical_depolarizing_probability": probability,
                        "data_reinitialized": False,
                        "steps": budget,
                        "trajectory": rows,
                        "final": rows[-1],
                        "simulation_seconds": time.perf_counter() - start,
                        "maximum_tracking_bound_excess": float(maximum_bound_excess),
                        "final_hold_first_sustained_sampled_time": first,
                        "peak_relative_residual": max(x["relative_residual"] for x in rows),
                        "peak_absolute_x1_error": max(x["absolute_x1_error"] for x in rows),
                        "physical_hardware_tested": False,
                    }
                )
                save_json(output / "dynamic-progress.json", {"cases": results, "inputs": inputs})
    return results, inputs


def native_study(output, settings, static_cases=()):
    from qiskit import qpy, transpile
    from qiskit.quantum_info import Statevector, partial_trace

    resources, comparisons = [], []
    for size in settings["sizes"]:
        for condition in settings["conditions"]:
            matrix = fixture(size, condition, "controlled_nonsymmetric", settings["seed"])
            rhs = np.random.default_rng(settings["seed"] + size).normal(size=size)
            for degree in settings["degrees"]:
                row = {"size": size, "condition": condition, "polynomial_degree": degree}
                start = time.perf_counter()
                try:
                    solver = build_polynomial_solver(matrix, rhs, degree, max_encoding_dimension=16)
                    preprocessing = time.perf_counter() - start
                    dimension = len(solver.no_jump)
                    reset = np.eye(dimension)[size]
                    kraus = [solver.no_jump] + [
                        np.outer(reset, r) for r in solver.residual_operator
                    ]
                    choi = sum(np.outer(k.ravel(order="F"), k.ravel(order="F")) for k in kraus)
                    completeness = sum(k.T @ k for k in kraus)
                    target = np.zeros(dimension)
                    target[:size] = (
                        solver.singular_ratio
                        * solver.matrix_scale
                        * np.linalg.solve(matrix, rhs)
                        / np.linalg.norm(rhs)
                    )
                    target[size] = 1
                    target /= np.linalg.norm(target)
                    target_density = np.outer(target, target)
                    stationary = sum(k @ target_density @ k.T for k in kraus)
                    row.update(
                        choi_minimum_eigenvalue=float(np.linalg.eigvalsh(choi)[0]),
                        kraus_completeness_maximum_error=float(
                            np.max(abs(completeness - np.eye(dimension)))
                        ),
                        kernel_residual=float(np.linalg.norm(solver.residual_operator @ target)),
                        stationary_maximum_error=float(np.max(abs(stationary - target_density))),
                    )
                    circuit = solver.circuit(1, workspace="reset")
                    start = time.perf_counter()
                    native = transpile(
                        circuit,
                        basis_gates=["rz", "sx", "x", "cx"],
                        optimization_level=1,
                        seed_transpiler=42,
                    )
                    row.update(
                        status="compiled",
                        width=native.num_qubits,
                        cx=native.count_ops().get("cx", 0),
                        depth=native.depth(),
                        resets_per_additional_update=solver.data_qubits + 1,
                        preprocessing_seconds=preprocessing,
                        transpile_seconds=time.perf_counter() - start,
                        topology="all-to-all native basis; no device calibration",
                    )
                    path = output / "circuits" / f"n{size}-k{condition}-m{degree}.qpy"
                    buffer = io.BytesIO()
                    qpy.dump(native, buffer)
                    path.write_bytes(buffer.getvalue())
                    row["qpy_file"] = str(path.relative_to(output))
                    matches = [
                        r
                        for r in static_cases
                        if r["size"] == size
                        and r["condition"] == condition
                        and r["family"] == "controlled_nonsymmetric"
                        and r["polynomial_degree"] == degree
                        and r["initial_state"] == "reference"
                    ]
                    first = matches[0]["first_sustained_sampled_update"] if matches else None
                    if first is not None and first <= 128 and first * row["cx"] <= 50000:
                        start = time.perf_counter()
                        complete = transpile(
                            solver.circuit(first, workspace="reset"),
                            basis_gates=["rz", "sx", "x", "cx"],
                            optimization_level=1,
                            seed_transpiler=42,
                        )
                        row.update(
                            converged_cost_status="compiled",
                            convergence_updates=first,
                            convergence_cx=complete.count_ops().get("cx", 0),
                            convergence_depth=complete.depth(),
                            convergence_resets=complete.count_ops().get("reset", 0),
                            complete_compile_seconds=time.perf_counter() - start,
                        )
                        buffer = io.BytesIO()
                        qpy.dump(complete, buffer)
                        (
                            output / "circuits" / f"n{size}-k{condition}-m{degree}-converged.qpy"
                        ).write_bytes(buffer.getvalue())
                    else:
                        row.update(
                            converged_cost_status="not_converged"
                            if first is None
                            else "resource_rejected",
                            convergence_updates=first,
                            convergence_cost_limit="128 updates / 50000 precheck CX",
                        )
                    if condition in [2, 100]:
                        for initial in ["reference", "random"]:
                            physical = np.zeros(1 << solver.data_qubits)
                            if initial == "reference":
                                physical[size] = 1
                            else:
                                physical[: size + 1] = np.random.default_rng(831 + size).normal(
                                    size=size + 1
                                )
                                physical /= np.linalg.norm(physical)
                            full = solver.circuit(2, workspace="coherent", initial_state=physical)
                            density = partial_trace(
                                Statevector(full), list(range(solver.data_qubits, full.num_qubits))
                            ).data
                            spectral = accelerate_density_channel(
                                build_density_channel(matrix, rhs), degree
                            )
                            state = spectral.basis.T @ physical
                            reduced = np.outer(state, state)
                            for _ in range(2):
                                spectral.step(reduced)
                            expected = spectral.physical_density(reduced)
                            comparisons.append(
                                {
                                    **row,
                                    "initial_state": initial,
                                    "steps": 2,
                                    "full_coherent_width": full.num_qubits,
                                    "maximum_density_error": float(np.max(abs(density - expected))),
                                    "full_state_trace": float(np.trace(density).real),
                                }
                            )
                except MemoryError as error:
                    row.update(
                        status="resource_rejected", reason=str(error), native_gate_simulation=False
                    )
                resources.append(row)
    return resources, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument(
        "--part", choices=["static", "dynamic", "native", "noise", "all"], default="all"
    )
    args = parser.parse_args()
    settings = DEFAULTS.copy()
    if args.config:
        config = json.loads(Path(args.config).read_text())
        settings.update(config.get("polynomial-recurrent", config))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "inputs").mkdir(exist_ok=True)
    (output / "circuits").mkdir(exist_ok=True)
    start = time.perf_counter()
    data = {
        "configuration": settings,
        "provenance": provenance(Path(__file__).resolve().parents[2]),
        "physical_hardware_tested": False,
    }
    if args.part in ["static", "all"]:
        data["static_cases"] = static_study(output, settings)
    if args.part in ["dynamic", "all"]:
        data["dynamic_cases"], data["dynamic_inputs"] = dynamic_study(output, settings)
    if args.part in ["native", "all"]:
        data["native_resources"], data["circuit_comparisons"] = native_study(
            output, settings, data.get("static_cases", [])
        )
    if args.part in ["noise", "all"]:
        from .polynomial_noise import gate_noise_study, noise_study

        data["logical_noise_cases"] = noise_study(output, settings)
        data["gate_noise_cases"] = gate_noise_study(output, settings)
    if args.part == "all":
        from .polynomial_prior import prior_comparison

        data["prior_comparison"] = prior_comparison(output, settings)
    data["wall_seconds"] = time.perf_counter() - start
    data["passed"] = (
        all(x["maximum_trace_error"] < 1e-8 for x in data.get("static_cases", []))
        and all(x["maximum_density_error"] < 1e-10 for x in data.get("circuit_comparisons", []))
        and all(x["maximum_tracking_bound_excess"] < 1e-8 for x in data.get("dynamic_cases", []))
    )
    save_json(output / ("result.json" if args.part == "all" else f"{args.part}-result.json"), data)
    if args.part == "all":
        from .polynomial_report import build_report

        build_report(output)
    files = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "MANIFEST.json"
    }
    save_json(output / "MANIFEST.json", {"files": files})
    if args.part == "all":
        from .polynomial_report import verify

        verify(output)
    if not data["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
