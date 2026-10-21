"""Matched three-method study; circuit, ideal query and assumed time costs differ."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Operator, Statevector, partial_trace
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error

from .chebyshev_recurrent import (
    chebyshev_response,
    chebyshev_schedule_circuit,
    filter_degree,
    filter_density_channel,
    qsp_filter_circuit,
    suppression_bound,
    synthesize_filter,
)
from .dissipative_comparison import stationary_depolarized
from .experiment_io import configuration, save_qpy_atomic
from .polynomial_benchmark import fixture, save_input, save_json, sustained
from .polynomial_recurrent import accelerate_density_channel, build_polynomial_solver
from .polynomial_renewal import renewal_observables
from .recurrent_cost import (
    amortized_cost,
    certified_readout_budget,
    coherence_readout_rotations,
    scheduled_latency,
)
from .recurrent_density import DensityObserver, build_density_channel
from .recurrent_quantum import build_recurrent_solver, density_diagnostics
from .validation import provenance

DEFAULTS = {
    "sizes": [2, 4, 10, 100, 1000],
    "conditions": [2, 10, 100, 1000],
    "families": ["controlled_spd", "controlled_nonsymmetric", "controlled_indefinite", "clustered"],
    "power_degrees": [0, 4, 16],
    "small_initial_states": ["reference", "basis", "random", "mixed"],
    "seed": 1729,
    "maximum_phase_degree": 64,
    "residual_target": 1e-6,
    "infidelity_target": 1e-6,
    "native_sizes": [2, 4, 10],
    "native_conditions": [2, 10, 100, 1000],
    "maximum_native_convergence_updates": 4096,
    "maximum_native_cx": 200000,
    "maximum_reset_density_qubits": 9,
    "cost_conditions": [2, 10],
}


def matrix_channel(superoperator, density, steps):
    result = np.linalg.matrix_power(superoperator, steps) @ density.reshape(-1, order="F")
    return result.reshape(density.shape, order="F")


def lower_bound_study():
    records = []
    for condition in [2, 10, 100, 1000]:
        matrix = np.diag([1.0, 1 / condition])
        rhs = np.array([1.0, 0.0])
        baseline = build_recurrent_solver(matrix, rhs)
        gap = 1 / (condition**2 + 1)
        degree = filter_degree(gap)
        values, vectors = np.linalg.eigh(baseline.residual_operator.T @ baseline.residual_operator)
        reference = np.eye(4)[:, 2]
        initial = np.diag([0.0, 1.0, 0.0, 0.0])
        target = np.array([1 / condition, 0, 1, 0])
        target /= np.linalg.norm(target)
        for method in ["original", "power_16", "chebyshev_ideal"]:
            c = np.sqrt(np.maximum(0, 1 - values))
            cost = 1
            if method == "power_16":
                c, cost = c**17, 17
            elif method == "chebyshev_ideal":
                c, cost = chebyshev_response(values, gap, degree), 2 * degree
            C = (vectors * c) @ vectors.T
            G = np.eye(4) - C.T @ C
            superoperator = np.kron(C, C) + np.outer(
                np.outer(reference, reference).reshape(-1, order="F"),
                G.T.reshape(-1, order="F"),
            )

            def error(steps, method=method, gap=gap, degree=degree):
                if method == "chebyshev_ideal":
                    eta = suppression_bound(gap, degree)
                    return float((gap + (1 - gap) * eta**2) ** steps)
                order = 17 if method == "power_16" else 1
                a = np.exp(order * np.log1p(-gap))
                survival = np.exp(order * steps * np.log1p(-gap))
                jump = -np.expm1(order * np.log1p(-gap))
                return float(survival + gap * jump * (survival - gap**steps) / (a - gap))

            low, high = 0, 1
            while error(high) > 1e-6:
                high *= 2
            while high - low > 1:
                middle = (low + high) // 2
                if error(middle) <= 1e-6:
                    high = middle
                else:
                    low = middle
            final = matrix_channel(superoperator, initial, high)
            direct_error = max(
                abs(
                    error(k)
                    - (1 - float(target @ matrix_channel(superoperator, initial, k) @ target))
                )
                for k in [0, 1, 2, 10, 100]
            )
            records.append(
                {
                    "condition": condition,
                    "method": method,
                    "degree": degree,
                    "updates_to_purified_distance_1e_3": high,
                    "primitive_queries": high * cost,
                    "final_purified_distance": float(np.sqrt(error(high))),
                    "previous_infidelity": error(high - 1),
                    "trace_error": float(abs(np.trace(final) - 1)),
                    "baseline_no_event_lower_bound": float(np.exp(high * np.log1p(-gap))),
                    "small_power_crosscheck_error": float(direct_error),
                    "simulation": "analytic population recurrence; small-K density cross-check",
                }
            )
    return records


def static_study(output, settings):
    rows = []
    for size in settings["sizes"]:
        for condition in settings["conditions"]:
            for family in settings["families"]:
                print(f"Chebyshev matched N={size} kappa={condition} family={family}", flush=True)
                matrix = fixture(size, condition, family, settings["seed"])
                rhs = np.random.default_rng(settings["seed"] + size).normal(size=size)
                reference = np.linalg.solve(matrix, rhs)  # scoring only
                file = f"inputs/n{size}-k{condition}-{family}.npz"
                checksum = save_input(
                    output / file, matrix=matrix, rhs=rhs, reference_solution=reference
                )
                start = time.perf_counter()
                base = build_density_channel(matrix, rhs)
                preprocessing = time.perf_counter() - start
                methods = [
                    (f"power_{m}", accelerate_density_channel(base, m), m + 1)
                    for m in settings["power_degrees"]
                ]
                filtered = filter_density_channel(base)
                methods.append(("chebyshev_ideal", filtered, 2 * filtered.filter_degree))
                initial_states = settings["small_initial_states"] if size <= 4 else ["reference"]
                for method, channel, calls in methods:
                    budget = 512 if condition == 2 else 4096 if condition == 10 else 16384
                    if method == "chebyshev_ideal":
                        budget = 64
                    times = np.unique(
                        np.r_[np.arange(min(17, budget + 1)), np.rint(np.geomspace(17, budget, 48))]
                    ).astype(int)
                    for initial in initial_states:
                        start = time.perf_counter()
                        trajectory = renewal_observables(channel, budget, times, initial, reference)
                        first = sustained(trajectory, settings)
                        rows.append(
                            {
                                "size": size,
                                "condition": condition,
                                "family": family,
                                "method": method,
                                "initial_state": initial,
                                "steps": budget,
                                "input_file": file,
                                "input_sha256": checksum,
                                "shared_preprocessing_seconds": preprocessing,
                                "simulation_seconds": time.perf_counter() - start,
                                "first_sustained_sampled_update": first,
                                "queries_to_sampled_accuracy": None
                                if first is None
                                else calls * first,
                                "accuracy_passed": first is not None,
                                "trajectory": trajectory,
                                "final": trajectory[-1],
                                "native_gate_simulation": False,
                                "budget_contract": (
                                    "same input/state/targets; macro caps differ; "
                                    "no equal-budget speedup claim"
                                ),
                            }
                        )
                save_json(output / "progress.json", {"static_completed": len(rows)})
    return rows


def _reduced_diagnostics(base, physical):
    output = density_diagnostics(base, physical)
    return {k: v for k, v in output.items() if not isinstance(v, np.ndarray)}


def native_study(output, settings):
    rows, phase_cache = [], {}
    for size in settings["native_sizes"]:
        for condition in settings["native_conditions"]:
            matrix = fixture(size, condition, "controlled_nonsymmetric", settings["seed"])
            rhs = np.random.default_rng(settings["seed"] + size).normal(size=size)
            gap = 1 / (condition**2 + 1)
            degree = filter_degree(gap)
            row = {"size": size, "condition": condition, "degree": degree}
            if degree > settings["maximum_phase_degree"]:
                rows.append(row | {"status": "resource_rejected", "reason": "phase degree budget"})
                continue
            start = time.perf_counter()
            baseline = build_recurrent_solver(matrix, rhs)
            preprocessing = time.perf_counter() - start
            input_file = f"inputs/native-n{size}-k{condition}.npz"
            input_sha = save_input(output / input_file, matrix=matrix, rhs=rhs)
            if condition not in phase_cache:
                start = time.perf_counter()
                spec = synthesize_filter(gap, max_degree=settings["maximum_phase_degree"])
                phase_cache[condition] = (spec, time.perf_counter() - start)
            spec, phase_time = phase_cache[condition]
            filter_circuit = qsp_filter_circuit(baseline, spec)
            dimension = 1 << baseline.data_qubits
            B = Operator(filter_circuit).data[:dimension, :dimension]
            values, vectors = np.linalg.eigh(
                baseline.residual_operator.T @ baseline.residual_operator
            )
            wanted = (
                vectors * (spec.attenuation * chebyshev_response(values, gap, degree))
            ) @ vectors.T
            block_error = float(np.max(abs(B - wanted)))
            points = []
            for initial_name in ["reference", "complex"]:
                initial = np.eye(dimension, dtype=complex)[:, size]
                if initial_name == "complex":
                    initial = np.arange(1, dimension + 1) + 1j * np.arange(dimension)
                    initial /= np.linalg.norm(initial)
                for workspace in ["coherent", "reset"]:
                    circuit = chebyshev_schedule_circuit(
                        [(baseline, spec)] * 2, workspace=workspace, initial_state=initial
                    )
                    if (
                        workspace == "reset"
                        and circuit.num_qubits > settings["maximum_reset_density_qubits"]
                    ):
                        points.append(
                            {
                                "workspace": workspace,
                                "initial_state": initial_name,
                                "status": "resource_rejected",
                                "reason": "complete density simulation width budget",
                            }
                        )
                        continue
                    reference_state = np.eye(dimension)[:, size]
                    rho = np.outer(initial, initial.conj())
                    for _ in range(2):
                        rho = B @ rho @ B.conj().T + np.trace(
                            (np.eye(dimension) - B.conj().T @ B) @ rho
                        ) * np.outer(reference_state, reference_state)
                    if workspace == "coherent":
                        actual = partial_trace(
                            Statevector.from_instruction(circuit),
                            list(range(baseline.data_qubits, circuit.num_qubits)),
                        ).data
                    else:
                        backend = AerSimulator(method="density_matrix", max_parallel_threads=1)
                        executable = circuit.copy()
                        executable.save_density_matrix(
                            list(range(baseline.data_qubits)), label="data"
                        )
                        executable = transpile(executable, backend, optimization_level=0)
                        result = backend.run(executable, shots=1, seed_simulator=42).result()
                        if not result.success:
                            raise RuntimeError(result.status)
                        actual = np.asarray(result.data(0)["data"])
                    file = f"circuits/cheb-n{size}-k{condition}-{initial_name}-{workspace}.qpy"
                    save_qpy_atomic(output / file, circuit)
                    points.append(
                        {
                            "workspace": workspace,
                            "status": "complete",
                            "initial_state": initial_name,
                            "maximum_density_error": float(np.max(abs(actual - rho))),
                            "width": circuit.num_qubits,
                            "qpy_file": file,
                        }
                    )
            circuit = chebyshev_schedule_circuit([(baseline, spec)], workspace="reset")
            start = time.perf_counter()
            native = transpile(
                circuit,
                basis_gates=["rz", "sx", "x", "cx", "reset"],
                optimization_level=1,
                seed_transpiler=42,
            )
            rows.append(
                row
                | {
                    "status": "compiled",
                    "preprocessing_seconds": preprocessing,
                    "phase_synthesis_seconds": phase_time,
                    "transpile_seconds": time.perf_counter() - start,
                    "phase_attenuation": spec.attenuation,
                    "phases": spec.phases.tolist(),
                    "input_file": input_file,
                    "input_sha256": input_sha,
                    "coefficient_error_estimate": spec.coefficient_error_estimate,
                    "maximum_block_error": block_error,
                    "cx": native.count_ops().get("cx", 0),
                    "depth": native.depth(),
                    "width": native.num_qubits,
                    "primitive_queries_per_macro": 2 * degree,
                    "complete_circuit_comparisons": points,
                }
            )
            save_json(output / "native-progress.json", rows)
    return rows


def cost_and_noise_study(output, settings):
    rows, noise_rows = [], []
    for condition in settings["cost_conditions"]:
        matrix = fixture(2, condition, "controlled_nonsymmetric", settings["seed"])
        rhs = np.random.default_rng(settings["seed"] + 2).normal(size=2)
        reference = np.linalg.solve(matrix, rhs)
        start = time.perf_counter()
        base_channel = build_density_channel(matrix, rhs)
        baseline = build_recurrent_solver(matrix, rhs)
        common_preprocessing = time.perf_counter() - start
        start = time.perf_counter()
        spec = synthesize_filter(1 / (condition**2 + 1))
        phase_synthesis = time.perf_counter() - start
        start = time.perf_counter()
        readout_rotations = [
            transpile(
                c, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=42
            )
            for c in coherence_readout_rotations(2)
        ]
        readout_compile = time.perf_counter() - start
        readout_rotation_time = (
            sum(
                scheduled_latency(c, cx_seconds=0.3e-6, one_qubit_seconds=30e-9, reset_seconds=0)
                for c in readout_rotations
            )
            / 3
        )
        for method in ["original", "power_4_dense", "power_16_dense", "chebyshev_qsp"]:
            start = time.perf_counter()
            if method == "chebyshev_qsp":
                filtered = filter_density_channel(base_channel)
                # Replay the actually synthesized signal response, not an ideal target.
                signal = qsp_filter_circuit(baseline, spec)
                B = Operator(signal).data[:4, :4]
                values = np.diag(base_channel.basis.T @ B @ base_channel.basis).real
                from dataclasses import replace

                channel = replace(
                    filtered,
                    no_jump_values=values,
                    residual_eigenvalues=np.maximum(0, 1 - values**2),
                )
                solver = None
            else:
                degree = 0 if method == "original" else 4 if method == "power_4_dense" else 16
                channel = accelerate_density_channel(base_channel, degree)
                solver = build_polynomial_solver(matrix, rhs, degree)
            transform_time = time.perf_counter() - start
            times = np.arange(4097)
            trajectory = renewal_observables(channel, 4096, times, "reference", reference)
            first = sustained(trajectory, settings)
            cost_first = next(
                (
                    p["time"]
                    for p in trajectory
                    if p["relative_residual"] is not None
                    and p["relative_residual"] <= settings["residual_target"]
                    and p["conditional_solution_fidelity"] is not None
                    and 1 - p["conditional_solution_fidelity"] <= settings["infidelity_target"]
                    and p["augmented_purified_distance"] <= 0.01 * baseline.singular_ratio / 200
                ),
                None,
            )
            row = {
                "condition": condition,
                "size": 2,
                "method": method,
                "transform_seconds": transform_time,
                "common_preprocessing_seconds": common_preprocessing,
                "readout_basis_compile_seconds": readout_compile,
                "readout_basis_cx_counts": [c.count_ops().get("cx", 0) for c in readout_rotations],
                "phase_synthesis_seconds": phase_synthesis if solver is None else 0.0,
                "first_joint_accuracy_update": first,
                "updates": cost_first,
                "accuracy_passed": first is not None,
                "final": trajectory[-1],
            }
            first = cost_first
            if first is None or first > settings["maximum_native_convergence_updates"]:
                rows.append(
                    row | {"status": "resource_rejected", "reason": "convergence update budget"}
                )
            else:
                circuit = (
                    chebyshev_schedule_circuit([(baseline, spec)] * first)
                    if solver is None
                    else solver.circuit(first, workspace="reset")
                )
                start = time.perf_counter()
                native = transpile(
                    circuit,
                    basis_gates=["rz", "sx", "x", "cx", "reset"],
                    optimization_level=1,
                    seed_transpiler=42,
                )
                compile_time = time.perf_counter() - start
                cx = native.count_ops().get("cx", 0)
                if cx > settings["maximum_native_cx"]:
                    rows.append(
                        row
                        | {"status": "resource_rejected", "reason": "complete CX budget", "cx": cx}
                    )
                else:
                    selected = trajectory[first]
                    executable = native.copy()
                    executable.save_density_matrix([0, 1], label="data")
                    backend = AerSimulator(method="density_matrix", max_parallel_threads=1)
                    run_start = time.perf_counter()
                    executed = backend.run(executable, shots=1, seed_simulator=42).result()
                    if not executed.success:
                        raise RuntimeError(executed.status)
                    full_density = np.asarray(executed.data(0)["data"])
                    full_seconds = time.perf_counter() - run_start
                    full_diagnostics = _reduced_diagnostics(baseline, full_density)
                    budgets = {
                        str(tol): certified_readout_budget(
                            2,
                            baseline.singular_ratio,
                            selected["augmented_purified_distance"],
                            residual_target=tol,
                        )
                        for tol in [1e-6, 1e-2]
                    }
                    models = []
                    for reset_us in [0.1, 1.0, 10.0]:
                        latency = scheduled_latency(
                            native,
                            cx_seconds=0.3e-6,
                            one_qubit_seconds=30e-9,
                            reset_seconds=reset_us * 1e-6,
                        )
                        for reuse in [1, 1000]:
                            shots = budgets["0.01"]["total_executions"]
                            models.append(
                                {
                                    "reset_microseconds": reset_us,
                                    "reuse": reuse,
                                    "modeled_circuit_seconds": latency,
                                    "modeled_total_seconds": None
                                    if shots is None
                                    else amortized_cost(
                                        compile_seconds=common_preprocessing
                                        + readout_compile
                                        + transform_time
                                        + compile_time
                                        + (phase_synthesis if solver is None else 0),
                                        reuse=reuse,
                                        executions=shots,
                                        circuit_seconds=latency,
                                        prepare_seconds=0,
                                        readout_seconds=1e-6 + readout_rotation_time,
                                    ),
                                    "assumption": (
                                        "CX=300ns, 1q=30ns, virtual RZ, readout=1us; uncalibrated"
                                    ),
                                }
                            )
                    save_qpy_atomic(
                        output / f"circuits/converged-k{condition}-{method}.qpy", native
                    )
                    rows.append(
                        row
                        | {
                            "status": "compiled",
                            "compile_seconds": compile_time,
                            "cx": cx,
                            "depth": native.depth(),
                            "width": native.num_qubits,
                            "reset_count": native.count_ops().get("reset", 0),
                            "complete_native_simulation_seconds": full_seconds,
                            "complete_native_diagnostics": full_diagnostics,
                            "readout_reserve_rule": "P <= (1e-2)*gamma/200, plus joint targets",
                            "selected_accuracy": selected,
                            "readout_budgets": budgets,
                            "assumed_timing_models": models,
                        }
                    )
            # All methods use the same finite 16-update noisy execution budget.
            circuit = (
                chebyshev_schedule_circuit([(baseline, spec)] * 16)
                if solver is None
                else solver.circuit(16, workspace="reset")
            )
            native = transpile(
                circuit,
                basis_gates=["rz", "sx", "x", "cx", "reset"],
                optimization_level=1,
                seed_transpiler=42,
            )
            for probability in [0.0, 1e-4, 1e-3]:
                noise = NoiseModel()
                if probability:
                    noise.add_all_qubit_quantum_error(depolarizing_error(probability, 2), "cx")
                    noise.add_all_qubit_quantum_error(
                        pauli_error([("X", 1e-3), ("I", 1 - 1e-3)]), "reset"
                    )
                backend = AerSimulator(
                    method="density_matrix", noise_model=noise, max_parallel_threads=1
                )
                executable = native.copy()
                executable.save_density_matrix([0, 1], label="data")
                start = time.perf_counter()
                result = backend.run(executable, shots=1, seed_simulator=42).result()
                if not result.success:
                    raise RuntimeError(result.status)
                physical = np.asarray(result.data(0)["data"])
                noise_rows.append(
                    {
                        "size": 2,
                        "condition": condition,
                        "method": method,
                        "cx_error": probability,
                        "reset_error": 1e-3 if probability else 0,
                        "cx": native.count_ops().get("cx", 0),
                        "width": native.num_qubits,
                        "steps": 16,
                        "seconds": time.perf_counter() - start,
                        **_reduced_diagnostics(baseline, physical),
                    }
                )
            save_json(output / "cost-progress.json", {"cost_cases": rows, "gate_noise": noise_rows})
    return rows, noise_rows


def logical_noise_study(output, settings):
    records = []
    for size in [n for n in settings["sizes"] if n >= 10]:
        for condition in [k for k in settings["conditions"] if k >= 100]:
            print(f"Chebyshev logical noise N={size} kappa={condition}", flush=True)
            matrix = fixture(size, condition, "controlled_nonsymmetric", settings["seed"])
            rhs = np.random.default_rng(settings["seed"] + size).normal(size=size)
            reference = np.linalg.solve(matrix, rhs)
            base = build_density_channel(matrix, rhs)
            observer = DensityObserver.create(base, reference)
            filtered = filter_density_channel(base)
            methods = [
                ("original", base, 1),
                ("power_16", accelerate_density_channel(base, 16), 17),
                ("chebyshev_ideal", filtered, 2 * filtered.filter_degree),
            ]
            file = f"inputs/logical-noise-n{size}-k{condition}.npz"
            checksum = save_input(
                output / file, matrix=matrix, rhs=rhs, reference_solution=reference
            )
            for method, channel, queries in methods:
                for p0 in [1e-4, 1e-3]:
                    for convention in ["fixed_macro", "query_coupled"]:
                        p = (
                            p0
                            if convention == "fixed_macro"
                            else float(-np.expm1(queries * np.log1p(-p0)))
                        )
                        times = np.unique(
                            np.r_[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
                        )
                        trajectory = renewal_observables(
                            channel, 4096, times, "reference", reference, p
                        )
                        stationary = stationary_depolarized(channel, p)
                        check = channel.step(stationary.copy(), depolarizing_probability=p)
                        records.append(
                            {
                                "size": size,
                                "condition": condition,
                                "method": method,
                                "noise_convention": convention,
                                "base_probability": p0,
                                "logical_probability_per_macro": p,
                                "queries_per_macro": queries,
                                "input_file": file,
                                "input_sha256": checksum,
                                "trajectory": trajectory,
                                "final": trajectory[-1],
                                "stationary": observer.read(stationary),
                                "stationary_equation_error": float(np.max(abs(check - stationary))),
                                "physical_gate_noise": False,
                            }
                        )
    return records


def build_report(output, data):
    import csv
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tables = {}
    for name, records in [
        (
            "static",
            [
                {k: v for k, v in r.items() if k != "trajectory"} | r["final"]
                for r in data["static_cases"]
            ],
        ),
        ("lower-bound", data["lower_bound"]),
        ("native", data["native_cases"]),
        ("cost", data["cost_cases"]),
        ("gate-noise", data["gate_noise_cases"]),
        (
            "logical-noise",
            [
                {k: v for k, v in r.items() if k != "trajectory"}
                | {
                    "stationary_residual": r["stationary"]["relative_residual"],
                    "transient_residual": r["final"]["relative_residual"],
                }
                for r in data["logical_noise_cases"]
            ],
        ),
    ]:
        simple = [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in records]
        keys = list(dict.fromkeys(k for r in simple for k in r))
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=keys)
        writer.writeheader()
        writer.writerows(simple)
        tables[name + ".csv"] = buf.getvalue()
        (output / (name + ".csv")).write_text(buf.getvalue())
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for method in ["original", "power_16", "chebyshev_ideal"]:
        rows = [r for r in data["lower_bound"] if r["method"] == method]
        ax[0].loglog(
            [r["condition"] for r in rows],
            [r["primitive_queries"] for r in rows],
            "o-",
            label=method,
        )
    ax[0].set_xlabel("Condition number")
    ax[0].set_ylabel("Residual-unitary queries to P <= 1e-3")
    ax[0].legend(fontsize=7)
    rows = [r for r in data["cost_cases"] if r["status"] == "compiled"]
    ax[1].bar(range(len(rows)), [r["cx"] for r in rows])
    ax[1].set_xticks(
        range(len(rows)), [f"{r['condition']}:{r['method']}" for r in rows], rotation=90, fontsize=6
    )
    ax[1].set_ylabel("Actually compiled CX to residual/infidelity criterion")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        buffer = io.BytesIO()
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else None
        fig.savefig(buffer, format=ext, dpi=180, metadata=metadata)
        (output / ("query-and-compiled-cost." + ext)).write_bytes(buffer.getvalue())
    plt.close(fig)
    return tables


def verify(output):
    import json

    output = Path(output).resolve()
    manifest = json.loads((output / "MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        path = (output / name).resolve()
        if (
            not path.is_relative_to(output)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"Chebyshev evidence checksum mismatch: {name}")
    data = json.loads((output / "result.json").read_text())
    cases = data["static_cases"] + data["logical_noise_cases"]
    cases += [r for r in data["native_cases"] if r["status"] == "compiled"]
    for row in cases:
        if (
            hashlib.sha256((output / row["input_file"]).read_bytes()).hexdigest()
            != row["input_sha256"]
        ):
            raise ValueError("Chebyshev input bytes differ from record")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    settings = configuration(args.config, "chebyshev-recurrent", DEFAULTS)
    output = Path(args.output_dir).resolve()
    (output / "inputs").mkdir(parents=True, exist_ok=True)
    (output / "circuits").mkdir(exist_ok=True)
    payload = {
        "configuration": settings,
        "provenance": provenance(Path.cwd()),
        "physical_hardware_tested": False,
        "lower_bound": lower_bound_study(),
    }
    payload["static_cases"] = static_study(output, settings)
    payload["native_cases"] = native_study(output, settings)
    payload["cost_cases"], payload["gate_noise_cases"] = cost_and_noise_study(output, settings)
    payload["logical_noise_cases"] = logical_noise_study(output, settings)
    payload["passed"] = all(
        r["maximum_block_error"] < 1e-9
        and all(
            p["maximum_density_error"] < 1e-9
            for p in r["complete_circuit_comparisons"]
            if p.get("status") == "complete"
        )
        for r in payload["native_cases"]
        if r["status"] == "compiled"
    )
    payload["passed"] &= all(
        r["complete_native_diagnostics"]["coherence_relative_residual"]
        <= 1.01 * settings["residual_target"]
        for r in payload["cost_cases"]
        if r["status"] == "compiled"
    )
    save_json(output / "result.json", payload)
    build_report(output, payload)
    save_json(
        output / "MANIFEST.json",
        {
            "files": {
                p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(output.rglob("*"))
                if p.is_file() and p.name != "MANIFEST.json"
            }
        },
    )
    verify(output)
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
