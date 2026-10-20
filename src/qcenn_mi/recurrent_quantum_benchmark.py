"""Audit complete recurrent circuits, convergence, final readout and state reuse.

Each prefix is a separate validation experiment. No measured prefix state or
classical reference is fed into a later circuit. All references are reporting
oracles only. This campaign never submits a physical quantum job.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, transpile
from qiskit_aer import AerSimulator

from .experiment_io import configuration, save_qpy_atomic, save_result
from .precision_benchmark import controlled_fixture
from .recurrent_quantum import (
    build_recurrent_solver,
    density_diagnostics,
    recurrent_resources,
    simulate_recurrent_schedule,
)

DEFAULTS = {
    "sizes": [2, 3, 4, 10],
    "families": ["controlled_spd", "controlled_nonsymmetric", "controlled_indefinite"],
    "condition": 2.0,
    "steps": [0, 1, 4, 16, 64],
    "coherent_steps": 3,
    "native_steps": 32,
    "shots": 4096,
    "seed": 1729,
    "minimum_augmented_fidelity": 0.99,
    "max_memory_mib": 256,
}


def _reference(operation):
    # Only the independent reporting oracle computes a classical solution.
    solution = np.linalg.solve(operation.matrix, operation.rhs)
    target = np.zeros(1 << operation.data_qubits)
    target[: operation.size] = (
        operation.singular_ratio
        * operation.matrix_scale
        * solution
        / np.linalg.norm(operation.rhs)
    )
    target[operation.reference_index] = 1
    target /= np.linalg.norm(target)
    return target, solution / np.linalg.norm(solution)


def _metrics(operation, result):
    target, direction = _reference(operation)
    conditional = result["conditional_solution_density"]
    return {
        "augmented_fidelity": float(np.vdot(target, result["density_matrix"] @ target).real),
        "conditional_solution_fidelity": None
        if conditional is None
        else float(np.vdot(direction, conditional @ direction).real),
        **{
            key: result[key]
            for key in (
                "solution_probability",
                "reference_probability",
                "padding_probability",
                "encoded_residual_expectation",
                "coherence_relative_residual",
                "trace",
                "purity",
            )
        },
    }


def _native_readout(output, settings):
    matrix = controlled_fixture(2, "controlled_nonsymmetric", 2, seed=settings["seed"])
    operation = build_recurrent_solver(matrix, [1, -0.5])
    body = operation.circuit(settings["native_steps"], workspace="reset")
    start = time.perf_counter()
    native = transpile(
        body,
        basis_gates=["rz", "sx", "x", "cx", "reset"],
        optimization_level=1,
        seed_transpiler=settings["seed"],
    )
    synthesis_seconds = time.perf_counter() - start
    measured = native.copy()
    classical = ClassicalRegister(operation.data_qubits, "final_data")
    measured.add_register(classical)
    measured.measure(list(range(operation.data_qubits)), classical)
    save_qpy_atomic(output / "recurrent-reset-native.qpy", [native, measured])
    coherent = operation.circuit(settings["coherent_steps"], workspace="coherent")
    save_qpy_atomic(output / "recurrent-coherent.qpy", coherent)
    coherent_native = transpile(
        coherent,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=1,
        seed_transpiler=settings["seed"],
    )
    coherent_measured = coherent_native.copy()
    coherent_bits = ClassicalRegister(operation.data_qubits, "final_data")
    coherent_measured.add_register(coherent_bits)
    coherent_measured.measure(list(range(operation.data_qubits)), coherent_bits)
    save_qpy_atomic(output / "recurrent-coherent-native.qpy", [coherent_native, coherent_measured])
    closed_backend = AerSimulator(method="statevector", max_parallel_threads=1)
    closed_diagnostic = coherent_native.copy()
    closed_diagnostic.save_density_matrix(list(range(operation.data_qubits)), label="final_data")
    closed_result = closed_backend.run(closed_diagnostic, shots=1).result()
    closed_density = np.asarray(closed_result.data(0)["final_data"])
    closed_reference = operation.simulate(settings["coherent_steps"], workspace="coherent")
    closed_discrepancy = float(
        np.linalg.norm(closed_density - closed_reference["density_matrix"])
    )
    closed_counts = closed_backend.run(
        coherent_measured, shots=settings["shots"], seed_simulator=settings["seed"]
    ).result().get_counts()
    backend = AerSimulator(method="density_matrix", max_parallel_threads=1)
    diagnostic = native.copy()
    diagnostic.save_density_matrix(list(range(operation.data_qubits)), label="final_data")
    job = backend.run(diagnostic, shots=1, seed_simulator=settings["seed"])
    density = np.asarray(job.result().data(0)["final_data"])
    reference = operation.simulate(settings["native_steps"])
    discrepancy = float(np.linalg.norm(density - reference["density_matrix"]))
    shot_job = backend.run(measured, shots=settings["shots"], seed_simulator=settings["seed"])
    counts = shot_job.result().get_counts()
    accepted = sum(count for bits, count in counts.items() if int(bits, 2) < operation.size)
    ideal = np.diag(reference["conditional_solution_density"]).real
    observed = np.array([counts.get(format(i, "02b"), 0) for i in range(2)], dtype=float)
    variation = None if accepted == 0 else float(np.abs(observed / accepted - ideal).sum() / 2)
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    noise_cases = []
    for probability in (1e-5, 1e-4):
        noise = NoiseModel()
        noise.add_all_qubit_quantum_error(depolarizing_error(probability, 2), "cx")
        noisy_backend = AerSimulator(
            method="density_matrix", noise_model=noise, max_parallel_threads=1
        )
        noisy_result = noisy_backend.run(diagnostic, shots=1).result()
        noisy_density = np.asarray(noisy_result.data(0)["final_data"])
        metrics = _metrics(operation, density_diagnostics(operation, noisy_density))
        noise_cases.append(
            {
                "cx_depolarizing_probability": probability,
                "scope": "synthetic CX-only noise; ideal resets, 1-qubit gates and readout",
                **metrics,
                "quality_gate": metrics["conditional_solution_fidelity"] >= 0.99,
            }
        )
    return {
        "size": 2,
        "steps": settings["native_steps"],
        "workspace": "reset",
        "qubits": native.num_qubits,
        "native_depth": native.depth(),
        "native_operations": dict(native.count_ops()),
        "synthesis_seconds": synthesis_seconds,
        "native_density_discrepancy": discrepancy,
        "shots": settings["shots"],
        "accepted_solution_indices": accepted,
        "raw_final_z_counts": counts,
        "conditional_z_total_variation": variation,
        "shot_job_id": shot_job.job_id(),
        "backend": "AerSimulator, ideal density_matrix",
        "physical_hardware_tested": False,
        "signed_inverse_reconstructed_from_counts": False,
        "coherent_native": {
            "steps": settings["coherent_steps"],
            "qubits": coherent_native.num_qubits,
            "depth": coherent_native.depth(),
            "operations": dict(coherent_native.count_ops()),
            "native_density_discrepancy": closed_discrepancy,
            "raw_final_z_counts": closed_counts,
            "intermediate_measurements": 0,
            "device_resets": 0,
            "classical_iterate_updates": 0,
            "physical_hardware_tested": False,
        },
        "synthetic_noise_cases": noise_cases,
        "passed": discrepancy < 1e-10
        and closed_discrepancy < 1e-10
        and sum(counts.values()) == settings["shots"]
        and sum(closed_counts.values()) == settings["shots"],
    }


def _plot(output, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(7, 4.4), layout="constrained")
    for size in sorted({row["size"] for row in rows}):
        selected = [
            row
            for row in rows
            if row["size"] == size and row["family"] == "controlled_nonsymmetric"
        ]
        if selected:
            axes.semilogy(
                [row["steps"] for row in selected],
                [max(1 - row["augmented_fidelity"], 1e-15) for row in selected],
                "o-",
                label=f"{size} x {size}",
            )
    axes.set(xlabel="Quantum channel updates", ylabel="Augmented-state infidelity")
    axes.set_title("Ideal full-circuit convergence; dense residual encoding")
    axes.grid(alpha=0.25)
    if axes.lines:
        axes.legend()
    figure.savefig(output / "convergence.svg")
    figure.savefig(output / "convergence.png", dpi=180)
    plt.close(figure)


def campaign(output, *, config=None, settings=None):
    settings = {**configuration(config, "recurrent-quantum", DEFAULTS), **(settings or {})}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (
        not settings["sizes"]
        or not settings["families"]
        or not settings["steps"]
        or settings["steps"] != sorted(set(settings["steps"]))
        or not 0 < settings["minimum_augmented_fidelity"] <= 1
        or settings["shots"] < 1
    ):
        raise ValueError(
            "Nonempty sizes/families, increasing steps and valid quality/shots required"
        )
    rows, comparisons = [], []
    for size in settings["sizes"]:
        for family in settings["families"]:
            matrix = controlled_fixture(size, family, settings["condition"], seed=settings["seed"])
            rhs = np.random.default_rng(settings["seed"] + size).normal(size=size)
            operation = build_recurrent_solver(matrix, rhs)
            for steps in settings["steps"]:
                start = time.perf_counter()
                result = operation.simulate(steps, max_memory_mib=settings["max_memory_mib"])
                row = {
                    "size": size,
                    "family": family,
                    "condition": settings["condition"],
                    "steps": steps,
                    "qubits": result["resources"]["total_qubits"],
                    "wall_seconds": time.perf_counter() - start,
                    **_metrics(operation, result),
                    **result["certificate"],
                }
                row["certificate_passed"] = (
                    1 - row["augmented_fidelity"]
                    <= row["augmented_infidelity_upper_bound"] + 1e-10
                )
                rows.append(row)
            coherent = operation.simulate(
                settings["coherent_steps"],
                workspace="coherent",
                max_memory_mib=settings["max_memory_mib"],
            )
            reset = operation.simulate(settings["coherent_steps"])
            discrepancy = float(
                np.linalg.norm(coherent["density_matrix"] - reset["density_matrix"])
            )
            comparisons.append(
                {
                    "size": size,
                    "family": family,
                    "steps": settings["coherent_steps"],
                    "full_coherent_qubits": coherent["resources"]["total_qubits"],
                    "reset_qubits": reset["resources"]["total_qubits"],
                    "density_discrepancy": discrepancy,
                    "matrix": matrix.tolist(),
                    "rhs": rhs.tolist(),
                    "passed": discrepancy < 1e-10,
                }
            )
    first = build_recurrent_solver([[2, 0.4], [-0.1, 1.5]], [1, -0.5])
    last = build_recurrent_solver([[1.8, 0.2], [0.1, 1.6]], [1, -0.5])
    schedule = [(first, 16), (last, max(settings["steps"]))]
    temporal_result = simulate_recurrent_schedule(schedule)
    temporal = {
        "schedule": [
            {"matrix": operation.matrix.tolist(), "rhs": operation.rhs.tolist(), "steps": steps}
            for operation, steps in schedule
        ],
        "data_reprepared_between_snapshots": False,
        "scope": "preprogrammed sequence; no live matrix acquisition or drift guarantee",
        **_metrics(last, temporal_result),
    }
    hard_probes = []
    for condition in (10, 100):
        hard_matrix = controlled_fixture(
            2, "controlled_nonsymmetric", condition, seed=settings["seed"]
        )
        hard = build_recurrent_solver(hard_matrix, [1, -0.5])
        hard_result = hard.simulate(max(settings["steps"]))
        metrics = _metrics(hard, hard_result)
        hard_probes.append(
            {
                "size": 2,
                "condition": condition,
                "steps": max(settings["steps"]),
                **metrics,
                **hard_result["certificate"],
                "meets_accuracy_target": metrics["augmented_fidelity"]
                >= settings["minimum_augmented_fidelity"],
                "certificate_passed": 1 - metrics["augmented_fidelity"]
                <= hard_result["certificate"]["augmented_infidelity_upper_bound"] + 1e-10,
                "scope": "accuracy stress probe; retained separately from condition-2 cases",
            }
        )
    native = _native_readout(output, settings)
    final_rows = [row for row in rows if row["steps"] == max(settings["steps"])]
    passed = (
        all(row["certificate_passed"] for row in rows)
        and all(row["passed"] for row in comparisons)
        and all(
            row["augmented_fidelity"] >= settings["minimum_augmented_fidelity"]
            for row in final_rows
        )
        and temporal["augmented_fidelity"] >= settings["minimum_augmented_fidelity"]
        and all(row["certificate_passed"] for row in hard_probes)
        and native["passed"]
    )
    payload = {
        "schema_version": 1,
        "passed": bool(passed),
        "settings": settings,
        "method": "augmented residual collision quantum channel; dense input compilation",
        "rhs_dependent_encoding": True,
        "classical_iterate_updates": 0,
        "intermediate_data_measurements": 0,
        "physical_hardware_tested": False,
        "all_synthetic_quality_gates_passed": all(
            row["quality_gate"] for row in native["synthetic_noise_cases"]
        ),
        "cases": rows,
        "coherent_reset_comparisons": comparisons,
        "preprogrammed_temporal_case": temporal,
        "hard_condition_probes": hard_probes,
        "native_final_readout": native,
        "width_estimates_only": [
            recurrent_resources(size, 64, workspace=workspace)
            for size in (100, 1000)
            for workspace in ("coherent", "reset")
        ],
        "limitations": [
            "Dense classical gate compilation, not an efficient quantum input oracle or speedup.",
            "RHS-dependent solver state; not the existing input-independent inverse operator.",
            "Separate residual-channel prototype; no Chebyshev or QSVT acceleration implemented.",
            "Coherent workspace grows with iteration count; reset mode discards device ancillas.",
            "Convergence certificates concern ideal static gates, not arbitrary hardware noise.",
            "Order-100/1000 entries are width estimates, not simulation or hardware execution.",
            "Final Z counts do not recover signed amplitudes or a complete classical inverse.",
        ],
    }
    save_result(output, payload)
    _plot(output, rows)
    lines = [
        "# Recurrent quantum circuit experiment",
        "",
        f"Software/numerical verdict: {'PASS' if passed else 'FAIL'}.",
        "Physical hardware: not run. Synthetic CX-only noise is reported separately below.",
        "",
        "Every update executes in one quantum circuit. The coherent circuit retains its",
        "environment; the reset circuit reuses it on device. Neither reads or reconstructs",
        "the data register between iterations. Classical solves are independent validation only.",
        "",
        "| Size | Family | Updates | Augmented fidelity | Conditional solution fidelity |",
        "|---|---|---|---|---|",
    ]
    for row in final_rows:
        lines.append(
            f"| {row['size']} | {row['family']} | {row['steps']} | "
            f"{row['augmented_fidelity']:.10f} | {row['conditional_solution_fidelity']:.10f} |"
        )
    lines += [
        "",
        "## Native circuit and synthetic noise",
        "",
        f"Order 2, {native['steps']} updates: {native['qubits']} qubits, "
        f"{native['native_operations'].get('cx', 0)} CX gates, depth {native['native_depth']}.",
        f"Final ideal Z readout: {native['accepted_solution_indices']}/{native['shots']} "
        "outcomes in the solution subspace. Signed inverse entries are not reconstructed.",
        "",
        "| CX depolarizing probability | Conditional solution fidelity | 0.99 quality gate |",
        "|---|---|---|",
    ]
    for row in native["synthetic_noise_cases"]:
        lines.append(
            f"| {row['cx_depolarizing_probability']} | "
            f"{row['conditional_solution_fidelity']:.8f} | {row['quality_gate']} |"
        )
    lines += [
        "",
        "These are CX-only synthetic channels with ideal resets, 1-qubit gates and readout.",
        "Their quality verdict does not change the software implementation verdict.",
        "",
        "## Conditioning stress probes",
        "",
        "| Condition number | Updates | Augmented fidelity | Meets accuracy target |",
        "|---|---|---|---|",
    ]
    for row in hard_probes:
        lines.append(
            f"| {row['condition']} | {row['steps']} | "
            f"{row['augmented_fidelity']:.8f} | {row['meets_accuracy_target']} |"
        )
    lines += [
        "",
        "Slow convergence at higher conditioning is retained as an accuracy limitation.",
        "",
        "## Limits",
        "",
        *(f"- {item}" for item in payload["limitations"]),
        "",
    ]
    (output / "SUMMARY.md").write_text("\n".join(lines))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/recurrent-quantum")
    parser.add_argument("--config")
    args = parser.parse_args()
    result = campaign(args.output_dir, config=args.config)
    print(json.dumps({"passed": result["passed"], "cases": len(result["cases"])}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
