"""Native-gate preparation and synthetic-noise verification for compact inversion.

A 42-qubit budget is a capacity check, not a calibration or device-specific
hardware guarantee. Noise profiles and line connectivity are explicit stand-ins.
No real jobs are submitted automatically.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
from qiskit import qpy, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from .coherent_large_benchmark import fixture
from .compact import build_compact_inverse

NATIVE_BASIS = ["rz", "sx", "x", "cx"]
PROFILES = {
    "optimistic_synthetic": {"one_qubit": 1e-5, "two_qubit": 1e-4, "readout": 0.005},
    "stress_synthetic": {"one_qubit": 1e-4, "two_qubit": 1e-3, "readout": 0.02},
}


def compile_native(
    circuit,
    *,
    qubit_budget=42,
    topology="line",
    backend=None,
    optimization_level=2,
    seed_transpiler=42,
):
    if (
        isinstance(optimization_level, bool)
        or not isinstance(optimization_level, int)
        or optimization_level not in range(4)
    ):
        raise ValueError("optimization_level must be 0, 1, 2, or 3")
    if not isinstance(qubit_budget, int) or qubit_budget < circuit.num_qubits:
        raise ValueError("Circuit exceeds the configured device qubit budget")
    if backend is not None:
        if backend.num_qubits < circuit.num_qubits:
            raise ValueError("Backend has insufficient qubits")
        return transpile(
            circuit, backend, optimization_level=optimization_level, seed_transpiler=seed_transpiler
        )
    if topology not in ("line", "all_to_all"):
        raise ValueError("Unknown topology")
    coupling = CouplingMap.from_line(circuit.num_qubits) if topology == "line" else None
    return transpile(
        circuit,
        basis_gates=NATIVE_BASIS,
        coupling_map=coupling,
        initial_layout=list(range(circuit.num_qubits)),
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )


def logical_statevector(compiled, physical):
    """Restore logical bit order after routing, including the data/ancilla split."""
    width = compiled.num_qubits
    mapping = (
        list(range(width)) if compiled.layout is None else compiled.layout.final_index_layout()
    )
    if len(mapping) != width:
        raise ValueError(
            "Simulation helper expects an active-width target without extra idle wires"
        )
    indices = np.arange(1 << width, dtype=np.uint64)
    logical = np.zeros_like(indices)
    for bit, physical_bit in enumerate(mapping):
        logical |= ((indices >> physical_bit) & 1) << bit
    out = np.empty_like(physical)
    out[logical.astype(int)] = physical
    return out


def noise_model(profile):
    if any(not 0 <= value <= 1 for value in profile.values()):
        raise ValueError("Noise probabilities must be between zero and one")
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(profile["one_qubit"], 1), ["sx", "x"])
    noise.add_all_qubit_quantum_error(depolarizing_error(profile["two_qubit"], 2), ["cx"])
    p = profile["readout"]
    noise.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return noise


def wilson(successes, total):
    if total == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = successes / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [float(max(0, min(p, center - radius))), float(min(1, max(p, center + radius)))]


def read_echo_counts(counts, *, data_qubits, total_qubits):
    """Ancilla success and all-zero reference return, in original classical order.

    Conditional return includes verifier gate errors and readout errors; it is
    not a tomography-derived state fidelity for a noisy device.
    """
    total, accepted, returned = 0, 0, 0
    for bits, count in counts.items():
        bits = bits.replace(" ", "").zfill(total_qubits)
        total += count
        if set(bits[:-data_qubits]) <= {"0"}:
            accepted += count
            if set(bits) <= {"0"}:
                returned += count
    fraction = returned / accepted if accepted else None
    return {
        "shots": total,
        "accepted": accepted,
        "returned": returned,
        "success_frequency": accepted / total if total else None,
        "unconditional_success_95ci": wilson(accepted, total),
        "effective_good_output_probability": returned / total if total else None,
        "effective_good_output_95ci": wilson(returned, total),
        "too_few_accepted_shots": accepted < 20,
        "observed_echo_return": fraction,
        "echo_return_95ci": wilson(returned, accepted),
        "quality_gate": bool(accepted >= 20 and wilson(returned, accepted)[0] >= 0.9),
        "raw_counts": dict(counts),
    }


def run_verification(circuit, *, backend=None, shots=32768, seed=42):
    """Execute only when explicitly called; backend must support Backend.run.

    Real providers exposing only runtime primitives need a provider adapter.
    """
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ValueError("shots must be a positive integer")
    backend = AerSimulator(max_parallel_threads=1) if backend is None else backend
    options = {"shots": shots}
    if isinstance(backend, AerSimulator):
        options["seed_simulator"] = seed
    job = backend.run(circuit, **options)
    result = job.result()
    return result.get_counts(), str(job.job_id())


def campaign(
    output,
    *,
    sizes=(2, 4, 10, 100),
    tolerance=1e-6,
    qubit_budget=42,
    shots=32768,
    large_noise_shots=64,
    run_noise=True,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audits = output / "audit"
    audits.mkdir(exist_ok=True)
    rows = []
    for n in sizes:
        matrix = fixture(n, "dense_spd")
        rhs = np.ones(n) / np.sqrt(n)
        operation = build_compact_inverse(matrix, tolerance=tolerance)
        ideal = operation.simulate(rhs)
        action = operation.circuit(rhs)
        compiled = compile_native(action, qubit_budget=qubit_budget)
        # Independent native statevector check, with final routing permutation restored.
        saved = compiled.copy()
        saved.save_statevector()
        result = AerSimulator(method="statevector", max_parallel_threads=1).run(saved).result()
        vector = logical_statevector(compiled, np.asarray(result.get_statevector()))
        native_solution = np.real_if_close(vector[:n] * operation.normalization).real
        native_residual = float(np.linalg.norm(matrix @ native_solution - rhs))
        # Classical solve is only a verifier reference, after building/executing the inverse.
        reference = np.linalg.solve(matrix, rhs)
        verifier = operation.circuit(rhs, verification_state=reference)
        verifier.measure_all()
        compiled_verifier = compile_native(verifier, qubit_budget=qubit_budget)
        filename = f"{n}-inverse-action.qpy"
        with (output / filename).open("wb") as handle:
            qpy.dump([compiled, compiled_verifier], handle)
        counts, job = run_verification(compiled_verifier, shots=shots)
        exact = read_echo_counts(
            counts,
            data_qubits=operation.resource_report["data_qubits"],
            total_qubits=compiled_verifier.num_qubits,
        )
        exact["job_id"] = job
        noise_rows = []
        if run_noise:
            for name, profile in PROFILES.items():
                if n >= 100 and name != "optimistic_synthetic":
                    continue
                # Use bounded stochastic trajectories at large width rather than a
                # prohibitively costly exact density-matrix channel calculation.
                method = "statevector" if n >= 100 else "density_matrix"
                budget = large_noise_shots if n >= 100 else shots
                backend = AerSimulator(
                    method=method, noise_model=noise_model(profile), max_parallel_threads=1
                )
                started = time.perf_counter()
                noisy_counts, job = run_verification(
                    compiled_verifier, backend=backend, shots=budget, seed=42
                )
                noise_result = read_echo_counts(
                    noisy_counts,
                    data_qubits=operation.resource_report["data_qubits"],
                    total_qubits=compiled_verifier.num_qubits,
                )
                noise_result.update(
                    profile=name,
                    parameters=profile,
                    job_id=job,
                    simulation_method=method,
                    seconds=time.perf_counter() - started,
                )
                audit_file = audits / f"{n}-{name}.json"
                audit_file.write_text(json.dumps(noise_result, indent=2) + "\n")
                noise_result.pop("raw_counts")
                noise_result["audit_file"] = str(audit_file.relative_to(output))
                noise_rows.append(noise_result)
        exact_audit = audits / f"{n}-ideal.json"
        exact_audit.write_text(json.dumps(exact, indent=2) + "\n")
        exact.pop("raw_counts")
        exact["audit_file"] = str(exact_audit.relative_to(output))
        rows.append(
            {
                "size": n,
                "input_family": "dense_spd",
                "degree": operation.degree,
                "required_qubits": operation.resource_report["total_qubits"],
                "data_qubits": operation.resource_report["data_qubits"],
                "device_qubit_budget": qubit_budget,
                "fits_qubit_budget": True,
                "classical_encoding": operation.encoding.encoding_kind,
                "residual_bound": operation.residual_bound,
                "structured_ideal_residual": ideal["relative_residual"],
                "native_ideal_residual": native_residual,
                "ideal_success_probability": ideal["success_probability"],
                "native_action_operations": dict(compiled.count_ops()),
                "native_action_depth": compiled.depth(),
                "native_verifier_operations": dict(compiled_verifier.count_ops()),
                "native_verifier_depth": compiled_verifier.depth(),
                "topology": "active-width bidirectional line; not a calibrated 42-qubit backend",
                "basis_gates": NATIVE_BASIS,
                "qpy_file": filename,
                "qpy_circuit_order": ["inverse_action", "reference_verifier_with_measurements"],
                "ideal_echo": exact,
                "noise": noise_rows,
                "native_correctness_passed": native_residual < tolerance and exact["quality_gate"],
            }
        )
        print(f"Completed hardware simulation case {n}x{n}", flush=True)
    passed = all(r["native_correctness_passed"] for r in rows)
    all_noise_quality = (
        all(p["quality_gate"] for r in rows for p in r["noise"]) if run_noise else None
    )
    payload = {
        "native_validation_passed": bool(passed),
        "all_synthetic_quality_gates_passed": all_noise_quality,
        "physical_hardware_tested": False,
        "calibration_derived_noise": False,
        "qubit_budget": qubit_budget,
        "target_tolerance": tolerance,
        "cases": rows,
        "noise_quality_gate": (
            "at least 20 accepted shots and 95% Wilson lower bound of reference return >= 0.9"
        ),
        "versions": {n: importlib.metadata.version(n) for n in ("numpy", "qiskit", "qiskit-aer")},
        "limits": (
            "Synthetic noise, assumed line topology; 100x100 uses bounded trajectory statistics. "
            "Echo includes reference-verifier errors. No full inverse tomography."
        ),
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Compact inverse hardware-simulation preparation",
        "",
        f"Native ideal correctness: {'PASS' if passed else 'FAIL'}",
        "",
        f"All synthetic-noise quality gates passed: {all_noise_quality}",
        "",
        "This is synthetic hardware simulation, not physical or calibration-derived execution.",
        "",
        "| Matrix | Data / total qubits | Ideal residual | Success | Routed CX | Depth |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['size']} | {r['data_qubits']} / {r['required_qubits']} | "
            f"{r['native_ideal_residual']:.3e} | {r['ideal_success_probability']:.1%} | "
            f"{r['native_action_operations'].get('cx', 0)} | {r['native_action_depth']} |"
        )
    lines += [
        "",
        "## Synthetic native-gate noise",
        "",
        "| Matrix | Profile | Shots | Accepted | Echo return | 95% interval | Gate |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for r in rows:
        for p in r["noise"]:
            value = p["observed_echo_return"]
            lines.append(
                f"| {r['size']} | {p['profile']} | {p['shots']} | {p['accepted']} | "
                f"{value if value is not None else 'undefined'} | "
                f"{p['echo_return_95ci']} | {'PASS' if p['quality_gate'] else 'FAIL'} |"
            )
    lines += [
        "",
        "A passing native simulation does not imply noisy or physical hardware accuracy.",
        "Poor noisy quality is reported, not hidden by classical correction or a passing CLI exit.",
        "The verification circuit appends an independently computed reference-state projection.",
        "QPY contains action and verifier circuits; "
        "a real backend needs recompilation with its target/calibration.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/hardware")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 10, 100])
    parser.add_argument("--qubit-budget", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--shots", type=int, default=32768)
    parser.add_argument("--large-noise-shots", type=int, default=64)
    parser.add_argument("--skip-noise", action="store_true")
    parser.add_argument("--require-noisy-quality", action="store_true")
    args = parser.parse_args()
    result = campaign(
        args.output_dir,
        sizes=args.sizes,
        tolerance=args.tolerance,
        qubit_budget=args.qubit_budget,
        shots=args.shots,
        large_noise_shots=args.large_noise_shots,
        run_noise=not args.skip_noise,
    )
    print((Path(args.output_dir) / "summary.md").read_text())
    passed = result["native_validation_passed"] and (
        not args.require_noisy_quality or result["all_synthetic_quality_gates_passed"] is True
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
