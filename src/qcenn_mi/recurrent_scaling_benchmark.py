"""Large static/noisy/time-varying recurrent quantum-channel simulations.

Large cases are exact reduced-density channel simulations, not native-gate or
hardware runs. Matrix/RHS inputs are archived. Classical references and x1
observations never feed a later quantum-channel update. Logical noise is a
global padded-data depolarizing channel once per update, not a gate error rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import depolarizing_error

from .experiment_io import configuration, save_npz_atomic, save_result
from .precision_benchmark import controlled_fixture
from .recurrent_density import DensityObserver, build_density_channel
from .recurrent_quantum import build_recurrent_solver, recurrent_circuit
from .validation import provenance

DEFAULTS = {
    "sizes": [10, 100, 1000],
    "conditions": [2, 10],
    "families": ["controlled_spd", "controlled_nonsymmetric", "controlled_indefinite"],
    "noise_probabilities": [0, 1e-5, 1e-4, 1e-3, 1e-2],
    "steps": {"2": 512, "10": 4096},
    "record_points": 65,
    "dynamic_steps": 1600,
    "dynamic_noise_probabilities": [0, 1e-4, 1e-3],
    "dynamic_stride": 8,
    "stress_condition": 100,
    "stress_steps": 4096,
    "seed": 1729,
    "workers": 3,
    "ideal_residual_target": 1e-6,
    "noisy_residual_target": 1e-2,
    "max_memory_mib": 512,
}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checkpoints(steps, points):
    return set(np.unique(np.rint(np.linspace(0, steps, points)).astype(int)).tolist())


def _trajectory(channel, reference, steps, probability, settings):
    observer = DensityObserver.create(channel, reference)
    density = channel.initial_density()
    work = np.empty_like(density)
    checkpoints = _checkpoints(steps, settings["record_points"])
    rows = []
    start = time.perf_counter()
    for step in range(steps + 1):
        if step in checkpoints:
            record = {"step": step, **observer.read(density)}
            record["ideal_infidelity_bound"] = channel.certificate(step)
            rows.append(record)
        if step < steps:
            channel.step(density, depolarizing_probability=probability, work=work)
    final = rows[-1]
    final["minimum_density_eigenvalue"] = float(np.linalg.eigvalsh(density)[0])
    target = settings["noisy_residual_target"] if probability else settings["ideal_residual_target"]
    integrity = (
        max(abs(row["trace"] - 1) for row in rows) < 1e-9
        and final["minimum_density_eigenvalue"] > -1e-10
        and all(np.isfinite(row["relative_residual"]) for row in rows)
    )
    certified = probability != 0 or all(
        1 - row["augmented_fidelity"] <= row["ideal_infidelity_bound"] + 1e-9 for row in rows
    )
    return {
        "steps": steps,
        "logical_depolarizing_probability": probability,
        "method": "exact reduced-density spectral channel simulation",
        "wall_seconds": time.perf_counter() - start,
        "residual_target": target,
        "accuracy_passed": final["relative_residual"] <= target,
        "numerical_integrity_passed": bool(integrity and certified),
        "ideal_certificate_checked": probability == 0,
        "final": final,
        "trajectory": rows,
    }


def _static_fixture(task):
    output, size, condition, family, settings, stress = task
    output = Path(output)
    identity = f"n{size}-{family}-k{condition}"
    matrix = controlled_fixture(size, family, condition, seed=settings["seed"])
    rng = np.random.default_rng(settings["seed"] + size)
    right_sides = {"signed_random": rng.normal(size=size)}
    if not stress:
        right_sides.update(basis_first=np.eye(1, size).ravel(), uniform=np.ones(size))
    results, inputs = [], {"matrix": matrix}
    compile_seconds = 0.0
    for kind, rhs in right_sides.items():
        start = time.perf_counter()
        channel = build_density_channel(matrix, rhs, max_memory_mib=settings["max_memory_mib"])
        compile_seconds += time.perf_counter() - start
        # This reference is only read by DensityObserver, never by channel.step.
        reference = np.linalg.solve(matrix, rhs)
        inputs[f"rhs_{kind}"] = rhs
        inputs[f"reference_{kind}"] = reference
        probabilities = (
            settings["noise_probabilities"] if kind == "signed_random" and not stress else [0]
        )
        steps = settings["stress_steps"] if stress else settings["steps"][str(condition)]
        for probability in probabilities:
            result = _trajectory(channel, reference, steps, probability, settings)
            result.update(
                id=f"{identity}-{kind}-p{probability:g}",
                size=size,
                condition=condition,
                family=family,
                rhs_kind=kind,
                stress_probe=stress,
            )
            results.append(result)
    path = output / "inputs" / f"{identity}.npz"
    save_npz_atomic(path, **inputs)
    return {
        "cases": results,
        "input": {
            "id": identity,
            "file": str(path.relative_to(output)),
            "sha256": file_sha(path),
            "dense_encoding_seconds": compile_seconds,
            "generation_seed": settings["seed"],
            "size": size,
            "condition": condition,
            "family": family,
        },
    }


def dynamic_angle(step):
    """Continuous rotation after settling, an abrupt jump, then a final hold."""
    active = min(max(step - 128, 0), 1272)
    angle = 0.4 * np.sin(2 * np.pi * active / 1200)
    if step >= 960:
        angle += 0.5
    return float(angle)


def _dynamic_fixture(task):
    output, size, settings = task
    output = Path(output)
    matrix = controlled_fixture(size, "controlled_nonsymmetric", 2, seed=settings["seed"])
    manufactured = np.random.default_rng(settings["seed"] + 2000 + size).normal(size=size)
    manufactured[:2] = [1, 0.4]
    rhs = matrix @ manufactured
    channel = build_density_channel(matrix, rhs, max_memory_mib=settings["max_memory_mib"])
    reference = np.linalg.solve(matrix, rhs)
    observer = DensityObserver.create(channel, reference)
    rows = []
    for probability in settings["dynamic_noise_probabilities"]:
        density = channel.initial_density()
        work = np.empty_like(density)
        previous = 0.0
        trajectory = []
        start = time.perf_counter()
        for step in range(settings["dynamic_steps"] + 1):
            theta = dynamic_angle(step)
            channel.change_rotation_basis(density, theta - previous)
            if step:
                channel.step(density, depolarizing_probability=probability, work=work)
            if step % settings["dynamic_stride"] == 0 or step == settings["dynamic_steps"]:
                trajectory.append(
                    {"step": step, "theta": theta, **observer.read(density, theta=theta)}
                )
            previous = theta
        rows.append(
            {
                "size": size,
                "condition": 2,
                "logical_depolarizing_probability": probability,
                "steps": settings["dynamic_steps"],
                "data_reinitialized": False,
                "matrix_family": "M(k)=M(0)G(theta(k)); fixed spectrum, rotating first two columns",
                "method": "exact reduced-density channel with rank-two changes of simulation basis",
                "wall_seconds": time.perf_counter() - start,
                "final": trajectory[-1],
                "maximum_x1_error_after_initial_settling": max(
                    row["absolute_x1_error"] for row in trajectory if row["step"] >= 256
                ),
                "minimum_density_eigenvalue": float(np.linalg.eigvalsh(density)[0]),
                "trajectory": trajectory,
            }
        )
    path = output / "inputs" / f"n{size}-dynamic-rhs.npz"
    save_npz_atomic(
        path,
        rhs=rhs,
        manufactured_seed_solution=manufactured,
        independent_reference_solution=reference,
        theta=np.array([dynamic_angle(step) for step in range(settings["dynamic_steps"] + 1)]),
    )
    return {
        "cases": rows,
        "input": {"file": str(path.relative_to(output)), "sha256": file_sha(path)},
    }


def circuit_crosschecks():
    records = []
    for size in (2, 3, 10):
        matrix = controlled_fixture(size, "controlled_nonsymmetric", 2, seed=321)
        rhs = np.random.default_rng(321 + size).normal(size=size)
        spectral = build_density_channel(matrix, rhs)
        dense = build_recurrent_solver(matrix, rhs)
        for probability in (0, 0.003):
            original = recurrent_circuit([(dense, 3)], workspace="reset")
            circuit = QuantumCircuit(original.num_qubits)
            swaps = 0
            for item in original.data:
                circuit.append(item.operation, [original.find_bit(q).index for q in item.qubits])
                if item.operation.name == "cswap":
                    swaps += 1
                    if probability and swaps % dense.data_qubits == 0:
                        circuit.append(
                            depolarizing_error(probability, dense.data_qubits).to_instruction(),
                            list(range(dense.data_qubits)),
                        )
            circuit.save_density_matrix(list(range(dense.data_qubits)), label="final_data")
            backend = AerSimulator(method="density_matrix", max_parallel_threads=1)
            executable = transpile(circuit, backend, optimization_level=0)
            actual = np.asarray(backend.run(executable, shots=1).result().data(0)["final_data"])
            expected = spectral.initial_density()
            for _ in range(3):
                spectral.step(expected, depolarizing_probability=probability)
            error = float(np.linalg.norm(spectral.physical_density(expected) - actual))
            records.append(
                {
                    "size": size,
                    "steps": 3,
                    "probability": probability,
                    "density_error": error,
                    "full_circuit_qubits": original.num_qubits,
                    "passed": error < 1e-10,
                }
            )
    return records


def audit_uploaded_evidence(path):
    if path is None:
        return None
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("publication-evidence/manifest.json"))
        checks = manifest["files"]
        failed = [
            name
            for name, sha in checks.items()
            if hashlib.sha256(archive.read(name)).hexdigest() != sha
        ]
        validation = json.loads(archive.read("validation.json"))
    return {
        "archive_name": Path(path).name,
        "archive_sha256": file_sha(path),
        "listed_files": len(checks),
        "failed_files": failed,
        "source_commit": validation["provenance"]["source_commit"],
        "source_sha256": validation["provenance"]["source_sha256"],
        "software_passed": validation["passed"],
        "physical_hardware_tested": False,
        "scope": "preserved supplied run; not pooled with the new study",
    }


def campaign(output, *, config=None, evidence=None, settings=None):
    settings = {**configuration(config, "recurrent-scaling", DEFAULTS), **(settings or {})}
    output = Path(output).resolve()
    if (output / "result.json").exists():
        raise FileExistsError("Choose a fresh output directory")
    (output / "inputs").mkdir(parents=True, exist_ok=True)
    cases, inputs = [], []
    tasks = [
        (str(output), size, condition, family, settings, False)
        for size in settings["sizes"]
        for condition in settings["conditions"]
        for family in settings["families"]
    ]
    if settings["stress_condition"]:
        tasks += [
            (
                str(output),
                size,
                settings["stress_condition"],
                "controlled_nonsymmetric",
                settings,
                True,
            )
            for size in settings["sizes"]
        ]
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=settings["workers"]) as pool:
        futures = [pool.submit(_static_fixture, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            cases.extend(result["cases"])
            inputs.append(result["input"])
            print(f"Completed {result['input']['id']}", flush=True)
        futures = [
            pool.submit(_dynamic_fixture, (str(output), size, settings))
            for size in settings["sizes"]
        ]
        dynamic, dynamic_inputs = [], []
        for future in as_completed(futures):
            result = future.result()
            dynamic.extend(result["cases"])
            dynamic_inputs.append(result["input"])
            print(f"Completed dynamic order {result['cases'][0]['size']}", flush=True)
    cases.sort(key=lambda row: row["id"])
    dynamic.sort(key=lambda row: (row["size"], row["logical_depolarizing_probability"]))
    crosschecks = circuit_crosschecks()
    audit = audit_uploaded_evidence(evidence)
    main_ideal = [
        row
        for row in cases
        if not row["stress_probe"] and row["logical_depolarizing_probability"] == 0
    ]
    noisy = [row for row in cases if row["logical_depolarizing_probability"] != 0]
    passed = (
        all(row["numerical_integrity_passed"] for row in cases)
        and all(row["accuracy_passed"] for row in main_ideal)
        and all(row["passed"] for row in crosschecks)
        and all(abs(row["final"]["trace"] - 1) < 1e-8 for row in dynamic)
        and (audit is None or not audit["failed_files"])
    )
    payload = {
        "schema_version": 1,
        "passed": bool(passed),
        "settings": settings,
        "provenance": provenance(Path.cwd()),
        "wall_seconds": time.perf_counter() - start,
        "physical_hardware_tested": False,
        "large_native_gate_execution": False,
        "classical_iterate_correction": False,
        "all_main_ideal_accuracy_passed": all(row["accuracy_passed"] for row in main_ideal),
        "all_logical_noisy_accuracy_passed": all(row["accuracy_passed"] for row in noisy),
        "noise_model": "(1-p)E(rho) + p Tr(E(rho))I/P, once per complete quantum update",
        "uploaded_evidence_audit": audit,
        "inputs": sorted(inputs, key=lambda row: row["id"]),
        "dynamic_inputs": dynamic_inputs,
        "cases": cases,
        "dynamic_cases": dynamic,
        "full_circuit_crosschecks": crosschecks,
        "limitations": [
            "Large runs are reduced-density quantum-channel simulations, not QPU/native-gate runs.",
            "Logical update noise is not a hardware gate error or calibrated device noise model.",
            "Dense classical input SVDs remain part of compilation; no speedup is measured.",
            "RHS-dependent encodings do not audit an input-independent inverse operator.",
            "x1 uses simulator coherences; hardware would require additional measurements.",
            "Dynamic matrices rotate two columns at fixed conditioning; inputs are preprogrammed.",
            "Noise, high-conditioning and transient accuracy failures are retained.",
        ],
    }
    save_result(output, payload)
    from .recurrent_scaling_report import generate

    generate(output)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/recurrent-scaling")
    parser.add_argument("--config")
    parser.add_argument("--evidence")
    args = parser.parse_args()
    result = campaign(args.output_dir, config=args.config, evidence=args.evidence)
    print(json.dumps({"passed": result["passed"], "cases": len(result["cases"])}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
