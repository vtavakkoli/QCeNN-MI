"""Matched logical and small native-gate noise studies; no calibration claims."""

from __future__ import annotations

import time

import numpy as np

from .dissipative_comparison import stationary_depolarized
from .polynomial_benchmark import fixture, save_input, save_json
from .polynomial_recurrent import accelerate_density_channel, build_polynomial_solver
from .polynomial_renewal import renewal_observables
from .recurrent_density import DensityObserver, build_density_channel


def noise_study(output, settings):
    records = []
    for size in [n for n in settings["sizes"] if n >= 10]:
        for condition in [k for k in settings["conditions"] if k >= 100]:
            print(f"Logical noise N={size} kappa={condition}", flush=True)
            matrix = fixture(size, condition, "controlled_nonsymmetric", settings["seed"])
            rhs = np.random.default_rng(settings["seed"] + size).normal(size=size)
            reference = np.linalg.solve(matrix, rhs)
            base = build_density_channel(matrix, rhs)
            file = f"inputs/noise-n{size}-k{condition}.npz"
            checksum = save_input(
                output / file, matrix=matrix, rhs=rhs, reference_solution=reference
            )
            budget = 4096
            for degree in [d for d in settings["degrees"] if d in [0, 4, 16]]:
                channel = accelerate_density_channel(base, degree)
                observer = DensityObserver.create(channel, reference)
                for primitive in [1e-4, 1e-3]:
                    for convention in ["fixed_macro_update", "polynomial_order_coupled"]:
                        probability = (
                            primitive
                            if convention == "fixed_macro_update"
                            else -np.expm1((degree + 1) * np.log1p(-primitive))
                        )
                        start = time.perf_counter()
                        rows = renewal_observables(
                            channel,
                            budget,
                            [0, 128, 512, 1024, 2048, 4096],
                            "reference",
                            reference,
                            probability,
                        )
                        fixed = stationary_depolarized(channel, probability)
                        check = fixed.copy()
                        channel.step(check, depolarizing_probability=probability)
                        records.append(
                            {
                                "size": size,
                                "condition": condition,
                                "polynomial_degree": degree,
                                "input_file": file,
                                "input_sha256": checksum,
                                "noise_convention": convention,
                                "base_probability": primitive,
                                "logical_probability_per_macro_update": float(probability),
                                "steps": budget,
                                "trajectory": rows,
                                "final": rows[-1],
                                "stationary": observer.read(fixed),
                                "stationary_equation_error": float(np.max(abs(check - fixed))),
                                "simulation_seconds": time.perf_counter() - start,
                                "coupled_model_is_physical_gate_noise": False,
                            }
                        )
                        save_json(output / "noise-progress.json", {"completed": len(records)})
    return records


def gate_noise_study(output, settings):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error

    cases = []
    for condition in [k for k in settings["conditions"] if k in [2, 100]]:
        matrix = fixture(2, condition, "controlled_nonsymmetric", settings["seed"])
        rhs = np.random.default_rng(settings["seed"] + 2).normal(size=2)
        reference = np.linalg.solve(matrix, rhs)
        for degree in [d for d in settings["degrees"] if d in [0, 4, 16]]:
            solver = build_polynomial_solver(matrix, rhs, degree)
            base = build_density_channel(matrix, rhs)
            observer = DensityObserver.create(base, reference)
            for cx_error, reset_error in [(0.0, 0.0), (1e-4, 1e-3), (1e-3, 1e-3)]:
                circuit = solver.circuit(16, workspace="reset")
                noise = NoiseModel()
                if cx_error:
                    noise.add_all_qubit_quantum_error(depolarizing_error(cx_error, 2), "cx")
                if reset_error:
                    noise.add_all_qubit_quantum_error(
                        pauli_error([("X", reset_error), ("I", 1 - reset_error)]), "reset"
                    )
                backend = AerSimulator(
                    method="density_matrix", noise_model=noise, max_parallel_threads=1
                )
                start = time.perf_counter()
                native = transpile(
                    circuit,
                    basis_gates=["rz", "sx", "x", "cx", "reset"],
                    optimization_level=1,
                    seed_transpiler=42,
                )
                physical_depth = native.depth()
                native.save_density_matrix(list(range(solver.data_qubits)), label="data")
                result = backend.run(native, shots=1, seed_simulator=42).result()
                if not result.success:
                    raise RuntimeError(result.status)
                physical = np.asarray(result.data(0)["data"])
                diagnostics = observer.read(base.basis.T @ physical @ base.basis)
                cases.append(
                    {
                        "size": 2,
                        "condition": condition,
                        "polynomial_degree": degree,
                        "steps": 16,
                        "cx_error": cx_error,
                        "reset_error": reset_error,
                        "cx": native.count_ops().get("cx", 0),
                        "depth": physical_depth,
                        "width": native.num_qubits,
                        "reset_count": native.count_ops().get("reset", 0),
                        "final": diagnostics,
                        "simulation_seconds": time.perf_counter() - start,
                        "noise_scope": (
                            "CX depolarization and post-reset X error; other gates/readout ideal"
                        ),
                        "physical_hardware_tested": False,
                    }
                )
    return cases
