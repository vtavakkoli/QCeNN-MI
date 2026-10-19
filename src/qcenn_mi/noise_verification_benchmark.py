"""Repeated synthetic noise with separate solver, verifier, and readout controls."""

from __future__ import annotations

import argparse
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from .coherent_large_benchmark import fixture
from .compact import build_compact_inverse
from .experiment_io import configuration, save_result
from .hardware import NATIVE_BASIS, PROFILES, compile_native, read_echo_counts, wilson

DEFAULTS = {
    "sizes": [2, 4],
    "seeds": [11, 29, 47],
    "shots": 2048,
    "profiles": list(PROFILES),
    "optimization_levels": [0, 2],
    "decompositions": ["input_fused", "input_independent"],
    "noise_scales": [0.5, 1.0],
    "max_density_qubits": 9,
}
CONTROLS = ["noiseless", "readout_only", "verifier_only", "solver_only", "solver_and_verifier"]


def insert_gate_noise(circuit, profile, enabled):
    out = QuantumCircuit(circuit.num_qubits)
    for instruction in circuit.data:
        indices = [circuit.find_bit(bit).index for bit in instruction.qubits]
        out.append(instruction.operation, indices)
        if enabled and instruction.operation.name in {"sx", "x", "cx"}:
            width = len(indices)
            probability = profile["two_qubit" if width == 2 else "one_qubit"]
            if probability:
                out.append(depolarizing_error(probability, width).to_instruction(), indices)
    out.global_phase = circuit.global_phase
    return out


def statistics(counts, data_qubits, width):
    row = read_echo_counts(counts, data_qubits=data_qubits, total_qubits=width)
    row.update(
        {
            "unconditional_success_95ci": wilson(row["accepted"], row["shots"]),
            "effective_good_output_probability": row["returned"] / row["shots"],
            "effective_good_output_95ci": wilson(row["returned"], row["shots"]),
            "too_few_accepted_shots": row["accepted"] < 20,
        }
    )
    return row


def campaign(output, **settings):
    cfg = {**DEFAULTS, **settings}
    if not cfg["seeds"] or cfg["shots"] < 1:
        raise ValueError("Positive shots and at least one seed required")
    rows, resources, skipped = [], [], []
    for n in cfg["sizes"]:
        matrix = fixture(n, "dense_spd")
        rhs = np.ones(n) / np.sqrt(n)
        op = build_compact_inverse(matrix)
        width = op.resource_report["total_qubits"]
        if width > cfg["max_density_qubits"]:
            skipped.append(
                {
                    "size": n,
                    "reason": "density-matrix qubit budget",
                    "required_qubits": width,
                    "statistical_power_resolved": False,
                }
            )
            continue
        # Independent reference AFTER the inverse operation is constructed.
        reference = np.linalg.solve(matrix, rhs)
        data = op.resource_report["data_qubits"]
        padded = np.zeros(1 << data)
        padded[:n] = reference / np.linalg.norm(reference)
        verification = QuantumCircuit(width)
        verification.append(StatePreparation(padded).inverse(), range(data))
        for decomposition in cfg["decompositions"]:
            if decomposition not in {"input_fused", "input_independent"}:
                raise ValueError("Unsupported decomposition")
            action = op.circuit(rhs, fuse_input=decomposition == "input_fused")
            for level in cfg["optimization_levels"]:
                start = time.perf_counter()
                native = compile_native(action, topology="all_to_all", optimization_level=level)
                verifier = transpile(
                    verification,
                    basis_gates=NATIVE_BASIS,
                    optimization_level=level,
                    seed_transpiler=42,
                )
                seconds = time.perf_counter() - start
                vector = Statevector.from_instruction(native).data
                residual = float(
                    np.linalg.norm(matrix @ (vector[:n] * op.normalization).real - rhs)
                )
                key = {"size": n, "decomposition": decomposition, "optimization_level": level}
                operations = dict(native.count_ops())
                resources.append(
                    {
                        **key,
                        "native_depth": native.depth(),
                        "single_qubit_gates": sum(v for k, v in operations.items() if k != "cx"),
                        "two_qubit_gates": operations.get("cx", 0),
                        "synthesis_and_transpile_seconds": seconds,
                        "native_ideal_residual": residual,
                        "passed": residual < 1e-6,
                        "topology": "all-to-all control study; main hardware study uses a line",
                    }
                )
                for profile_name in cfg["profiles"]:
                    if profile_name not in PROFILES:
                        raise ValueError("Unsupported profile")
                    for scale in cfg["noise_scales"]:
                        if not 0 <= scale <= 1:
                            raise ValueError("Noise scale must be in [0,1]")
                        profile = {k: scale * v for k, v in PROFILES[profile_name].items()}
                        for control in CONTROLS:
                            circuit = insert_gate_noise(
                                native, profile, control in {"solver_only", "solver_and_verifier"}
                            )
                            circuit.compose(
                                insert_gate_noise(
                                    verifier,
                                    profile,
                                    control in {"verifier_only", "solver_and_verifier"},
                                ),
                                inplace=True,
                            )
                            circuit.measure_all()
                            readout = (
                                profile["readout"]
                                if control in {"readout_only", "solver_and_verifier"}
                                else 0.0
                            )
                            model = NoiseModel()
                            if readout:
                                model.add_all_qubit_readout_error(
                                    ReadoutError([[1 - readout, readout], [readout, 1 - readout]])
                                )
                            backend = AerSimulator(
                                method="density_matrix", noise_model=model, max_parallel_threads=1
                            )
                            for seed in cfg["seeds"]:
                                start = time.perf_counter()
                                counts = (
                                    backend.run(circuit, shots=cfg["shots"], seed_simulator=seed)
                                    .result()
                                    .get_counts()
                                )
                                row = {
                                    **key,
                                    "profile": profile_name,
                                    "noise_scale": scale,
                                    "parameters": profile,
                                    "control": control,
                                    "seed": seed,
                                    "simulator_seconds": time.perf_counter() - start,
                                    **statistics(counts, data, width),
                                }
                                rows.append(row)
        print(f"Repeated noise controls completed for N={n}", flush=True)
    grouped = []
    keys = sorted(
        {
            (
                r["size"],
                r["decomposition"],
                r["optimization_level"],
                r["profile"],
                r["noise_scale"],
                r["control"],
            )
            for r in rows
        }
    )
    for key in keys:
        group = [
            r
            for r in rows
            if (
                r["size"],
                r["decomposition"],
                r["optimization_level"],
                r["profile"],
                r["noise_scale"],
                r["control"],
            )
            == key
        ]
        counts = {}
        for row in group:
            for bit, count in row["raw_counts"].items():
                counts[bit] = counts.get(bit, 0) + count
        width = build_compact_inverse(fixture(key[0], "dense_spd")).resource_report["total_qubits"]
        grouped.append(
            {
                k: v
                for k, v in zip(
                    [
                        "size",
                        "decomposition",
                        "optimization_level",
                        "profile",
                        "noise_scale",
                        "control",
                    ],
                    key,
                    strict=True,
                )
            }
            | {"replicates": len(group), **statistics(counts, (key[0] - 1).bit_length(), width)}
        )
    return save_result(
        output,
        {
            "passed": bool(resources) and all(r["passed"] for r in resources),
            "configuration": cfg,
            "samples": rows,
            "pooled_results": grouped,
            "resources": resources,
            "skipped": skipped,
            "all_synthetic_quality_gates_passed": all(r["quality_gate"] for r in grouped),
            ("uncertainty"): (
                "95% Wilson binomial intervals; independent seeded shot "
                "replicates; no calibration variation"
            ),
            ("observable"): (
                "conditional reference-echo return, including the selected "
                "verifier/readout errors; not state tomography"
            ),
            ("large_order_power"): (
                "N=100 remains unresolved by the historical 64-shot trajectory "
                "study unless explicitly rerun at an adequate budget"
            ),
            "physical_hardware_tested": False,
            "calibration_derived_noise": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/noise-verification")
    parser.add_argument("--config")
    for name in ("sizes", "seeds", "optimization-levels"):
        parser.add_argument("--" + name, type=int, nargs="+")
    parser.add_argument("--shots", type=int)
    parser.add_argument("--noise-scales", type=float, nargs="+")
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--decompositions", nargs="+")
    args = parser.parse_args()
    cfg = configuration(args.config, "noise-verification", DEFAULTS)
    cfg.update({k: v for k, v in vars(args).items() if k in DEFAULTS and v is not None})
    result = campaign(args.output_dir, **cfg)
    print(
        "Noise study: "
        f"{len(result['samples'])}"
        " seeded controls; software="
        f"{result['passed']}"
        "; quality="
        f"{result['all_synthetic_quality_gates_passed']}"
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
