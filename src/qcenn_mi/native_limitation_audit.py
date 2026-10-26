"""Native-gate limitation audit for Chebyshev reset channels.

Uses actual Qiskit real-Julia QSP synthesis and native routing. Small selected
cases are simulated with *synthetic gate-local* depolarizing noise in Aer.
No calibrated hardware noise, fault-tolerant resource model, or QPU is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


def gate_local_survival(counts, *, p1, p2, readout, readout_bits):
    """Probability the specified stochastic error *events* do not occur.

    This is not an output-fidelity estimate: coherent errors, correlations,
    postselection, leakage, crosstalk and calibration drift are excluded.
    """
    if any(not 0 <= p < 1 for p in (p1, p2, readout)):
        raise ValueError("noise rates must be in [0, 1)")
    single = sum(int(value) for gate, value in counts.items() if gate in {"sx", "x"})
    two = int(counts.get("cx", 0))
    p_gate = math.exp(single * math.log1p(-p1) + two * math.log1p(-p2))
    p_readout = math.exp(readout_bits * math.log1p(-readout))
    return {
        "one_qubit_gate_sites": single,
        "two_qubit_gate_sites": two,
        "readout_bits_assumed": readout_bits,
        "synthetic_no_error_gate_event_probability": p_gate,
        "synthetic_no_error_gate_and_readout_event_probability": p_gate * p_readout,
        "not_a_state_fidelity_bound": True,
    }


def native_density_diagnostic(compiled, *, matrix, rhs, profile):
    """Compute exact finite-gate density output under specified gate errors.

    Readout errors are NOT included in this state fidelity because no
    tomography or measurement is performed; readout cost is a separate bound.
    """
    from qiskit.quantum_info import DensityMatrix, partial_trace
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    from .recurrent_density import build_density_channel
    from .temporal_chebyshev_publication import target_state

    noise = NoiseModel()
    p1, p2 = profile["p1"], profile["p2"]
    if p1:
        noise.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["x", "sx"])
    if p2:
        noise.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    simulated = compiled.copy()
    simulated.save_density_matrix()
    backend = AerSimulator(
        method="density_matrix", noise_model=noise, max_parallel_threads=1
    )
    result = backend.run(simulated, shots=1, seed_simulator=420).result()
    density = DensityMatrix(result.data(0)["density_matrix"])
    n_data = len(matrix).bit_length()
    layout = (
        list(range(compiled.num_qubits)) if compiled.layout is None
        else compiled.layout.final_index_layout()
    )
    qubits = [layout[i] for i in range(n_data)]
    trace_out = [i for i in range(compiled.num_qubits) if i not in qubits]
    data = partial_trace(density, trace_out).data
    _, target = target_state(matrix, rhs, build_density_channel(matrix, rhs))
    # target coordinates are embedded into the Qiskit data-register width.
    padded = np.zeros(1 << n_data)
    padded[: len(target)] = target
    fidelity = float(np.clip(np.vdot(padded, data @ padded).real, 0, 1))
    return {
        "augmented_target_fidelity_before_measurement": fidelity,
        "augmented_target_infidelity_before_measurement": 1 - fidelity,
        "actual_synthetic_aer_gate_local_noise": True,
        "readout_error_simulated": False,
        "physical_hardware_tested": False,
        "noise_profile": profile,
    }


def native_case(size, condition, *, include_noise):
    from .chebyshev_recurrent import (
        chebyshev_schedule_circuit,
        qsp_filter_circuit,
        synthesize_filter,
    )
    from .hardware import compile_native
    from .recurrent_quantum import build_recurrent_solver

    matrix = np.diag(np.geomspace(1.0, 1.0 / condition, size))
    rhs = np.arange(1, size + 1, dtype=float)
    rhs /= np.linalg.norm(rhs)
    start = time.perf_counter()
    base = build_recurrent_solver(matrix, rhs)
    encoding_seconds = time.perf_counter() - start
    gap = base.singular_ratio**2 / (1 + base.singular_ratio**2)
    start = time.perf_counter()
    phase = synthesize_filter(gap, max_degree=64)
    phase_seconds = time.perf_counter() - start
    start = time.perf_counter()
    signal = qsp_filter_circuit(base, phase)
    circuit = chebyshev_schedule_circuit([(base, phase)], workspace="reset")
    native = compile_native(circuit, topology="line", optimization_level=1)
    compile_seconds = time.perf_counter() - start
    counts = {str(k): int(v) for k, v in native.count_ops().items()}
    row = {
        "size": size, "condition": condition,
        "status": "compiled_small_complete_circuit",
        "qubits": native.num_qubits,
        "data_qubits": base.data_qubits,
        "filter_degree": phase.degree,
        "primitive_calls_per_macro": phase.queries_per_update,
        "signal_circuit_residual_calls": signal.metadata["residual_unitary_calls"],
        "native_depth_line_topology": native.depth(),
        "native_counts_line_topology": counts,
        "native_cx": counts.get("cx", 0),
        "native_resets": counts.get("reset", 0),
        "data_reference_initial_x_gates": size.bit_count(),
        "environment_reference_initial_x_gates_per_macro": size.bit_count(),
        "data_reference_preparations_resident_unmeasured_12_snapshots": 1,
        "data_reference_preparations_fresh_12_snapshots": 12,
        "classical_dense_svd_and_dilation_seconds_local": encoding_seconds,
        "phase_synthesis_seconds_local": phase_seconds,
        "native_transpile_seconds_local": compile_seconds,
        "synthesis_timing_nonportable": True,
        "dense_input_preprocessing_cost": "O(N^3) SVD; may expose solution-space data",
        "readout_settings_for_all_N_coherences": size + 1,
        "readout_executions_not_in_oracle_accounting": True,
        "postselection_amplification_implemented": False,
        "native_noise_calibrated_to_device": False,
        "physical_hardware_tested": False,
    }
    profiles = {
        "optimistic_synthetic": {"p1": 1e-5, "p2": 1e-4, "readout": 0.005},
        "stress_synthetic": {"p1": 1e-4, "p2": 1e-3, "readout": 0.02},
    }
    model = {}
    for name, profile in profiles.items():
        row_model = gate_local_survival(
            counts, p1=profile["p1"], p2=profile["p2"],
            readout=profile["readout"], readout_bits=size + 1,
        )
        if include_noise:
            row_model["aer_density_output"] = native_density_diagnostic(
                native, matrix=matrix, rhs=rhs, profile=profile
            )
        else:
            row_model["aer_density_output"] = {
                "status": "omitted_by_declared_small_circuit_budget"
            }
        model[name] = row_model
    row["synthetic_noise_profiles"] = model
    return row


def run_audit(output_dir, *, smoke=False):
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Native audit requires fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    cases = [(2, 2)] if smoke else [(2, 2), (2, 10), (4, 2), (4, 10)]
    records = []
    for size, condition in cases:
        # Limit full density-matrix noise simulation to <= 7 qubits.
        row = native_case(size, condition, include_noise=(size == 2))
        records.append(row)
        print(
            f"Native limitation audit: N={size}, kappa={condition}, "
            f"CX={row['native_cx']}, depth={row['native_depth_line_topology']}",
            flush=True,
        )
    evidence = {
        "schema_version": 1, "profile": "smoke" if smoke else "full",
        "records": records, "real_qsp_phase_synthesis_used": True,
        "native_gates_compiled": True, "noisy_aer_gate_simulation_on_small_cases": True,
        "noise_rates_synthetic_not_calibrated": True,
        "complete_high_N_native_qsp_implemented": False,
        "hardware_tested": False,
        "hardware_advantage_claimed": False,
    }
    (out / "result.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Native QSP preparation and gate-local noise limitations",
        "",
        "Small dense real-Julia circuits transpiled to [rz,sx,x,cx] on a line.",
        "The rates are hypothetical stochastic local gate depolarization;",
        "they are NOT backend calibration or measured hardware results.",
        "Readout is separately charged: no readout-error tomography is performed.",
        "",
        "| N | Condition | QSP degree | Width | CX | Native depth |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['size']} | {row['condition']} | {row['filter_degree']} | "
            f"{row['qubits']} | {row['native_cx']} | "
            f"{row['native_depth_line_topology']} |"
        )
    lines += [
        "", "Dense spectral synthesis is O(N^3) classical preprocessing.",
        "State reuse does not eliminate the need to prepare multiple copies",
        "for destructive measurements. No scalable native QSP or full-vector",
        "readout performance has been demonstrated.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    checksums = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir()) if p.is_file()
    }
    (out / "MANIFEST.json").write_text(
        json.dumps({"sha256": checksums}, indent=2) + "\n", encoding="utf-8"
    )
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_audit(args.output_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
