"""Small complete-QSP phase accuracy and logical noise-bias audit.

Requires Qiskit and pyqsp; does not substitute logical noise for native-gate
noise or a calibrated quantum backend. Stores both successes and failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _reset_step(operator, state, reference, noise_probability=0):
    no_event = operator @ state @ operator.conj().T
    failure = float(np.trace(state - no_event).real)
    result = no_event + failure * reference
    if noise_probability:
        dimension = len(result)
        result = (1 - noise_probability) * result + (
            noise_probability * np.trace(result) / dimension * np.eye(dimension))
    return (result + result.conj().T) / 2


def _coherence_residual(density, matrix, rhs, gamma):
    n = len(matrix)
    p_ref = float(density[n, n].real)
    if p_ref <= 1e-14:
        return None
    estimate = (np.linalg.norm(rhs) / (gamma * np.linalg.norm(matrix, 2))) * (
        density[:n, n] / p_ref)
    return float(np.linalg.norm(matrix @ estimate.real - rhs) / np.linalg.norm(rhs))


def evaluate(condition, *, steps=5, per_oracle_noise=0.001):
    from qiskit.quantum_info import Operator

    from .chebyshev_recurrent import (
        chebyshev_response,
        qsp_filter_circuit,
        synthesize_filter,
    )
    from .recurrent_quantum import build_recurrent_solver

    matrix = np.diag([1.0, 1 / condition])
    rhs = np.array([0.8, 0.6])
    base = build_recurrent_solver(matrix, rhs)
    gap = base.singular_ratio**2 / (1 + base.singular_ratio**2)
    specification = synthesize_filter(gap)
    circuit = qsp_filter_circuit(base, specification)
    full = Operator(circuit).data
    d = 1 << base.data_qubits
    block = full[:d, :d]
    g = base.residual_operator.T @ base.residual_operator
    eigenvalues, basis = np.linalg.eigh(g)
    exact = (basis * chebyshev_response(
        eigenvalues, gap, specification.degree)) @ basis.T
    attenuated = specification.attenuation * exact
    phase_error = float(np.max(np.abs(block - attenuated)))
    r = np.zeros(d)
    r[len(matrix)] = 1
    reference = np.outer(r, r)
    target = basis[:, np.argmin(eigenvalues)]
    stationary = np.outer(target, target)
    kernel_leakage = float(np.max(np.abs(
        _reset_step(block, stationary, reference) - stationary)))
    traces = []
    trajectories = []
    per_update_noise = -np.expm1(
        specification.queries_per_update * np.log1p(-per_oracle_noise))
    for method, operator, noise in [
        ("exact_ideal", exact, 0.0),
        ("phase_synthesized", block, 0.0),
        ("phase_synthesized_logical_noise", block, per_update_noise),
    ]:
        rho = reference.copy().astype(complex)
        for _ in range(steps):
            rho = _reset_step(operator, rho, reference, noise)
        traces.append(float(abs(np.trace(rho) - 1)))
        trajectories.append({
            "method": method,
            "target_infidelity": float(max(0, 1 - np.vdot(target, rho @ target).real)),
            "relative_residual": _coherence_residual(
                rho, matrix, rhs, base.singular_ratio),
            "minimum_eigenvalue": float(np.linalg.eigvalsh(rho).min()),
        })
    by_name = {row["method"]: row for row in trajectories}
    return {
        "condition": condition,
        "degree": specification.degree,
        "calls_per_macro": specification.queries_per_update,
        "steps": steps,
        "coefficient_l1_error_estimate": specification.coefficient_error_estimate,
        "coefficient_estimate_interval_certified": False,
        "attenuation": specification.attenuation,
        "phase_iterations": specification.iterations,
        "maximum_qsp_block_error": phase_error,
        "stationary_kernel_leakage_max_entry": kernel_leakage,
        "unitary_error": float(np.linalg.norm(full.conj().T @ full - np.eye(len(full)), 2)),
        "logical_noise_per_oracle_call": per_oracle_noise,
        "logical_noise_per_macro": float(per_update_noise),
        "noise_model": "depolarization after each macro; rate coupled to query count",
        "native_gate_noise_calibrated": False,
        "maximum_trace_error": max(traces),
        "phase_induced_infidelity_bias": (
            by_name["phase_synthesized"]["target_infidelity"] -
            by_name["exact_ideal"]["target_infidelity"]),
        "noise_induced_infidelity_bias": (
            by_name["phase_synthesized_logical_noise"]["target_infidelity"] -
            by_name["phase_synthesized"]["target_infidelity"]),
        "trajectories": trajectories,
        "numerical_validity_passed": bool(phase_error < 1e-8 and
                                         max(traces) < 1e-9 and
                                         all(row["minimum_eigenvalue"] > -1e-9
                                             for row in trajectories)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--conditions", nargs="+", type=float, default=[2, 10])
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Phase audit output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for condition in args.conditions:
        try:
            records.append({"status": "complete", **evaluate(condition)})
        except (MemoryError, ArithmeticError, ValueError) as error:
            records.append({"status": "failure", "condition": condition,
                            "reason": str(error)})
    data = {"level": "complete signal-block unitary plus logical-channel model",
            "hardware_tested": False, "records": records,
            "passed": all(r.get("numerical_validity_passed", False) for r in records)}
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    (out / "result.json").write_text(payload)
    (out / "SHA256SUMS.json").write_text(json.dumps({
        "result.json": hashlib.sha256(payload.encode()).hexdigest()}, indent=2) + "\n")
    if not data["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
