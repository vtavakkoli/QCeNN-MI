"""Independent full-state and two-sided audits of an unfused quantum operator."""

from __future__ import annotations

import argparse

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

from .compact import build_compact_inverse
from .experiment_io import configuration, save_result
from .hardware import NATIVE_BASIS, compile_native, logical_statevector
from .precision_benchmark import controlled_fixture

DEFAULTS = {
    "sizes": [2, 4, 10],
    "seed": 1729,
    "tolerance": 1e-8,
    "families": ["nonnormal", "signed_indefinite", "repeated_singular_values"],
}


def matrix_fixture(size, family, seed):
    if family == "nonnormal":
        return controlled_fixture(size, "controlled_nonsymmetric", 4, seed=seed)
    if family == "signed_indefinite":
        return controlled_fixture(size, "controlled_indefinite", 4, seed=seed)
    if family == "repeated_singular_values":
        rng = np.random.default_rng(seed)
        u, _ = np.linalg.qr(rng.normal(size=(size, size)))
        v, _ = np.linalg.qr(rng.normal(size=(size, size)))
        values = np.repeat([1.0, 4.0], (size + 1) // 2)[:size]
        return (u * values) @ v.T
    raise ValueError("Unsupported operator-audit family")


def audit_case(size, family, *, seed=1729, tolerance=1e-8):
    matrix = matrix_fixture(size, family, seed)
    op = build_compact_inverse(matrix, tolerance=tolerance)
    native = compile_native(op.operator_circuit(), topology="line", optimization_level=1)
    optimized = compile_native(op.operator_circuit(), topology="line", optimization_level=2)
    data = op.resource_report["data_qubits"]
    padded_size = 1 << data
    basis = np.eye(padded_size)
    basis[:size, :size] = op.encoding.output_basis
    inputs = list(np.eye(size)) + list(np.random.default_rng(seed + 100).normal(size=(3, size)))
    vectors, samples, optimization_comparison = [], [], []
    for index, rhs in enumerate(inputs):
        circuit = QuantumCircuit(native.num_qubits)
        padded = np.zeros(padded_size)
        padded[:size] = rhs / np.linalg.norm(rhs)
        preparation = QuantumCircuit(data)
        preparation.append(StatePreparation(padded), range(data))
        preparation = transpile(preparation, basis_gates=NATIVE_BASIS, optimization_level=1)
        circuit.compose(preparation, range(data), inplace=True)
        circuit.compose(native, inplace=True)
        observed = logical_statevector(
            native, np.asarray(Statevector.from_instruction(circuit).data)
        )
        structured = op.simulate(rhs, return_state=True)
        expected = (structured["spectral_statevector"] @ basis.T).ravel()
        if index >= size:
            candidate = QuantumCircuit(optimized.num_qubits)
            candidate.compose(preparation, range(data), inplace=True)
            candidate.compose(optimized, inplace=True)
            candidate_state = logical_statevector(
                optimized, np.asarray(Statevector.from_instruction(candidate).data)
            )
            candidate_error = float(np.linalg.norm(candidate_state - expected))
            optimization_comparison.append(
                {
                    "input_index": index,
                    "optimization_level": 2,
                    "full_state_error": candidate_error,
                    "meets_strict_full_state_target": candidate_error < 1e-8,
                    "native_depth": optimized.depth(),
                    "native_cx": int(optimized.count_ops().get("cx", 0)),
                }
            )
        full_error = float(np.linalg.norm(observed - expected))
        solution = (observed[:size] * op.normalization * np.linalg.norm(rhs)).real
        samples.append(
            {
                "input_index": index,
                "unseen_rhs": index >= size,
                "full_state_error_including_failure_sectors": full_error,
                "state_norm_error": abs(float(np.linalg.norm(observed) ** 2) - 1),
                "success_probability": float(np.linalg.norm(observed[:padded_size]) ** 2),
                "padding_success_weight": float(np.linalg.norm(observed[size:padded_size]) ** 2),
                "relative_residual": float(
                    np.linalg.norm(matrix @ solution - rhs) / np.linalg.norm(rhs)
                ),
            }
        )
        if index < size:
            vectors.append(solution)
    reconstructed = np.column_stack(vectors)
    left = float(np.linalg.norm(matrix @ reconstructed - np.eye(size), ord=2))
    right = float(np.linalg.norm(reconstructed @ matrix - np.eye(size), ord=2))
    return {
        "size": size,
        "family": family,
        "seed": seed,
        "degree": op.degree,
        "left_operator_residual": left,
        "right_operator_residual": right,
        "analytic_residual_bound": op.residual_bound,
        "samples": samples,
        "input_independent_operator": True,
        "strict_full_state_optimization_level": 1,
        "optimization_comparison": optimization_comparison,
        "compiler_precision_note": (
            "Level 2 can drop tiny endpoint rotations; strict audit uses level 1 "
            "and retains failed optimization comparisons."
        ),
        "fused_input": False,
        "routing_permutation_restored": True,
        "native_depth": native.depth(),
        "native_cx": int(native.count_ops().get("cx", 0)),
        "passed": max(left, right) < tolerance + 1e-8
        and all(
            s["full_state_error_including_failure_sectors"] < 1e-8
            and s["state_norm_error"] < 1e-8
            and s["padding_success_weight"] < 1e-16
            for s in samples
        ),
    }


def campaign(output, **settings):
    cfg = {**DEFAULTS, **settings}
    rows = [
        audit_case(n, family, seed=cfg["seed"], tolerance=cfg["tolerance"])
        for n in cfg["sizes"]
        for family in cfg["families"]
    ]
    return save_result(
        output,
        {
            "passed": all(r["passed"] for r in rows),
            "configuration": cfg,
            "cases": rows,
            "physical_hardware_tested": False,
            ("evidence"): (
                "fully uncomputed routed native Qiskit statevector; every "
                "success and non-success amplitude"
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/operator-audit")
    parser.add_argument("--config")
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg = configuration(args.config, "operator-audit", DEFAULTS)
    cfg.update({k: v for k, v in vars(args).items() if k in DEFAULTS and v is not None})
    result = campaign(args.output_dir, **cfg)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
