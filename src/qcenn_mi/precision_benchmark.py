"""1000-dimensional precision and conditioning study of coherent inverse action.

The full-operator audit reads simulator amplitudes, not quantum measurements.
Classical references and residuals are reporting-only, never solver inputs.
"""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .coherent_large_benchmark import fixture
from .compact import build_compact_inverse


def controlled_fixture(size, family, condition, *, seed=42):
    """Dense real matrices with prescribed nonzero singular values.

    Positive definiteness of M M.T is obtained without forming normal equations,
    which would unnecessarily square the condition number of the solve.
    """
    if size < 2 or not np.isfinite(condition) or condition < 1:
        raise ValueError("size >= 2 and finite condition >= 1 required")
    rng = np.random.default_rng(seed)
    left, _ = np.linalg.qr(rng.normal(size=(size, size)))
    values = np.geomspace(1.0, condition, size)
    if family == "controlled_spd":
        return (left * values) @ left.T
    if family == "controlled_indefinite":
        values[::2] *= -1
        return (left * values) @ left.T
    if family == "controlled_nonsymmetric":
        right, _ = np.linalg.qr(rng.normal(size=(size, size)))
        return (left * values) @ right.T
    raise ValueError(f"Unknown controlled family: {family}")


def audit_operator(operation):
    """Reconstruct the simulated success operator for validation only.

    SELECT is diagonal in the encoded data basis. A balanced input excites every
    encoded coordinate; dividing its simulated success amplitudes by the known
    input amplitudes yields every diagonal response. No 1/sigma table is used.
    Dense residuals then check all right-hand sides, including basis-change error.
    This is a simulator-only audit, not an efficient measured inverse readout.
    """
    encoding = operation.encoding
    size = len(encoding.matrix)
    rhs = encoding.input_basis @ (np.ones(size) / np.sqrt(size))
    result = operation.simulate(rhs, return_success=True)
    input_amplitudes = encoding.input_basis.T @ (rhs / np.linalg.norm(rhs))
    response = (
        result["success_spectral_amplitudes"][:size] / input_amplitudes
    ).real * operation.normalization
    reconstructed = (encoding.output_basis * response) @ encoding.input_basis.T
    residual = encoding.matrix @ reconstructed - np.eye(size)
    # Independent reference is constructed only AFTER the quantum simulation.
    reference = np.linalg.solve(encoding.matrix, np.eye(size))
    return {
        "operator_relative_residual_2": float(np.linalg.norm(residual, ord=2)),
        "inverse_residual_frobenius_normalized": float(np.linalg.norm(residual) / np.sqrt(size)),
        "relative_inverse_error_frobenius": float(
            np.linalg.norm(reconstructed - reference) / np.linalg.norm(reference)
        ),
        "encoded_spectral_residual_max": float(
            np.max(np.abs(encoding.singular_values * response - 1))
        ),
        "reference_operator_residual_2": float(
            np.linalg.norm(encoding.matrix @ reference - np.eye(size), ord=2)
        ),
        "minimum_success_probability_all_inputs": float(
            np.min(np.square(response / operation.normalization))
        ),
        "state_norm_error": abs(result["state_norm"] - 1),
        "padding_success_weight": result["padding_success_weight"],
        "readout": "simulator amplitude access, no physical measurement or tomography",
    }


def campaign(output, *, size=1000, tolerance=1e-12, residual_limit=1e-10, max_degree=1023):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    configurations = [
        (family, None)
        for family in ("dense_spd", "tridiagonal_spd", "dense_nonsymmetric", "signed_indefinite")
    ] + [
        (family, condition)
        for condition in (10, 100, 1000)
        for family in ("controlled_spd", "controlled_nonsymmetric", "controlled_indefinite")
    ]
    for family, condition in configurations:
        matrix = (
            fixture(size, family)
            if condition is None
            else controlled_fixture(size, family, condition)
        )
        operation = build_compact_inverse(matrix, tolerance=tolerance, max_degree=max_degree)
        rng = np.random.default_rng(84)
        rhs_set = [np.eye(1, size, 0).ravel(), np.ones(size), rng.normal(size=size)]
        samples = [operation.simulate(rhs) for rhs in rhs_set]
        audit = audit_operator(operation)
        baseline = build_compact_inverse(matrix, tolerance=1e-6, max_degree=max_degree)
        baseline_residual = max(baseline.simulate(rhs)["relative_residual"] for rhs in rhs_set)
        row = {
            "size": size,
            "family": family,
            "requested_condition_number": condition,
            "condition_number": operation.encoding.condition_number,
            "degree": operation.degree,
            "truncation_residual_bound": operation.residual_bound,
            "resources": operation.resource_report,
            "worst_sampled_relative_residual": max(r["relative_residual"] for r in samples),
            "baseline_target_tolerance": 1e-6,
            "baseline_worst_sampled_relative_residual": baseline_residual,
            "sampled_residual_improvement_factor": baseline_residual
            / max(r["relative_residual"] for r in samples),
            "minimum_sampled_success_probability": min(r["success_probability"] for r in samples),
            **audit,
        }
        row["passed"] = bool(
            row["operator_relative_residual_2"] <= residual_limit
            and row["worst_sampled_relative_residual"] <= residual_limit
            and row["relative_inverse_error_frobenius"] <= residual_limit
            and row["state_norm_error"] < 1e-10
            and row["padding_success_weight"] < 1e-24
            and row["truncation_residual_bound"] <= tolerance
        )
        cases.append(row)
        print(
            f"{size} {family} kappa={row['condition_number']:.3g} d={operation.degree}: "
            f"operator residual={row['operator_relative_residual_2']:.3e}",
            flush=True,
        )
    # A positive-definite input outside the configured budget must fail honestly.
    try:
        build_compact_inverse(
            np.diag(np.geomspace(1.0, 10000.0, size)),
            tolerance=tolerance,
            max_degree=max_degree,
        )
        limited = {"size": size, "condition_number": 10000, "explicitly_rejected": False}
    except ValueError as error:
        limited = {
            "size": size,
            "condition_number": 10000,
            "explicitly_rejected": "budget" in str(error),
            "reason": str(error),
        }
    payload = {
        "passed": all(row["passed"] for row in cases) and limited["explicitly_rejected"],
        "size": size,
        "polynomial_tolerance": tolerance,
        "numerical_residual_limit": residual_limit,
        "max_degree": max_degree,
        "environment": {
            "python": platform.python_version(),
            **{package: version(package) for package in ("numpy", "scipy", "qiskit")},
        },
        "seeds": {"controlled_fixture": 42, "sampled_rhs": 84},
        "cases": cases,
        "resource_limited_positive_definite_case": limited,
        "method": "ideal structured coherent statevector; lazy selector/output basis changes",
        "classical_encoding": "dense eigendecomposition/SVD, O(N^3)",
        "classical_reference": "reporting only, after quantum simulation",
        "physical_hardware_tested": False,
        "native_gate_simulation_tested_at_this_size": False,
        "all_matrices_proved_by_this_experiment": False,
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = [
        f"# {size}×{size} coherent inverse precision study",
        "",
        f"Numerical validation: {'PASS' if payload['passed'] else 'FAIL'}",
        "",
        f"Polynomial truncation target {tolerance:.0e}; numerical operator-residual gate "
        f"{residual_limit:.0e}. These are separate: floating-point encoding and gate "
        "simulation errors are not covered by the truncation bound.",
        "",
        "| Family | Condition | Degree | Qubits | Operator residual 2-norm | "
        "Relative inverse error F | Min success (all inputs) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cases:
        lines.append(
            f"| {row['family']} | {row['condition_number']:.3g} | {row['degree']} | "
            f"{row['resources']['total_qubits']} | {row['operator_relative_residual_2']:.3e} | "
            f"{row['relative_inverse_error_frobenius']:.3e} | "
            f"{row['minimum_success_probability_all_inputs']:.3e} |"
        )
    lines += [
        "",
        "On the four earlier fixture families, the same-input worst residual improves "
        f"by {min(r['sampled_residual_improvement_factor'] for r in cases[:4]):.2e}–"
        f"{max(r['sampled_residual_improvement_factor'] for r in cases[:4]):.2e} "
        "relative to the 1e-6 polynomial target.",
        "",
        "The full-operator audit reconstructs the simulated success operator and checks "
        "||M X - I||_2, which bounds ||M X b - b||_2 / ||b||_2 for every input b of "
        "each tested matrix. It uses exact simulator amplitudes, not measured inverse readout.",
        "",
        "M M.T > 0 guarantees invertibility for a real square M, but neither a uniform "
        f"degree budget nor uniform success probability. The {size}×{size} "
        "condition-10000 input is "
        f"explicitly resource-limited: {limited.get('reason', 'unexpectedly accepted')}",
        "",
        "Classical dense spectral/SVD encoding remains required. This is not native noisy "
        "1000-dimensional execution, a hardware result, or evidence of quantum speed-up.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/precision-1000")
    args = parser.parse_args()
    result = campaign(args.output_dir)
    print((Path(args.output_dir) / "summary.md").read_text())
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
