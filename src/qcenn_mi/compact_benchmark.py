"""Numerical and qubit improvement study for compact inverse action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .coherent_large_benchmark import fixture
from .compact import build_compact_inverse


def campaign(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in (10, 100, 1000):
        for family in ("dense_spd", "tridiagonal_spd", "dense_nonsymmetric", "signed_indefinite"):
            matrix = fixture(n, family)
            op = build_compact_inverse(matrix, tolerance=1e-6)
            rng = np.random.default_rng(42)
            rhs_set = [np.eye(1, n, 0).ravel(), np.ones(n) / np.sqrt(n), rng.normal(size=n)]
            results = [op.simulate(b) for b in rhs_set]
            rows.append(
                {
                    "size": n,
                    "family": family,
                    "degree": op.degree,
                    "resources": op.resource_report,
                    "residual_bound": op.residual_bound,
                    "worst_relative_residual": max(r["relative_residual"] for r in results),
                    "minimum_success_probability": min(r["success_probability"] for r in results),
                    "old_degree7_neumann_qubits": (n - 1).bit_length() + 10,
                }
            )
    hard = build_compact_inverse(np.diag(np.linspace(1.0, 100.0, 10)), tolerance=1e-6)
    hard_result = hard.simulate(np.eye(10)[0])
    hard_report = {
        "condition_number": 100,
        "degree": hard.degree,
        "qubits": hard.resource_report["total_qubits"],
        "relative_residual": hard_result["relative_residual"],
        "previous_degree7_neumann_residual": 0.852,
    }
    passed = (
        all(r["worst_relative_residual"] < 1e-6 for r in rows)
        and hard_report["relative_residual"] < 1e-6
    )
    payload = {
        "passed": passed,
        "target_tolerance": 1e-6,
        "cases": rows,
        "hard_case": hard_report,
        "classical_encoding": "dense eigendecomposition/SVD and input-basis preparation",
        "physical_hardware_tested": False,
        "native_noise_in_separate_campaign": True,
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Compact coherent inverse improvements",
        "",
        f"Numerical validation: {'PASS' if passed else 'FAIL'}",
        "",
        "| Size | Family | Degree | Total qubits | Old qubits | Worst residual | Min success |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['size']} | {r['family']} | {r['degree']} | {r['resources']['total_qubits']} | "
            f"{r['old_degree7_neumann_qubits']} | {r['worst_relative_residual']:.3e} | "
            f"{r['minimum_success_probability']:.1%} |"
        )
    lines += [
        "",
        f"Condition-number-100 case: degree {hard.degree}, {hard_report['qubits']} qubits, "
        f"residual {hard_report['relative_residual']:.3e} (previously about 0.852).",
        "",
        "These are ideal structured-statevector results; native noise is reported separately.",
        "Classical spectral/SVD encoding is still required; quantum advantage is not established.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/compact")
    args = parser.parse_args()
    result = campaign(args.output_dir)
    print((Path(args.output_dir) / "summary.md").read_text())
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
