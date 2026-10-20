"""Standalone coherent inverse-action campaign; all corrections occur in circuit."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from .coherent import build_inverse_action
from .experiment_io import save_qpy_atomic


def campaign(output, *, shots=131072, seed=42):
    if not isinstance(shots, int) or shots < 1:
        raise ValueError("shots must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audit_dir = output / "audit"
    audit_dir.mkdir(exist_ok=True)
    backend = AerSimulator(max_parallel_threads=1)
    matrices = {
        "spd2": np.array([[2.0, 0.3], [0.3, 1.5]]),
        "spd4": np.array(
            [[2.0, 0.1, 0, 0.1], [0.1, 1.7, 0.1, 0], [0, 0.1, 1.8, 0.1], [0.1, 0, 0.1, 2.1]]
        ),
    }
    rows = []
    for name, matrix in matrices.items():
        degree_rows = []
        for degree in (1, 3, 7):
            op = build_inverse_action(matrix, degree=degree)
            inverse = np.column_stack([op.ideal(b)["solution"] for b in np.eye(len(matrix))])
            degree_rows.append(
                {
                    "degree": degree,
                    "qubits": op.operator.num_qubits,
                    "residual": float(np.linalg.norm(matrix @ inverse - np.eye(len(matrix)))),
                    "residual_bound": float(
                        np.sqrt(len(matrix)) * op.contraction_bound ** (degree + 1)
                    ),
                }
            )
        op = build_inverse_action(matrix, degree=7)
        ideal_columns, shot_columns, measurements, circuits = [], [], [], []
        for index, rhs in enumerate(np.eye(len(matrix))):
            ideal = op.ideal(rhs)
            measured = op.measure(rhs, shots=shots, seed=seed + index, backend=backend)
            ideal_columns.append(ideal["solution"])
            shot_columns.append(measured["solution"])
            audit_path = audit_dir / f"{name}-column{index}.json"
            audit_path.write_text(json.dumps(measured["audit"], indent=2) + "\n")
            audit = dict(measured["audit"])
            audit.pop("raw_counts")
            measurements.append(
                {
                    "column": index,
                    "ideal_success_probability": ideal["success_probability"],
                    "measured_success_probability": measured["success_probability"],
                    "measured_residual": measured["residual"],
                    "audit_file": str(audit_path.relative_to(output)),
                    "measurement_summary": audit,
                }
            )
            circuits.append(op.circuit(rhs))
        ideal_inverse = np.column_stack(ideal_columns)
        shot_inverse = np.column_stack(shot_columns)
        # Reference solve occurs only after circuit construction and execution.
        reference = np.linalg.solve(matrix, np.eye(len(matrix)))
        rhs = np.arange(1.0, len(matrix) + 1)
        superposition = op.ideal(rhs)
        compiled = transpile(
            circuits,
            basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=1,
            seed_transpiler=seed,
        )
        filename = f"{name}-coherent.qpy"
        save_qpy_atomic(output / filename, circuits)
        rows.append(
            {
                "case": name,
                "matrix": matrix.tolist(),
                "degree_sweep": degree_rows,
                "degree": 7,
                "qubits": op.operator.num_qubits,
                "tau": op.tau,
                "contraction_bound": op.contraction_bound,
                "normalization": op.normalization,
                "ideal_residual": float(
                    np.linalg.norm(matrix @ ideal_inverse - np.eye(len(matrix)))
                ),
                "shot_residual": float(np.linalg.norm(matrix @ shot_inverse - np.eye(len(matrix)))),
                "ideal_relative_inverse_error": float(
                    np.linalg.norm(ideal_inverse - reference) / np.linalg.norm(reference)
                ),
                "shot_relative_inverse_error": float(
                    np.linalg.norm(shot_inverse - reference) / np.linalg.norm(reference)
                ),
                "ideal_inverse": ideal_inverse.tolist(),
                "shot_inverse": shot_inverse.tolist(),
                "superposition_residual": superposition["residual"],
                "superposition_relative_residual": superposition["residual"]
                / float(np.linalg.norm(rhs)),
                "measurements": measurements,
                "qpy_file": filename,
                "native_basis_depths": [qc.depth() for qc in compiled],
                "native_basis_operations": [dict(qc.count_ops()) for qc in compiled],
            }
        )
    passed = all(
        r["ideal_residual"] < 1e-3
        and r["shot_residual"] < 0.15
        and r["superposition_relative_residual"] < 1e-3
        and all(
            a["residual"] > b["residual"]
            for a, b in zip(r["degree_sweep"], r["degree_sweep"][1:], strict=False)
        )
        for r in rows
    )
    payload = {
        "passed": bool(passed),
        "shots": shots,
        "seed": seed,
        "cases": rows,
        "versions": {n: importlib.metadata.version(n) for n in ("qiskit", "qiskit-aer", "numpy")},
        "physical_hardware_tested": False,
        "noise_model": None,
        "classical_work": (
            "Matrix validation, dense square-root block encoding, input and selector preparation, "
            "circuit compilation, final tomography; "
            "no inverse, target fitting or residual correction"
        ),
        "acceptance": {
            "ideal_residual": 1e-3,
            "shot_residual": 0.15,
            "superposition_relative_residual": 1e-3,
        },
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Standalone coherent inverse action",
        "",
        f"Status: {'PASS' if passed else 'FAIL'}",
        "",
        "Coherent Neumann-LCU circuit; no classical fitting or residual correction.",
        "Classical dense block-encoding construction remains required. "
        "Ideal and finite-shot noiseless simulation only.",
        "",
        "| Case | Qubits | Ideal residual | Shot residual | Min success | Max CX |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['qubits']} | {r['ideal_residual']:.3e} | "
            f"{r['shot_residual']:.3e} | "
            f"{min(m['ideal_success_probability'] for m in r['measurements']):.2%} | "
            f"{max(c.get('cx', 0) for c in r['native_basis_operations'])} |"
        )
    lines += [
        "",
        "Degree convergence:",
        "",
        "| Case | Degree | Residual | Bound |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        for sweep in r["degree_sweep"]:
            lines.append(
                f"| {r['case']} | {sweep['degree']} | {sweep['residual']:.3e} | "
                f"{sweep['residual_bound']:.3e} |"
            )
    lines += [
        "",
        "Measurement scales come from ancilla success frequency, not least-squares refitting.",
        "Raw counts and QPY files are generated alongside this report and uploaded by CI.",
        "High ancilla/gate counts and low postselection probability limit hardware practicality.",
        "This is a coherent inverse-action baseline; "
        "quantum advantage and local QCeNN dynamics are not established.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/coherent")
    parser.add_argument("--shots", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = campaign(args.output_dir, shots=args.shots, seed=args.seed)
    print((Path(args.output_dir) / "summary.md").read_text())
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
