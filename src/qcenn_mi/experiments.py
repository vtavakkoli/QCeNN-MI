"""Reproducible A2-A4 simulator campaign and portable QPY circuit export."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from qiskit import qpy, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from .coupled import solve_coupled
from .polynomial import quantum_polynomial_correct
from .tracking import InverseTracker, polynomial_correct


def noise_backend(probability):
    if not 0 <= probability <= 1:
        raise ValueError("noise probability must be between zero and one")
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(probability, 1), ["ry", "h", "sdg"])
    model.add_all_qubit_quantum_error(depolarizing_error(probability, 2), ["cx"])
    return AerSimulator(noise_model=model, basis_gates=["ry", "rz", "h", "sdg", "cx"])


def campaign(output, *, depths=(1, 2, 4), shots=8192, seed=42, noise=0.002, steps=5):
    output = Path(output)
    if not depths or any(d < 1 for d in depths) or steps < 2:
        raise ValueError("Positive depths and at least two tracking steps required")
    output.mkdir(parents=True, exist_ok=True)
    matrices = {
        "spd2": np.array([[2.0, 0.5], [0.5, 1.5]]),
        "nonsymmetric4": np.array(
            [[2.0, -0.2, 0.1, 0], [0.3, 1.8, 0.2, 0.1], [0, -0.1, 1.6, 0.2], [0.1, 0, 0.3, 2.1]]
        ),
        "signed2": np.array([[-2.0, 0.2], [0.3, 1.5]]),
    }
    rows, exports = [], []
    noisy = noise_backend(noise)
    for name, m in matrices.items():
        for depth in depths:
            solved = solve_coupled(m, depth=depth, seed=seed)
            measured, diagnostics = solved.measure(shots=shots, seed=seed)
            perturbed, noisy_diagnostics = solved.measure(backend=noisy, shots=shots, seed=seed)
            corrected = polynomial_correct(m, perturbed)
            circuits = solved.circuits()
            compiled = transpile(circuits, noisy, optimization_level=1, seed_transpiler=seed)
            filename = f"{name}-depth{depth}.qpy"
            with (output / filename).open("wb") as handle:
                qpy.dump(circuits, handle)
            exports.append(filename)

            def residual(x, matrix=m):
                return float(np.linalg.norm(matrix @ x - np.eye(len(matrix))))

            rows.append(
                {
                    "case": name,
                    "matrix": m.tolist(),
                    "depth": depth,
                    "condition_number": float(np.linalg.cond(m)),
                    "ideal_residual": solved.residual,
                    "converged": solved.success,
                    "shot_residual": residual(measured),
                    "noisy_residual": residual(perturbed),
                    "classically_corrected_noisy_residual": residual(corrected),
                    "evaluations": solved.evaluations,
                    "measurement_settings_per_column": 3 ** circuits[0].num_qubits,
                    "compiled_depths": [qc.depth() for qc in compiled],
                    "compiled_operations": [dict(qc.count_ops()) for qc in compiled],
                    "tomography": diagnostics,
                    "noisy_tomography": noisy_diagnostics,
                    "rotations": solved.rotations.tolist(),
                    "interactions": solved.interactions.tolist(),
                    "ideal_estimate": solved.estimate.tolist(),
                    "shot_estimate": measured.tolist(),
                    "noisy_estimate": perturbed.tolist(),
                }
            )
    polynomial_rows = []
    for name, m in matrices.items():
        initial = m.T / (np.linalg.norm(m, 1) * np.linalg.norm(m, np.inf))
        polynomial = quantum_polynomial_correct(m, initial, order=3)
        measured, diagnostics = polynomial.measure(m, shots=shots, seed=seed)
        expected = polynomial_correct(m, initial, order=3)
        filename = f"{name}-polynomial.qpy"
        with (output / filename).open("wb") as handle:
            qpy.dump(polynomial.circuits, handle)
        exports.append(filename)
        polynomial_rows.append(
            {
                "case": name,
                "order": 3,
                "initial_estimate": initial.tolist(),
                "ideal_residual": polynomial.residual,
                "initial_residual": float(np.linalg.norm(m @ initial - np.eye(len(m)))),
                "classical_agreement_error": float(np.linalg.norm(polynomial.estimate - expected)),
                "shot_residual": float(np.linalg.norm(m @ measured - np.eye(len(m)))),
                "success_probabilities": polynomial.success_probabilities,
                "normalization": polynomial.normalization,
                "measurements": diagnostics,
            }
        )
    tracking = []
    for name, base in matrices.items():
        tracker = InverseTracker(depth=2)
        for time in np.linspace(0, 0.4, steps):
            m = base + 0.05 * np.sin(time) * np.eye(len(base))
            record = tracker.update(m, time=time)
            cold = solve_coupled(m, depth=2, seed=seed)
            record.update(
                case=name,
                cold_evaluations=cold.evaluations,
                cold_residual=cold.residual,
                cold_converged=cold.success,
            )
            tracking.append(record)
    # These are fixed campaign gates, not universal claims for arbitrary matrices/noise.
    passed = all(
        r["converged"]
        and r["ideal_residual"] < 1e-7
        and r["shot_residual"] < 0.15
        and r["noisy_residual"] < 0.25
        and r["classically_corrected_noisy_residual"] <= r["noisy_residual"]
        for r in rows
    )
    passed = passed and all(
        r["corrected_residual"] < 1e-8 and r["cold_converged"] for r in tracking
    )
    passed = passed and all(
        r["classical_agreement_error"] < 1e-10
        and r["ideal_residual"] <= r["initial_residual"]
        and r["shot_residual"] < r["initial_residual"]
        for r in polynomial_rows
    )
    payload = {
        "schema_version": 1,
        "passed": bool(passed),
        "seed": seed,
        "shots": shots,
        "noise": {
            "kind": "synthetic depolarizing",
            "probability": noise,
            "calibration_derived": False,
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "qiskit", "qiskit-aer")
        },
        "acceptance": {"ideal": 1e-7, "shots": 0.15, "noisy": 0.25, "tracking": 1e-8},
        "static": rows,
        "tracking": tracking,
        "quantum_polynomial": polynomial_rows,
        "qpy_exports": exports,
        "physical_hardware_tested": False,
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = [
        "# A2–A4 simulator campaign",
        "",
        f"Status: {'PASS' if passed else 'FAIL'}",
        "",
        "Synthetic noise; no physical hardware execution. Safeguarded corrections are classical.",
        "",
        "| Case | Layers | Ideal residual | Shot residual | Noisy residual | Corrected |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['depth']} | {r['ideal_residual']:.3e} | "
            f"{r['shot_residual']:.3e} | {r['noisy_residual']:.3e} | "
            f"{r['classically_corrected_noisy_residual']:.3e} |"
        )
    lines += [
        "",
        "## A4 dense quantum polynomial dilation",
        "",
        "| Case | Initial residual | Ideal operator residual | Postselected shot residual |",
        "|---|---:|---:|---:|",
    ]
    for r in polynomial_rows:
        lines.append(
            f"| {r['case']} | {r['initial_residual']:.3e} | "
            f"{r['ideal_residual']:.3e} | {r['shot_residual']:.3e} |"
        )
    lines += [
        "",
        f"Tracking samples: {len(tracking)}; maximum corrected residual: "
        f"{max(r['corrected_residual'] for r in tracking):.3e}.",
        "",
        "QPY files contain state preparation circuits. Add the X/Y/Z measurement settings",
        "using readout_direction(), transpile for the selected backend and record its calibration.",
        "Full tomography uses 3**qubits settings per column and is not a scalable readout.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/a2-a4")
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", type=float, default=0.002)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    result = campaign(
        args.output_dir,
        depths=args.depths,
        shots=args.shots,
        seed=args.seed,
        noise=args.noise,
        steps=args.steps,
    )
    print((Path(args.output_dir) / "summary.md").read_text())
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
