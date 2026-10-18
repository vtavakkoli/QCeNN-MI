"""Requested-size coherent inverse-action study with explicit verification levels."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from scipy.linalg import solve

from .coherent_large import build_large_inverse_action


def fixture(size, family):
    diagonal = np.diag(np.linspace(1.5, 2.5, size))
    if family == "dense_spd":
        random = np.random.default_rng(42 + size).normal(0, 0.02 / np.sqrt(size), (size, size))
        return diagonal + (random + random.T) / 2
    if family == "tridiagonal_spd":
        return diagonal + 0.08 * np.eye(size, k=1) + 0.08 * np.eye(size, k=-1)
    if family == "dense_nonsymmetric":
        random = np.random.default_rng(84 + size).normal(0, 0.03 / np.sqrt(size), (size, size))
        return diagonal + random
    if family == "signed_indefinite":
        random = np.random.default_rng(126 + size).normal(0, 0.02 / np.sqrt(size), (size, size))
        signs = np.where(np.arange(size) % 2, -1.0, 1.0)
        return diagonal * signs + (random + random.T) / 2
    raise ValueError("Unknown fixture family")


def aer_crosscheck(matrix, rhs, degree):
    """Independent native-gate Aer statevector execution for manageable circuits."""
    op = build_large_inverse_action(matrix, degree=degree)
    circuit = transpile(op.circuit(rhs), basis_gates=["u", "cx"], optimization_level=1)
    counts = dict(circuit.count_ops())
    depth = circuit.depth()
    circuit.save_statevector()
    backend = AerSimulator(method="statevector", max_parallel_threads=1)
    vector = np.asarray(backend.run(circuit).result().get_statevector())
    solution = np.real_if_close(vector[: len(matrix)] * op.normalization * np.linalg.norm(rhs)).real
    structured = op.simulate(rhs)
    error = float(np.linalg.norm(solution - structured["solution"]))
    return {
        "size": len(matrix),
        "degree": degree,
        "method": "native-gate Aer statevector",
        "qubits": circuit.num_qubits,
        "native_operations": counts,
        "native_depth": depth,
        "structured_agreement_error": error,
        "passed": error < 1e-8,
    }


def campaign(output, *, sizes=(10, 100, 1000), shots=131072, max_memory_mib=512, crosscheck=True):
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("sizes must be nonempty and unique")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audit_dir = output / "audit"
    audit_dir.mkdir(exist_ok=True)
    rows = []
    for size in sizes:
        for family in ("dense_spd", "tridiagonal_spd", "dense_nonsymmetric", "signed_indefinite"):
            matrix = fixture(size, family)
            started = time.perf_counter()
            op = build_large_inverse_action(matrix, degree=7, max_memory_mib=max_memory_mib)
            encoding_seconds = time.perf_counter() - started
            rhs_set = {
                "basis_first": np.eye(1, size, 0).ravel(),
                "uniform": np.ones(size) / np.sqrt(size),
                "signed_random": np.random.default_rng(42).normal(size=size),
            }
            rhs_set["signed_random"] /= np.linalg.norm(rhs_set["signed_random"])
            observed = []
            solutions = []
            for index, (name, rhs) in enumerate(rhs_set.items()):
                started = time.perf_counter()
                result = op.simulate(rhs)
                simulation_seconds = time.perf_counter() - started
                solutions.append(result["solution"])
                sampling = op.sample_z(result, shots=shots, seed=42 + index)
                audit_file = audit_dir / f"{size}-{family}-{name}.json"
                audit_file.write_text(json.dumps(sampling, indent=2) + "\n")
                sampling.pop("conditional_counts")
                sampling["audit_file"] = str(audit_file.relative_to(output))
                observed.append(
                    {
                        "rhs": name,
                        "relative_residual": result["relative_residual"],
                        "success_probability": result["success_probability"],
                        "state_norm": result["state_norm"],
                        "padding_success_weight": result["padding_success_weight"],
                        "simulation_seconds": simulation_seconds,
                        "z_sampling": sampling,
                    }
                )
            # Independent reference computed only after circuit-state simulation.
            reference = solve(
                matrix,
                np.column_stack(list(rhs_set.values())),
                assume_a="pos" if family.endswith("spd") else "gen",
            )
            for index, result in enumerate(observed):
                result["relative_solution_error"] = float(
                    np.linalg.norm(solutions[index] - reference[:, index])
                    / np.linalg.norm(reference[:, index])
                )
            convergence = []
            # Reuse encoding; degree variants change only selector and flag evolution.
            for degree in (1, 3):
                smaller = build_large_inverse_action(
                    matrix, degree=degree, max_memory_mib=max_memory_mib
                )
                convergence.append(
                    {
                        "degree": degree,
                        "relative_residual": smaller.simulate(rhs_set["signed_random"])[
                            "relative_residual"
                        ],
                    }
                )
            convergence.append(
                {"degree": 7, "relative_residual": observed[-1]["relative_residual"]}
            )
            q = float(np.max(np.abs(op.contractions)))
            rows.append(
                {
                    "size": size,
                    "family": family,
                    "condition_number": op.condition_number,
                    "encoding_kind": op.encoding_kind,
                    "encoding_seconds": encoding_seconds,
                    "resources": op.resource_report,
                    "simulation_method": (
                        "exact structured statevector in classically computed spectral input basis"
                    ),
                    "native_aer_executed_at_this_degree": False,
                    "tau": op.tau,
                    "contraction_norm": q,
                    "all_input_relative_residual_bound": q**8,
                    "full_inverse_frobenius_residual_bound": float(np.sqrt(size) * q**8),
                    "full_inverse_read_out": False,
                    "rhs_results": observed,
                    "degree_convergence": convergence,
                    "passed": all(
                        r["relative_residual"] < 1e-3
                        and r["relative_solution_error"] < 1e-3
                        and abs(r["state_norm"] - 1) < 1e-9
                        and r["padding_success_weight"] < 1e-20
                        for r in observed
                    )
                    and all(
                        a["relative_residual"] > b["relative_residual"]
                        for a, b in zip(convergence, convergence[1:], strict=False)
                    ),
                }
            )
    checks = []
    if crosscheck:
        for size, degree in ((10, 3), (100, 1)):
            if size in sizes:
                checks.append(
                    aer_crosscheck(
                        fixture(size, "dense_spd"), np.ones(size) / np.sqrt(size), degree
                    )
                )
    # Deliberately hard SPD case; a fixed degree cannot solve every condition number.
    hard = build_large_inverse_action(np.diag(np.linspace(1.0, 100.0, 10)), degree=7)
    hard_residual = hard.simulate(np.eye(10)[0])["relative_residual"]
    limitations = {
        "ill_conditioned_spd": {
            "size": 10,
            "condition_number": 100,
            "degree": 7,
            "relative_residual": hard_residual,
            "meets_accuracy_target": hard_residual < 1e-3,
        },
        "unsupported_domains": ["singular", "numerically singular", "complex"],
        "1000_native_aer": (
            "not attempted: dense native gate compilation/execution is not validated at this size"
        ),
        "shot_readout": "Z-basis distribution sampling only; no signed inverse reconstruction",
        "preprocessing": "classical dense eigendecomposition/SVD O(N^3); no quantum speed-up claim",
    }
    passed = all(r["passed"] for r in rows) and all(c["passed"] for c in checks)
    passed = passed and hard_residual > 0.5
    payload = {
        "matrix_generation": {
            "seed": "family-specific fixed seeds plus matrix size",
            "reference_only_after_simulation": True,
        },
        "passed": bool(passed),
        "sizes": list(sizes),
        "shots": shots,
        "seed": 42,
        "cases": rows,
        "native_aer_crosschecks": checks,
        "limitations": limitations,
        "physical_hardware_tested": False,
        "noise_model": None,
        "versions": {
            n: importlib.metadata.version(n) for n in ("numpy", "scipy", "qiskit", "qiskit-aer")
        },
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Large standalone coherent inverse-action study",
        "",
        f"Status: {'PASS' if passed else 'FAIL'}",
        "",
        "Exact structured statevector simulation, including all selector and flag ancillas.",
        "Classical spectral input encoding is required. "
        "Results do not establish universal convergence.",
        "",
        "| Size | Family | Padded | Qubits | RHS residual | Min success | Inverse bound |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['size']} | {r['family']} | {r['resources']['padded_size']} | "
            f"{r['resources']['total_qubits']} | "
            f"{max(x['relative_residual'] for x in r['rhs_results']):.3e} | "
            f"{min(x['success_probability'] for x in r['rhs_results']):.2%} | "
            f"{r['full_inverse_frobenius_residual_bound']:.3e} |"
        )
    lines += [
        "",
        "The full-inverse column is an analytical bound, not full-matrix tomography.",
        "Z sampling does not reconstruct signed inverse entries.",
        "",
        "## Native Aer crosschecks",
        "",
    ]
    for c in checks:
        lines.append(
            f"- {c['size']}×{c['size']}, degree {c['degree']}: "
            f"agreement error {c['structured_agreement_error']:.3e}; "
            f"{c['native_operations'].get('cx', 0)} CX gates."
        )
    lines += [
        "",
        f"Counterexample: condition number 100, degree 7 gives residual {hard_residual:.3f}.",
        "Higher degree, better encoding/preconditioning, or another method is required.",
        "1000×1000 native-gate Aer, native-gate noise and physical hardware are not validated.",
        "The classical eigendecomposition/SVD and dense encoding "
        "prevent any claim of quantum advantage.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/coherent-large")
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 100, 1000])
    parser.add_argument("--shots", type=int, default=131072)
    parser.add_argument("--max-memory-mib", type=float, default=512)
    parser.add_argument("--skip-aer-crosscheck", action="store_true")
    args = parser.parse_args()
    result = campaign(
        args.output_dir,
        sizes=args.sizes,
        shots=args.shots,
        max_memory_mib=args.max_memory_mib,
        crosscheck=not args.skip_aer_crosscheck,
    )
    print((Path(args.output_dir) / "summary.md").read_text())
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
