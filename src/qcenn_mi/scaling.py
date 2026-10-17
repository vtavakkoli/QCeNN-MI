"""Block-hybrid size tests: quantum 2x2 seeds, classical full-matrix correction.

A 100x100 test here is not a 100x100 amplitude-encoded variational solve.
Cross-block interactions are handled by classical dense polynomial products.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .coupled import solve_coupled, validate_square_matrix
from .tracking import InverseTracker, polynomial_correct


def solve_block_hybrid(matrix, *, tolerance=1e-8, max_steps=12, shots=None, seed=42):
    """Even-size block-Jacobi quantum initializer followed by residual polynomials.

    Explicitly returns failure if blocks are singular or the full residual does
    not converge; no classical inverse is substituted on failure. Shot readout,
    if requested, is applied to every 2x2 seed before the classical corrections.
    """
    m = validate_square_matrix(matrix)
    n = len(m)
    if n % 2:
        raise ValueError("Block-hybrid solver requires an even matrix size")
    if not np.isfinite(tolerance) or tolerance <= 0 or max_steps < 0:
        raise ValueError("Invalid tolerance or step budget")
    x = np.zeros_like(m)
    block_residuals = []
    for start in range(0, n, 2):
        block = solve_coupled(m[start : start + 2, start : start + 2], seed=seed + start)
        if not block.success:
            raise RuntimeError(f"Quantum seed block {start // 2} failed to converge")
        value = block.estimate
        if shots is not None:
            value, _ = block.measure(shots=shots, seed=seed + start)
        x[start : start + 2, start : start + 2] = value
        block_residuals.append(float(np.linalg.norm(block.matrix @ value - np.eye(2))))
    residuals = [float(np.linalg.norm(m @ x - np.eye(n)))]
    for _ in range(max_steps):
        if residuals[-1] < tolerance:
            break
        x = polynomial_correct(m, x, order=2)
        residuals.append(float(np.linalg.norm(m @ x - np.eye(n))))
        if residuals[-1] >= residuals[-2]:
            break
    return {
        "size": n,
        "mode": "2x2 quantum blocks + classical dense polynomial correction",
        "quantum_blocks": n // 2,
        "shots_per_setting": shots,
        "block_residuals": block_residuals,
        "residual_history": residuals,
        "converged": residuals[-1] < tolerance,
        "estimate": x,
    }


def fixture(size, *, seed=42, symmetric=True):
    rng = np.random.default_rng(seed)
    off = rng.normal(0, 0.08 / np.sqrt(size), (size, size))
    np.fill_diagonal(off, 0)
    if symmetric:
        off = (off + off.T) / 2
    return np.diag(np.linspace(1.5, 2.5, size)) + off


def campaign(output, *, shots=4096, steps=21):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in (10, 100):
        for symmetric in (True, False):
            m = fixture(n, symmetric=symmetric)
            for seed_shots in (None, shots):
                result = solve_block_hybrid(m, shots=seed_shots)
                estimate = result.pop("estimate")
                reference = np.linalg.inv(m)  # Independent reporting only, after solve.
                result.update(
                    case="symmetric" if symmetric else "nonsymmetric",
                    relative_inverse_error=float(
                        np.linalg.norm(estimate - reference) / np.linalg.norm(reference)
                    ),
                    condition_number=float(np.linalg.cond(m)),
                )
                rows.append(result)
    tracker = InverseTracker()
    dynamic = []
    for t in np.linspace(0, 2 * np.pi, steps):
        # Both diagonal and off-diagonal entries vary; no fixed eigenbasis.
        m = np.array(
            [
                [2 + 0.2 * np.sin(t), 0.3 * np.cos(t)],
                [0.2 * np.sin(t + 0.4), 1.5 + 0.15 * np.cos(t)],
            ]
        )
        record = tracker.update(m, time=t)
        measured, _ = tracker.state.measure(shots=shots, seed=42 + len(dynamic))
        record.update(
            matrix=m.tolist(), shot_residual=float(np.linalg.norm(m @ measured - np.eye(2)))
        )
        dynamic.append(record)
    passed = all(r["converged"] and r["relative_inverse_error"] < 1e-8 for r in rows)
    passed = passed and all(
        r["quantum_residual"] < 1e-7 and r["corrected_residual"] < 1e-8 and r["shot_residual"] < 0.1
        for r in dynamic
    )
    payload = {
        "passed": bool(passed),
        "seed": 42,
        "large_matrix": rows,
        "dynamic_2x2": dynamic,
        "physical_hardware_tested": False,
        "scope": "Large cases use quantum 2x2 blocks and classical full-matrix correction",
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Larger matrix and dynamic simulator checks",
        "",
        f"Status: {'PASS' if passed else 'FAIL'}",
        "",
        payload["scope"] + ".",
        "",
        "| Size | Case | Seed shots | Initial residual | Final residual | Relative error |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['size']} | {r['case']} | {r['shots_per_setting'] or 'ideal'} | "
            f"{r['residual_history'][0]:.3e} | {r['residual_history'][-1]:.3e} | "
            f"{r['relative_inverse_error']:.3e} |"
        )
    lines += [
        "",
        f"Dynamic 2x2: {len(dynamic)} time points; maximum quantum residual "
        f"{max(r['quantum_residual'] for r in dynamic):.3e}; maximum shot residual "
        f"{max(r['shot_residual'] for r in dynamic):.3e} ({shots} shots per setting).",
        "",
        "No physical hardware test or large amplitude-encoded quantum solve is claimed.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/scaling")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=21)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    result = campaign(args.output_dir, shots=args.shots, steps=args.steps)
    print((Path(args.output_dir) / "summary.md").read_text())
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
