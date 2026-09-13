"""Command-line entry point for the first QCeNN-MI feasibility benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .solver import InversionResult, solve_inverse_2x2

PAPER_DOI = "10.3390/s19184002"
PAPER_URL = "https://doi.org/10.3390/s19184002"
ASSUMPTION = (
    "A1: each column of the inverse of a real nonsingular 2x2 matrix can be "
    "represented by a normalized one-qubit real-amplitude state, up to a "
    "classical scalar, and recovered by residual minimization."
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qcenn-mi",
        description="Run QCeNN-MI Assumption A1 on a Qiskit quantum simulator.",
    )
    parser.add_argument(
        "--matrix",
        nargs=4,
        type=float,
        metavar=("M00", "M01", "M10", "M11"),
        default=[2.0, 0.5, 0.5, 1.5],
        help="2x2 matrix entries in row-major order.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=8192,
        help="Finite shots used for hardware-compatible X/Z readout verification.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=257,
        help="Coarse theta grid used before local scalar minimization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by the shot-based simulator.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/assumption-a1"),
        help="Directory receiving result.json and summary.md.",
    )
    parser.add_argument(
        "--exact-residual-threshold",
        type=float,
        default=1e-7,
        help="Acceptance threshold for ideal-statevector residual.",
    )
    parser.add_argument(
        "--shot-residual-threshold",
        type=float,
        default=8e-2,
        help="Acceptance threshold for finite-shot verification residual.",
    )
    return parser


def _as_markdown(
    result: InversionResult,
    *,
    exact_threshold: float,
    shot_threshold: float,
    passed: bool,
) -> str:
    matrix = np.asarray(result.matrix)
    estimate = np.asarray(result.inverse_estimate)
    exact = np.asarray(result.exact_inverse)
    shot_estimate = (
        None
        if result.shot_inverse_estimate is None
        else np.asarray(result.shot_inverse_estimate)
    )

    def matrix_block(value: np.ndarray) -> str:
        rendered = np.array2string(value, precision=8, suppress_small=True)
        return f"```text\n{rendered}\n```"

    exact_residual_row = (
        "| Frobenius residual `||MX-I||_F` | "
        f"{result.frobenius_residual:.6e} | < {exact_threshold:.1e} |"
    )

    lines = [
        "# QCeNN-MI Assumption A1 Result",
        "",
        f"**Status:** {'PASS' if passed else 'FAIL'}",
        "",
        f"**Assumption:** {ASSUMPTION}",
        "",
        "## Input matrix",
        matrix_block(matrix),
        "## Qiskit statevector inverse estimate",
        matrix_block(estimate),
        "## Classical reference inverse",
        matrix_block(exact),
        "## Metrics",
        "",
        "| Metric | Value | Acceptance |",
        "|---|---:|---:|",
        exact_residual_row,
        (
            "| Relative inverse error | "
            f"{result.relative_inverse_error:.6e} | informational |"
        ),
        f"| Condition number | {result.condition_number:.6f} | informational |",
    ]

    if result.shot_frobenius_residual is not None and shot_estimate is not None:
        shot_row = (
            f"| Finite-shot residual ({result.shots} shots) | "
            f"{result.shot_frobenius_residual:.6e} | < {shot_threshold:.2e} |"
        )
        lines.extend(
            [
                shot_row,
                "",
                "## Finite-shot inverse estimate",
                matrix_block(shot_estimate),
            ]
        )

    context = (
        "This experiment is a feasibility test of the smallest quantum state "
        "representation used by QCeNN-MI. It does **not** claim quantum advantage "
        "and does not yet implement the full locally coupled QCeNN dynamics."
    )
    citation = (
        "V. Tavakkoli, J. C. Chedjou, and K. Kyamakya, *A Novel Recurrent Neural "
        "Network-Based Ultra-Fast, Robust, and Scalable Solver for Inverting a "
        "‘Time-Varying Matrix’*, Sensors 19(18), 4002 (2019)."
    )

    lines.extend(
        [
            "## Scientific context",
            "",
            context,
            "",
            "The matrix-inversion objective is motivated by the prior dynamical solver:",
            "",
            citation,
            f"DOI: {PAPER_URL}",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    matrix = np.asarray(args.matrix, dtype=float).reshape(2, 2)
    result = solve_inverse_2x2(
        matrix,
        grid_points=args.grid_points,
        verification_shots=args.shots,
        seed=args.seed,
    )

    exact_ok = result.frobenius_residual < args.exact_residual_threshold
    shot_ok = (
        result.shot_frobenius_residual is not None
        and result.shot_frobenius_residual < args.shot_residual_threshold
    )
    passed = bool(exact_ok and shot_ok)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = result.to_dict()
    payload.update(
        {
            "assumption": ASSUMPTION,
            "paper_doi": PAPER_DOI,
            "acceptance": {
                "exact_residual_threshold": args.exact_residual_threshold,
                "shot_residual_threshold": args.shot_residual_threshold,
                "passed": passed,
            },
        }
    )

    result_path = args.output_dir / "result.json"
    summary_path = args.output_dir / "summary.md"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        _as_markdown(
            result,
            exact_threshold=args.exact_residual_threshold,
            shot_threshold=args.shot_residual_threshold,
            passed=passed,
        ),
        encoding="utf-8",
    )

    print(summary_path.read_text(encoding="utf-8"))
    print(f"\nJSON result: {result_path}")
    return 0 if passed else 1


def main() -> None:
    args = _parser().parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
