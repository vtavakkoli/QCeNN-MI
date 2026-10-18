"""Check truncation, operator, state and success bounds against simulated gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .compact import build_compact_inverse
from .precision_benchmark import audit_operator, controlled_fixture


def campaign(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in (2, 4):
        for condition in (2, 10, 100):
            for family in ("controlled_spd", "controlled_nonsymmetric", "controlled_indefinite"):
                matrix = controlled_fixture(size, family, condition)
                for degree in (1, 3, 7, 15, 31, 63, 127, 255, 511):
                    operation = build_compact_inverse(matrix, degree=degree, tolerance=1e-12)
                    audit = audit_operator(operation)
                    rhs = np.random.default_rng(42).normal(size=size)
                    result = operation.simulate(rhs)
                    reference = np.linalg.solve(matrix, rhs)
                    relative_error = float(
                        np.linalg.norm(result["solution"] - reference) / np.linalg.norm(reference)
                    )
                    state_error = float(
                        np.linalg.norm(
                            result["solution"] / np.linalg.norm(result["solution"])
                            - reference / np.linalg.norm(reference)
                        )
                    )
                    bound = operation.residual_bound
                    lower = max(0.0, 1 - bound) ** 2 / condition**2
                    passed = bool(
                        audit["operator_relative_residual_2"] <= bound + 1e-10
                        and relative_error <= bound + 1e-10
                        and state_error <= 2 * bound + 1e-10
                        and audit["minimum_success_probability_all_inputs"] >= lower - 1e-10
                    )
                    rows.append(
                        {
                            "size": size,
                            "family": family,
                            "condition_number": condition,
                            "degree": degree,
                            "residual_bound": bound,
                            "relative_solution_error": relative_error,
                            "normalized_state_error": state_error,
                            "success_lower_bound": lower,
                            **audit,
                            "passed": passed,
                        }
                    )
    payload = {
        "passed": all(r["passed"] for r in rows),
        "cases": rows,
        "floating_point_slack": 1e-10,
        "physical_hardware_tested": False,
        "method": "structured controlled-rotation gate simulation; reference audit only",
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (output / "summary.md").write_text(
        "# Theory-bound regression\n\n"
        f"{'PASS' if payload['passed'] else 'FAIL'}: {len(rows)} degree/condition/family cases.\n\n"
        "Operator residual, per-input relative solution error, normalized-state error and "
        "worst-input success probability are checked against the exact-arithmetic bounds "
        "with explicit 1e-10 floating-point slack. Low-degree inaccurate approximations "
        "are retained; bound agreement does not imply target accuracy.\n"
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/theory")
    args = parser.parse_args()
    result = campaign(args.output_dir)
    print(f"Theory-bound campaign: {len(result['cases'])} cases, passed={result['passed']}")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
