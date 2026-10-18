"""Audit refreshed and delayed quantum inverse action on changing dense matrices.

Every snapshot is classically encoded and compiled independently. No continuous
quantum tracker, device-noise immunity, real-time deadline, or QPU run is claimed.
Reconstructed operators are simulator-only audit outputs, never solver inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .compact import build_compact_inverse
from .precision_benchmark import audit_operator


def tracking_bound(delta, minimum_singular_value, drift_norm, noise_norm=0.0):
    """Bound both held-operator residuals with matrix drift and acquisition error."""
    values = np.asarray([delta, minimum_singular_value, drift_norm, noise_norm], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Finite parameters required")
    if not 0 <= delta < 1 or minimum_singular_value <= 0:
        raise ValueError("Require 0 <= delta < 1 and a positive singular-value bound")
    if drift_norm < 0 or not 0 <= noise_norm < minimum_singular_value:
        raise ValueError("Nonnegative drift and noise strictly below the singular gap required")
    return float(
        delta + (drift_norm + noise_norm) * (1 + delta) / (minimum_singular_value - noise_norm)
    )


def trajectory_bases(size, *, seed=1729):
    rng = np.random.default_rng(seed + size)
    left, _ = np.linalg.qr(rng.normal(size=(size, size)))
    right, _ = np.linalg.qr(rng.normal(size=(size, size)))
    return left, right


def trajectory_matrix(time, family, bases):
    """Rotating dense bases and kappa(t)=10**(1+2*t), t in [0,1]."""
    if not np.isfinite(time) or not 0 <= time <= 1:
        raise ValueError("Time must be in [0,1]")
    left, right = (b.copy() for b in bases)
    size = len(left)
    if size < 2 or family not in {"slow_spd", "fast_nonsymmetric"}:
        raise ValueError("A supported family and size >= 2 are required")
    rate = 0.2 if family == "slow_spd" else 4.0

    def rotate(basis, angle):
        c, s = np.cos(angle), np.sin(angle)
        first, last = basis[:, 0].copy(), basis[:, -1].copy()
        basis[:, 0] = c * first + s * last
        basis[:, -1] = -s * first + c * last

    rotate(left, rate * time)
    if family == "slow_spd":
        right = left
    else:
        rotate(right, -0.7 * rate * time)
    singular_values = np.geomspace(1.0, 10.0 ** (-1 - 2 * time), size)
    return (left * singular_values) @ right.T


def simulated_operator(operation):
    """Recover the complete success block using exact simulator amplitudes."""
    encoding = operation.encoding
    size = len(encoding.matrix)
    rhs = encoding.input_basis @ (np.ones(size) / np.sqrt(size))
    result = operation.simulate(rhs, return_success=True)
    amplitudes = encoding.input_basis.T @ (rhs / np.linalg.norm(rhs))
    response = (
        result["success_spectral_amplitudes"][:size] / amplitudes
    ).real * operation.normalization
    return (encoding.output_basis * response) @ encoding.input_basis.T


def campaign(output, *, sizes=(2, 10, 100, 1000), times=(0.0, 0.25, 0.5, 0.75, 1.0)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    snapshots, held, acquisition, singular = [], [], [], []
    tolerance, slack, residual_limit = 1e-12, 1e-10, 1e-10
    for size in sizes:
        bases = trajectory_bases(size)
        for family in ("slow_spd", "fast_nonsymmetric"):
            previous = None
            for time in times:
                matrix = trajectory_matrix(time, family, bases)
                operation = build_compact_inverse(matrix, tolerance=tolerance)
                audit = audit_operator(operation)
                inverse = simulated_operator(operation)
                right_residual = float(np.linalg.norm(inverse @ matrix - np.eye(size), ord=2))
                minimum = float(np.min(operation.encoding.singular_values))
                row = {
                    "size": size,
                    "family": family,
                    "time": time,
                    "condition_number": operation.encoding.condition_number,
                    "minimum_singular_value": minimum,
                    "degree": operation.degree,
                    "residual_bound": operation.residual_bound,
                    "resources": operation.resource_report,
                    **audit,
                    "right_operator_residual_2": right_residual,
                    "passed": max(audit["operator_relative_residual_2"], right_residual)
                    <= residual_limit
                    and audit["relative_inverse_error_frobenius"] <= residual_limit
                    and audit["state_norm_error"] < slack
                    and audit["padding_success_weight"] < 1e-24,
                }
                snapshots.append(row)
                if previous is not None:
                    old_matrix, old_inverse, old_row = previous
                    drift = float(np.linalg.norm(matrix - old_matrix, ord=2))
                    bound = tracking_bound(
                        old_row["residual_bound"], old_row["minimum_singular_value"], drift
                    )
                    left = float(np.linalg.norm(matrix @ old_inverse - np.eye(size), ord=2))
                    right = float(np.linalg.norm(old_inverse @ matrix - np.eye(size), ord=2))
                    held.append(
                        {
                            "size": size,
                            "family": family,
                            "from_time": old_row["time"],
                            "to_time": time,
                            "drift_norm": drift,
                            "left_residual": left,
                            "right_residual": right,
                            "tracking_bound": bound,
                            "passed": max(left, right) <= bound + slack,
                        }
                    )
                previous = matrix, inverse, row
                print(
                    f"N={size} {family} t={time:g} kappa={row['condition_number']:.3g} "
                    f"fresh residual={audit['operator_relative_residual_2']:.3e}",
                    flush=True,
                )
            # Matrix acquisition error is distinct from quantum gate noise.
            index = int(np.argmin(operation.encoding.singular_values))
            direction = np.outer(
                operation.encoding.input_basis[:, index],
                operation.encoding.output_basis[:, index],
            )
            for ratio in (0.001, 0.1):
                noisy_matrix = matrix + ratio * minimum * direction
                noise = float(np.linalg.norm(noisy_matrix - matrix, ord=2))
                noisy = build_compact_inverse(noisy_matrix, tolerance=tolerance)
                noisy_inverse = simulated_operator(noisy)
                actual = max(
                    float(np.linalg.norm(matrix @ noisy_inverse - np.eye(size), ord=2)),
                    float(np.linalg.norm(noisy_inverse @ matrix - np.eye(size), ord=2)),
                )
                bound = tracking_bound(noisy.residual_bound, minimum, 0.0, noise)
                acquisition.append(
                    {
                        "size": size,
                        "family": family,
                        "time": time,
                        "noise_to_gap_ratio": ratio,
                        "noise_norm": noise,
                        "operator_residual": actual,
                        "tracking_bound": bound,
                        "passed": actual <= bound + slack,
                    }
                )
            try:
                build_compact_inverse(matrix - minimum * direction, tolerance=tolerance)
                rejected = False
                reason = "Unexpectedly accepted"
            except ValueError as error:
                reason = str(error)
                rejected = "singular" in reason.lower()
            singular.append(
                {"size": size, "family": family, "explicitly_rejected": rejected, "reason": reason}
            )
    payload = {
        "passed": all(r["passed"] for r in snapshots + held + acquisition)
        and all(r["explicitly_rejected"] for r in singular),
        "snapshots": snapshots,
        "held_operator_pairs": held,
        "acquisition_errors": acquisition,
        "singular_endpoints": singular,
        "configuration": {
            "sizes": list(sizes),
            "times": list(times),
            "basis_seed": 1729,
            "condition_number_law": "10**(1+2*t)",
            "rotation_rates": {"slow_spd": 0.2, "fast_nonsymmetric": 4.0},
            "polynomial_tolerance": tolerance,
            "numerical_residual_limit": residual_limit,
            "bound_comparison_slack": slack,
        },
        "method": "snapshot-compiled ideal structured quantum statevector",
        "classical_encoding": "fresh dense spectral decomposition/SVD at every snapshot",
        "physical_hardware_tested": False,
        "native_gate_simulation_tested": False,
        "real_time_deadline_tested": False,
        "continuous_quantum_tracking_tested": False,
        "acquisition_error_is_quantum_gate_noise": False,
    }
    (output / "result.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (output / "summary.md").write_text(
        "# Time-varying quantum inverse-action audit\n\n"
        f"Validation: {'PASS' if payload['passed'] else 'FAIL'}. "
        f"{len(snapshots)} refreshed snapshots, {len(held)} delayed-operator pairs, "
        f"{len(acquisition)} acquisition-error cases.\n\n"
        "Condition numbers increase from 10 to 1000 along rotating dense SPD and "
        "nonsymmetric trajectories. Delayed and perturbed operators are checked against "
        "bounds, not required to meet the fresh-operator accuracy gate. No gate noise, "
        "hardware execution, real-time deadline, or continuous quantum tracker is tested.\n"
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/time-varying")
    args = parser.parse_args()
    result = campaign(args.output_dir)
    print((Path(args.output_dir) / "summary.md").read_text())
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
