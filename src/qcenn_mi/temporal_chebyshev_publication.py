"""Matched time-varying resident, restarted, and one-shot Chebyshev filtering.

These are classical density-matrix simulations of quantum channels, not native
circuits or hardware speedups. Direct one-shot filtering is postselected and
its expected retry cost is explicitly recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from .recurrent_density import build_density_channel


def certified_degree(gap: float, suppression: float = 0.1) -> int:
    if not (0 < gap < 1 and 0 < suppression < 1):
        raise ValueError("gap and suppression must be strictly between zero and one")
    return max(1, int(np.ceil(np.arccosh(1 / suppression) /
                              (2 * np.arctanh(np.sqrt(gap))))))


def _logcosh(values):
    values = np.abs(values)
    return values + np.log1p(np.exp(-2 * values)) - np.log(2.0)


def response(eigenvalues: np.ndarray, gap: float, degree: int) -> np.ndarray:
    """Independent Chebyshev response with its signed positive-mode values."""
    if not (0 < gap < 1) or degree < 1 or isinstance(degree, bool):
        raise ValueError("invalid Chebyshev specification")
    lam = np.asarray(eigenvalues, dtype=float)
    if not np.isfinite(lam).all() or np.any(lam < -1e-11) or np.any(lam > 1 + 1e-11):
        raise ValueError("eigenvalues must lie in [0,1]")
    x = (1 + gap - 2 * np.clip(lam, 0, 1)) / (1 - gap)
    normalizer = _logcosh(degree * np.arccosh((1 + gap) / (1 - gap)))
    positive = np.exp(_logcosh(degree * np.arccosh(np.maximum(x, 1))) - normalizer)
    signed = np.cos(degree * np.arccos(np.clip(x, -1, 1))) * np.exp(-normalizer)
    return np.where(x > 1, positive, signed)


def matrix_family(size: int, condition: float, seed: int, snapshot: int,
                  snapshots: int, drift: str):
    """Rotate right singular vectors and change the right-hand side."""
    if size < 2 or condition <= 1 or snapshots < 2 or drift not in {"slow", "abrupt"}:
        raise ValueError("unsupported experiment fixture")
    rng = np.random.default_rng(seed + 7 * size)
    left, _ = np.linalg.qr(rng.normal(size=(size, size)))
    right, _ = np.linalg.qr(rng.normal(size=(size, size)))
    theta = (0.10 if drift == "slow" else 0.75) * snapshot / (snapshots - 1)
    if drift == "abrupt" and snapshot >= snapshots // 2:
        theta += 1.10
    rotation = np.eye(size)
    rotation[:2, :2] = [[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]]
    singular = np.geomspace(1.0, 1 / condition, size)
    matrix = left @ np.diag(singular) @ (right @ rotation).T
    basis_rhs = rng.normal(size=size)
    disturbance = rng.normal(size=size)
    rhs = basis_rhs + (0.05 if drift == "slow" else 0.4) * np.sin(
        (snapshot + 1) * 1.7) * disturbance
    rhs /= np.linalg.norm(rhs)
    return matrix, rhs


def target_state(matrix, rhs, channel):
    """Reference solution is exclusively a scorer, never an update input."""
    solution = np.linalg.solve(matrix, rhs)
    z = channel.singular_ratio * channel.matrix_scale * solution / np.linalg.norm(rhs)
    target = np.zeros(channel.dimension)
    target[:channel.size] = z
    target[channel.size] = 1
    return solution, target / np.linalg.norm(target)


def physical_filter(channel, gap, degree, attenuation=1.0):
    values = attenuation * response(channel.residual_eigenvalues, gap, degree)
    return (channel.basis * values) @ channel.basis.T


def apply_recurrent(density, operator, reference, noise_per_query=0.0,
                    queries=0):
    """Reset CPTP channel plus optional query-coupled logical depolarization."""
    no_event = operator @ density @ operator.T
    event = np.trace(density) - np.trace(no_event)
    result = no_event + float(event.real) * reference
    if noise_per_query:
        p = -np.expm1(queries * np.log1p(-noise_per_query))
        result = (1 - p) * result + p * np.trace(result) * np.eye(len(result)) / len(result)
    return (result + result.T) / 2


def direct_postselect(operator, reference, noise_per_query=0.0, queries=0):
    branch = operator @ reference @ operator.T
    success = float(np.trace(branch).real)
    if success <= 1e-14:
        return None, success
    accepted = branch / success
    if noise_per_query:
        # Logical depolarization AFTER postselection is not physical QSP gate noise.
        p = -np.expm1(queries * np.log1p(-noise_per_query))
        accepted = (1 - p) * accepted + p * np.eye(len(accepted)) / len(accepted)
    return (accepted + accepted.T) / 2, success


def observe(density, matrix, rhs, channel, solution, target, target_residual,
            target_infidelity):
    if density is None:
        return {"relative_residual": None, "conditional_infidelity": None,
                "purified_distance": None, "reference_probability": None,
                "accuracy_passed": False, "trace_error": None,
                "minimum_eigenvalue": None}
    n = channel.size
    pref = float(density[n, n])
    if pref > 1e-14:
        estimate = np.asarray(density[:n, n]).real * (
            np.linalg.norm(rhs) / (channel.singular_ratio * channel.matrix_scale * pref))
        relative_residual = float(np.linalg.norm(matrix @ estimate - rhs) /
                                  np.linalg.norm(rhs))
    else:
        relative_residual = None
    psol = float(np.trace(density[:n, :n]).real)
    if psol > 1e-14:
        axis = solution / np.linalg.norm(solution)
        infidelity = float(max(0, 1 - axis @ density[:n, :n] @ axis / psol))
    else:
        infidelity = None
    fidelity = float(target @ density @ target)
    output = {
        "relative_residual": relative_residual,
        "conditional_infidelity": infidelity,
        "purified_distance": float(np.sqrt(max(0, 1 - np.clip(fidelity, 0, 1)))),
        "reference_probability": pref,
        "solution_sector_probability": psol,
        "trace_error": float(abs(np.trace(density) - 1)),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(density).min()),
    }
    output["accuracy_passed"] = bool(relative_residual is not None and
                                     infidelity is not None and
                                     relative_residual <= target_residual and
                                     infidelity <= target_infidelity)
    return output


def run_case(*, size=2, condition=2.0, seed=23, snapshots=6,
             drift="slow", query_budget=64, suppression=0.1,
             noise_per_query=0.0, target_residual=1e-3,
             target_infidelity=1e-3, attenuation=1.0):
    """Equal per-snapshot query ceiling; never equate postselected shot cost."""
    if query_budget < 2 or snapshots < 2 or not 0 <= noise_per_query < 1:
        raise ValueError("invalid budget, schedule or noise probability")
    if not 0 < attenuation <= 1:
        raise ValueError("attenuation must be in (0,1]")
    gap = 1 / (1 + condition**2)
    degree = certified_degree(gap, suppression)
    calls = 2 * degree
    updates = query_budget // calls
    if updates < 1:
        raise ValueError("budget cannot pay for one recurrent Chebyshev macro")
    direct_degree = query_budget // 2
    per_snapshot = []
    resident = None
    maxima = {method: {"passed": 0, "relative_residual": 0.0}
              for method in ["resident", "fresh", "one_shot"]}
    for snapshot in range(snapshots):
        matrix, rhs = matrix_family(size, condition, seed, snapshot, snapshots, drift)
        channel = build_density_channel(matrix, rhs)
        if gap > channel.singular_ratio**2 / (1 + channel.singular_ratio**2) + 1e-12:
            raise AssertionError("fixture violates certified gap")
        solution, target = target_state(matrix, rhs, channel)
        reference = np.zeros((channel.dimension, channel.dimension))
        reference[channel.size, channel.size] = 1
        operator = physical_filter(channel, gap, degree, attenuation)
        if resident is None:
            resident = reference.copy()
        fresh = reference.copy()
        for _ in range(updates):
            resident = apply_recurrent(resident, operator, reference,
                                       noise_per_query, calls)
            fresh = apply_recurrent(fresh, operator, reference,
                                    noise_per_query, calls)
        direct_operator = physical_filter(channel, gap, direct_degree, attenuation)
        direct, probability = direct_postselect(
            direct_operator, reference, noise_per_query, 2 * direct_degree)
        observations = {}
        for name, density in {"resident": resident, "fresh": fresh,
                              "one_shot": direct}.items():
            result = observe(density, matrix, rhs, channel, solution, target,
                             target_residual, target_infidelity)
            result["charged_queries_per_snapshot"] = (updates * calls
                                                       if name != "one_shot"
                                                       else 2 * direct_degree)
            if name == "one_shot":
                result["postselection_success_probability"] = probability
                result["expected_queries_one_accepted_preparation"] = (
                    2 * direct_degree / probability if probability > 0 else None)
                result["one_accepted_preparation_within_expected_budget"] = bool(
                    probability > 0 and 2 * direct_degree / probability <= query_budget)
            observations[name] = result
            maxima[name]["passed"] += int(result["accuracy_passed"])
            if result["relative_residual"] is not None:
                maxima[name]["relative_residual"] = max(
                    maxima[name]["relative_residual"], result["relative_residual"])
            if result["trace_error"] is not None and (
                result["trace_error"] > 1e-9 or result["minimum_eigenvalue"] < -1e-9
            ):
                raise ArithmeticError("invalid density in matched benchmark")
        per_snapshot.append({"snapshot": snapshot, "condition": condition,
                             "rhs_norm": float(np.linalg.norm(rhs)),
                             "matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
                             "rhs_sha256": hashlib.sha256(rhs.tobytes()).hexdigest(),
                             "observations": observations})
    return {"case": {"size": size, "condition": condition, "seed": seed,
                     "snapshots": snapshots, "drift": drift,
                     "query_budget_per_snapshot": query_budget,
                     "suppression": suppression, "degree": degree,
                     "macro_updates": updates, "recurrent_queries_charged": updates * calls,
                     "one_shot_degree": direct_degree,
                     "one_shot_queries_charged": 2 * direct_degree,
                     "logical_noise_per_oracle_call": noise_per_query,
                     "filter_attenuation": attenuation,
                     "target_residual": target_residual,
                     "target_infidelity": target_infidelity},
            "summary": maxima, "trajectory": per_snapshot,
            "comparison_scope": (
                "Exact classical reduced-density simulation. Equal per-snapshot "
                "primitive-query ceiling; one-shot output is postselected "
                "and expected retries are charged separately. Any persistent "
                "benefit is empirical, not a proved speedup. No native circuit "
                "or hardware run was performed.")}


def independently_verify(case, *, tolerance=1e-9):
    """Numerically distinct Chebyshev polynomial evaluation cross-check."""
    spec = case["case"]
    gap = 1 / (1 + spec["condition"]**2)
    if spec["one_shot_degree"] > 64:
        return {"status": "not_checked", "reason": "degree > 64 cross-check cap"}
    xs = np.linspace(gap, 1, 101)
    x = (1 + gap - 2 * xs) / (1 - gap)
    coefficients = [0] * spec["one_shot_degree"] + [1]
    direct = np.polynomial.chebyshev.chebval(x, coefficients)
    denom = np.polynomial.chebyshev.chebval((1 + gap) / (1 - gap), coefficients)
    discrepancy = float(np.max(np.abs(
        response(xs, gap, spec["one_shot_degree"]) - direct / denom)))
    if discrepancy > tolerance:
        raise ArithmeticError(f"independent Chebyshev response mismatch: {discrepancy}")
    return {"status": "passed", "maximum_response_discrepancy": discrepancy,
            "method": "independent numpy.polynomial.chebyshev.chebval"}


def run_grid(*, output_dir: Path, sizes=(2, 4), conditions=(2, 10),
             budgets=(32, 64, 128), drifts=("slow", "abrupt"),
             noises=(0.0, 0.001), seeds=(23, 29), snapshots=6):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory must be empty: preserve earlier evidence")
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for size in sizes:
        for condition in conditions:
            for budget in budgets:
                for drift in drifts:
                    for noise in noises:
                        for seed in seeds:
                            if budget < 2 * certified_degree(1 / (1 + condition**2)):
                                continue
                            result = run_case(size=size, condition=condition, seed=seed,
                                              drift=drift, query_budget=budget,
                                              noise_per_query=noise, snapshots=snapshots)
                            result["independent_check"] = independently_verify(result)
                            cases.append(result)
    if not cases:
        raise ValueError("No affordable configurations in grid")
    summary = []
    for case in cases:
        row = dict(case["case"])
        for method, values in case["summary"].items():
            row[f"{method}_passing_snapshots"] = values["passed"]
            row[f"{method}_worst_relative_residual"] = values["relative_residual"]
        row["resident_minus_fresh_passing_snapshots"] = (
            case["summary"]["resident"]["passed"] - case["summary"]["fresh"]["passed"])
        summary.append(row)
    payload = {"schema_version": 1, "cases": cases, "summary": summary,
               "physical_hardware_tested": False, "qsp_phase_synthesis_tested": False,
               "status": "completed_classical_reduced_channel_experiment"}
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n")
    diffs = [r["resident_minus_fresh_passing_snapshots"] for r in summary]
    lines = ["# Time-varying matched Chebyshev comparison", "",
             "All cases use the same per-snapshot primitive-query **ceiling**.",
             "One-shot direct filtering is conditional on postselection; expected",
             "retry cost is included in result.json. Noise is a logical",
             "query-coupled depolarizing model, **not** physical QSP gate noise.",
             "", "| Cases | Resident better than fresh | Equal | Fresh better |",
             "|---:|---:|---:|---:|",
             f"| {len(diffs)} | {sum(x > 0 for x in diffs)} | "
             f"{sum(x == 0 for x in diffs)} | {sum(x < 0 for x in diffs)} |",
             "", "Do not claim a recurrent speedup when evidence is tied or negative.",
             "The resident, fresh and postselected tasks have different",
             "outputs/resources beyond the raw query cap; consult result.json."]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(output_dir.iterdir()) if p.is_file()}
    (output_dir / "MANIFEST.json").write_text(
        json.dumps({"sha256": checksums}, indent=2) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        run_grid(output_dir=args.output_dir, sizes=(2,), conditions=(2,),
                 budgets=(32,), seeds=(23,), noises=(0.0,), snapshots=4)
    else:
        run_grid(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
