"""Matched dense classical solves and coherent inverse-action simulator costs."""

from __future__ import annotations

import argparse
import os
import platform
import time

import numpy as np
from scipy import linalg

from .compact import build_compact_inverse
from .experiment_io import configuration, save_result
from .precision_benchmark import controlled_fixture

DEFAULTS = {
    "sizes": [2, 10, 100, 1000],
    "conditions": [2, 10, 100, 1000],
    "families": ["controlled_spd", "controlled_nonsymmetric"],
    "tolerances": [1e-6, 1e-12],
    "seed": 1729,
    "repeats": 3,
    "warmups": 1,
}


def classical_solve(matrix, rhs, method):
    if method == "scipy_solve":
        return linalg.solve(matrix, rhs, assume_a="gen", check_finite=True)
    if method == "lu":
        return linalg.lu_solve(linalg.lu_factor(matrix), rhs)
    if method == "qr":
        q, r = linalg.qr(matrix, mode="economic")
        return linalg.solve_triangular(r, q.T @ rhs)
    if method == "svd":
        u, values, vh = linalg.svd(matrix, full_matrices=False)
        return vh.T @ ((u.T @ rhs) / values)
    raise ValueError("Unknown classical solver")


def cpu_description():
    try:
        from pathlib import Path

        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def campaign(output, **settings):
    cfg = {**DEFAULTS, **settings}
    if cfg["repeats"] < 1 or cfg["warmups"] < 0:
        raise ValueError("Positive repeats and nonnegative warmups required")
    rows = []
    for n in cfg["sizes"]:
        for condition in cfg["conditions"]:
            for family in cfg["families"]:
                start = time.perf_counter()
                matrix = controlled_fixture(n, family, condition, seed=cfg["seed"])
                rhs = np.random.default_rng(cfg["seed"] + n).normal(size=n)
                generation = time.perf_counter() - start
                start = time.perf_counter()
                matrix, rhs = matrix.copy(), rhs.copy()
                loading = time.perf_counter() - start
                key = {
                    "size": n,
                    "condition": condition,
                    "family": family,
                    "generation_seconds": generation,
                    "in_memory_copy_seconds": loading,
                }
                for method in ("scipy_solve", "lu", "qr", "svd"):
                    for _ in range(cfg["warmups"]):
                        classical_solve(matrix, rhs, method)
                    elapsed = []
                    for _ in range(cfg["repeats"]):
                        start = time.perf_counter()
                        answer = classical_solve(matrix, rhs, method)
                        elapsed.append(time.perf_counter() - start)
                    residual = float(np.linalg.norm(matrix @ answer - rhs) / np.linalg.norm(rhs))
                    rows.append(
                        {
                            **key,
                            "method": method,
                            "output": "complete classical solution vector",
                            "solve_seconds": elapsed,
                            "median_solve_seconds": float(np.median(elapsed)),
                            "median_end_to_end_seconds": generation
                            + loading
                            + float(np.median(elapsed)),
                            "relative_residual": residual,
                            "passed": residual < 1e-10,
                        }
                    )
                for tolerance in cfg["tolerances"]:
                    start = time.perf_counter()
                    operation = build_compact_inverse(matrix, tolerance=tolerance)
                    encoding = time.perf_counter() - start
                    start = time.perf_counter()
                    result = operation.simulate(rhs)
                    simulation = time.perf_counter() - start
                    reference = linalg.solve(matrix, rhs)
                    rows.append(
                        {
                            **key,
                            "method": "chebyshev_lcu_structured_simulator",
                            ("output"): (
                                "coherent inverse-action state; classical vector exposed only "
                                "by simulator audit"
                            ),
                            "target_tolerance": tolerance,
                            "degree": operation.degree,
                            "encoding_and_polynomial_seconds": encoding,
                            "structured_simulation_seconds": simulation,
                            "end_to_end_simulator_seconds": generation
                            + loading
                            + encoding
                            + simulation,
                            "relative_residual": result["relative_residual"],
                            "relative_solution_error": float(
                                np.linalg.norm(result["solution"] - reference)
                                / np.linalg.norm(reference)
                            ),
                            "success_probability": result["success_probability"],
                            "expected_attempts_per_accepted_output": 1
                            / result["success_probability"],
                            "resources": operation.resource_report,
                            "passed": result["relative_residual"] < max(1e-10, tolerance),
                        }
                    )
        print(f"Matched solver baseline completed for N={n}", flush=True)
    return save_result(
        output,
        {
            "passed": all(r["passed"] for r in rows),
            "configuration": cfg,
            "cases": rows,
            "environment": {
                "cpu": cpu_description(),
                "platform": platform.platform(),
                "logical_cpus": os.cpu_count(),
                "threads": {
                    k: os.environ.get(k)
                    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
                },
            },
            ("comparison_contract"): (
                "same dense real matrix and rhs; generation and in-memory "
                "loading stated; classical vector output and quantum-state "
                "output are different tasks"
            ),
            "quantum_speedup_ratio_reported": False,
            "quantum_baselines": [
                {
                    "method": "HHL",
                    "comparison": "analytical; not directly comparable numerical timing",
                    ("access_model"): (
                        "Hamiltonian simulation/oracle and efficient state preparation assumptions"
                    ),
                    "output": "solution-state observables, not all classical entries",
                    "source": "https://doi.org/10.1103/PhysRevLett.103.150502",
                },
                {
                    "method": "QSVT",
                    "comparison": "analytical; not an implemented benchmark",
                    "access_model": "block-encoding queries plus phase synthesis and preparation",
                    ("output"): (
                        "polynomially transformed state/block; readout and "
                        "amplification cost additional"
                    ),
                    "source": "https://arxiv.org/abs/1806.01838",
                },
            ],
            "physical_hardware_tested": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/baselines")
    parser.add_argument("--config")
    for name, kind in [("sizes", int), ("conditions", float), ("tolerances", float)]:
        parser.add_argument("--" + name, type=kind, nargs="+")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg = configuration(args.config, "baselines", DEFAULTS)
    cfg.update({k: v for k, v in vars(args).items() if k in DEFAULTS and v is not None})
    result = campaign(args.output_dir, **cfg)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
