"""Attach polynomial responses to the identical archived Shang comparison grid."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .polynomial_benchmark import save_input, sustained
from .polynomial_recurrent import accelerate_density_channel
from .polynomial_renewal import renewal_observables
from .recurrent_density import build_density_channel


def prior_comparison(output, settings):
    source = output.parent / "dissipative-comparison" / "result.json"
    if not source.exists():
        return {
            "available": False,
            "reason": "Run the full validation pipeline for the matched prior records.",
        }
    baseline = json.loads(source.read_text())
    found = []
    for row in baseline["cases"]:
        found.append(
            {
                "method": row["method"],
                "size": row["size"],
                "condition": row["condition"],
                "polynomial_degree": 0 if row["method"] == "discrete" else None,
                "final": row["final"],
                "trajectory": row["trajectory"],
                "first_sustained_sampled_update": row["first_sustained_sampled_time"],
                "archived_prior_record": True,
            }
        )
        if (
            row["method"] != "discrete"
            or row["size"] not in settings["sizes"]
            or row["condition"] not in settings["conditions"]
        ):
            continue
        path = source.parent / row["input_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["input_sha256"]:
            raise ValueError("Matched prior input checksum mismatch")
        with np.load(path) as arrays:
            matrix, rhs, reference = arrays["matrix"], arrays["rhs"], arrays["reference_solution"]
        file = f"inputs/prior-n{row['size']}-k{row['condition']}.npz"
        checksum = save_input(output / file, matrix=matrix, rhs=rhs, reference_solution=reference)
        base = build_density_channel(matrix, rhs)
        times = [point["time"] for point in row["trajectory"]]
        for degree in settings["degrees"]:
            channel = accelerate_density_channel(base, degree)
            rows = renewal_observables(channel, times[-1], times, "reference", reference)
            agreement = None
            if degree == 0:
                agreement = max(
                    abs(a["relative_residual"] - b["relative_residual"])
                    for a, b in zip(rows, row["trajectory"], strict=True)
                )
                if agreement > 1e-8:
                    raise ValueError("Renewal and prior baseline disagree")
            found.append(
                {
                    "method": "polynomial_discrete",
                    "size": row["size"],
                    "condition": row["condition"],
                    "polynomial_degree": degree,
                    "input_file": file,
                    "input_sha256": checksum,
                    "final": rows[-1],
                    "trajectory": rows,
                    "first_sustained_sampled_update": sustained(rows, settings),
                    "baseline_representation_residual_discrepancy": agreement,
                    "archived_prior_record": False,
                }
            )
    return {
        "available": True,
        "prior_source_commit": baseline["provenance"]["source_commit"],
        "same_rate_and_sampling_contract": baseline["comparison_contract"],
        "cases": found,
    }
