"""Audit new paired temporal results against the retained historical static study.

The static and temporal test populations differ: never pool their rates or
claim a historical-to-new speedup. Validate the manuscript's exact table
against new reproduced raw results. Does not certify quantum hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

METHODS = ("resident", "fresh", "one_shot")


def summarize(cases):
    if not cases:
        raise ValueError("Temporal case list is empty")
    paired = Counter()
    passing = Counter()
    subgroup = {}
    ratios = []
    total_snapshots = 0
    for case in cases:
        spec = case["case"]
        records = case["trajectory"]
        if len(records) != spec["snapshots"]:
            raise ValueError("Incomplete temporal trajectory")
        counts = {m: 0 for m in METHODS}
        for record in records:
            total_snapshots += 1
            obs = record["observations"]
            for method in METHODS:
                row = obs[method]
                if row["charged_queries_per_snapshot"] > spec["query_budget_per_snapshot"]:
                    raise ValueError("Primitive call budget exceeded")
                if row["accuracy_passed"]:
                    counts[method] += 1
                    passing[method] += 1
                    key = (spec["drift"], spec["logical_noise_per_oracle_call"])
                    subgroup.setdefault(key, Counter())[method] += 1
                if row["trace_error"] is not None and row["trace_error"] > 1e-8:
                    raise ValueError("Trace violation")
            accept = obs["one_shot"]
            if accept["postselection_success_probability"] <= 0:
                raise ValueError("No accepted one-shot preparations")
            ratios.append(
                accept["expected_queries_one_accepted_preparation"]
                / spec["query_budget_per_snapshot"]
            )
        if counts != {m: case["summary"][m]["passed"] for m in METHODS}:
            raise ValueError("Temporal case summary differs from raw trajectory")
        delta = counts["resident"] - counts["fresh"]
        paired["resident_wins" if delta > 0 else "fresh_wins" if delta < 0 else "ties"] += 1
    import statistics

    return {
        "number_of_cases": len(cases),
        "snapshots_per_method": total_snapshots,
        "passing": {m: passing[m] for m in METHODS},
        "paired": dict(paired),
        "by_drift": {
            drift: {m: sum(v[m] for (d, _), v in subgroup.items() if d == drift) for m in METHODS}
            for drift in sorted({case["case"]["drift"] for case in cases})
        },
        "by_noise": {
            str(noise): {
                m: sum(v[m] for (_, n), v in subgroup.items() if n == noise) for m in METHODS
            }
            for noise in sorted({case["case"]["logical_noise_per_oracle_call"] for case in cases})
        },
        "one_shot_expected_calls_over_single_attempt_budget": {
            "minimum": min(ratios),
            "median": statistics.median(ratios),
            "maximum": max(ratios),
        },
    }


def assert_manuscript_table(tex, summary):
    """Fail on manuscript/evidence divergence for the complete canonical grid."""
    text = Path(tex).read_text(encoding="utf-8")
    if summary["number_of_cases"] != 96 or summary["snapshots_per_method"] != 576:
        raise ValueError("Canonical manuscript table requires exactly 96 six-snapshot cases")

    def extract(label):
        match = re.search(rf"{label}[^\n&]*\([^)]*\)\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*(\d+)", text)
        if not match:
            raise ValueError(f"Expected manuscript table row missing: {label}")
        return dict(zip(METHODS, map(int, match.groups()), strict=True))

    expected = {
        "All snapshots": summary["passing"],
        "Gradual change": summary["by_drift"]["slow"],
        "Abrupt change": summary["by_drift"]["abrupt"],
    }
    for label, actual in expected.items():
        observed = extract(label)
        if observed != actual:
            raise ValueError(f"Manuscript table mismatch {label}: {observed} != {actual}")
    noisy = summary["by_noise"].get("0.001")
    if noisy is None or extract("Logical") != noisy:
        raise ValueError("Manuscript logical-noise row differs from the raw evidence")
    return {"status": "passed", "verified_rows": list(expected) + ["Logical p=0.001"]}


def build_report(previous, temporal, phase, manuscript):
    previous_metrics = {
        "static_cases": previous["static_cases"],
        "static_accuracy_passed": previous["static_accuracy_passed"],
        "static_accuracy_failed": previous["static_cases"] - previous["static_accuracy_passed"],
        "original_derived_from": "separate historical static polynomial recurrence experiment",
    }
    if previous_metrics["static_accuracy_failed"] < 0:
        raise ValueError("Invalid historical counts")
    latest = summarize(temporal["cases"])
    if not all(item.get("independent_check", {}).get("status") == "passed"
               for item in temporal["cases"]):
        raise ValueError("Independent Chebyshev numerical check not verified")
    if not phase["records"] or not all(row.get("numerical_validity_passed", False)
                                      for row in phase["records"]):
        raise ValueError("QSP phase audit failed numerical validity checks")
    table = assert_manuscript_table(manuscript, latest)
    return {
        "schema_version": 1,
        "previous_static": previous_metrics,
        "new_paired_temporal": latest,
        "paper_table": table,
        "qsp_phase": [{
            "condition": row["condition"],
            "degree": row["degree"],
            "maximum_qsp_block_error": row["maximum_qsp_block_error"],
            "coefficient_error_estimate": row["coefficient_l1_error_estimate"],
            "noise_induced_infidelity_bias": row["noise_induced_infidelity_bias"],
        } for row in phase["records"]],
        "matched_study_verified": True,
        "can_claim_end_to_end_quantum_speedup": False,
        "physical_device_results": False,
        "full_paper_editorial_signoff": False,
        "comparison_scope": (
            "Prior static and new temporal snapshots are different populations; "
            "do not subtract or pool their pass rates as method effects."
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--previous", type=Path,
        default=Path("results/polynomial-access-validation/polynomial-summary.json"),
    )
    p.add_argument("--temporal", type=Path, required=True)
    p.add_argument("--phase", type=Path, required=True)
    p.add_argument("--manuscript", type=Path, default=Path("manuscript/QCeNN_MI.tex"))
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    for path in (args.previous, args.temporal, args.phase, args.manuscript):
        if not path.is_file():
            p.error(f"Required input not found: {path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Comparison output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        json.loads(args.previous.read_text()),
        json.loads(args.temporal.read_text()),
        json.loads(args.phase.read_text()),
        args.manuscript,
    )
    output = args.output_dir / "comparison.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    prior = report["previous_static"]
    recent = report["new_paired_temporal"]
    ratio = recent["one_shot_expected_calls_over_single_attempt_budget"]["median"]
    rows = [
        "# QCeNN-MI: historical and new publication evidence",
        "",
        "| Population | Method | Passed | Total | Comparison boundary |",
        "|---|---|---:|---:|---|",
        (f"| Historical static | Power-polynomial grid | {prior['static_accuracy_passed']} | "
         f"{prior['static_cases']} | Historical only |"),
        (f"| New changing input | Resident Chebyshev | {recent['passing']['resident']} | "
         f"{recent['snapshots_per_method']} | Strict paired restart control |"),
        (f"| New changing input | Restarted Chebyshev | {recent['passing']['fresh']} | "
         f"{recent['snapshots_per_method']} | Same CPTP query ceiling |"),
        (f"| New changing input | Direct one-shot | {recent['passing']['one_shot']} | "
         f"{recent['snapshots_per_method']} | Postselected, expected calls charged separately |"),
        "",
        (
            f"Paired case outcomes: {recent['paired'].get('resident_wins', 0)} resident wins, "
            f"{recent['paired'].get('ties', 0)} ties, "
            f"{recent['paired'].get('fresh_wins', 0)} fresh wins."
        ),
        "",
        f"Median expected calls per accepted direct state: {ratio:.3f} times one attempt's budget.",
        "",
        "**Noise and limitations:** Not a calibrated gate-noise or hardware result. "
        "Do not claim a general solver speedup, noisy robustness, or journal acceptance.",
        "The original static benchmark and the new temporal grid are not comparable samples.",
        "",
        "The revised IEEE Access manuscript table matches the temporal raw evidence.",
    ]
    (args.output_dir / "COMPARISON.md").write_text("\n".join(rows) + "\n")
    manifest = {name: hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
                for name in ["comparison.json", "COMPARISON.md"]}
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps({"sha256": manifest}, indent=2) + "\n"
    )
    print("\n".join(rows))


if __name__ == "__main__":
    main()
