"""Generate and independently verify data-driven tables, plots, and final reports.

No manuscript sources are required. Historical and current runs are never pooled.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import shutil
from pathlib import Path

TABLES = {
    "native-resources": (
        "hardware/result.json",
        [
            "size",
            "required_qubits",
            "native_action_depth",
            "native_action_operations",
            "native_ideal_residual",
        ],
    ),
    "native-noise": (
        "hardware/result.json",
        [
            "size",
            "profile",
            "shots",
            "accepted",
            "observed_echo_return",
            "echo_return_95ci",
            "quality_gate",
        ],
    ),
    "precision": (
        "precision-1000/result.json",
        [
            "size",
            "family",
            "condition_number",
            "degree",
            "operator_relative_residual_2",
            "minimum_success_probability_all_inputs",
        ],
    ),
    "refresh-accuracy": (
        "temporal-refresh/result.json",
        [
            "size",
            "condition",
            "family",
            "update_time",
            "noise_to_gap_ratio",
            "refresh_age",
            "left_residual",
            "right_residual",
            "tracking_bound_eq24",
        ],
    ),
    "refresh-intervals": (
        "temporal-refresh/result.json",
        [
            "size",
            "condition",
            "family",
            "noise_to_gap_ratio",
            "target",
            "sufficient_refresh_age_eq25",
            "empirical_maximum_sampled_feasible_age",
            "sufficient_condition_validated",
        ],
    ),
    "noise-controls": (
        "noise-verification/result.json",
        [
            "size",
            "decomposition",
            "optimization_level",
            "profile",
            "noise_scale",
            "control",
            "shots",
            "accepted",
            "observed_echo_return",
            "echo_return_95ci",
            "effective_good_output_probability",
            "quality_gate",
        ],
    ),
    "optimization-resources": (
        "noise-verification/result.json",
        [
            "size",
            "decomposition",
            "optimization_level",
            "native_depth",
            "single_qubit_gates",
            "two_qubit_gates",
            "synthesis_and_transpile_seconds",
            "native_ideal_residual",
            "passed",
        ],
    ),
    "refresh-latency": (
        "temporal-refresh/result.json",
        [
            "size",
            "condition",
            "family",
            "noise_to_gap_ratio",
            "update_time",
            "encoding_and_polynomial_seconds",
            "isolated_svd_seconds_not_added_twice",
            "native_synthesis_seconds",
            "native_transpile_seconds",
            "native_gate_simulation_seconds",
            "physical_acquisition_seconds",
        ],
    ),
    "solver-baselines": (
        "baselines/result.json",
        [
            "size",
            "condition",
            "family",
            "method",
            "target_tolerance",
            "median_solve_seconds",
            "median_end_to_end_seconds",
            "end_to_end_simulator_seconds",
            "relative_residual",
        ],
    ),
    "conditioning-resources": (
        "resources/result.json",
        [
            "size",
            "condition",
            "target_tolerance",
            "status",
            "degree",
            "raw_qubits",
            "encoded_operator_residual",
            "minimum_success_probability",
            "native_cx",
            "native_depth",
        ],
    ),
    "operator-correctness": (
        "operator-audit/result.json",
        [
            "size",
            "family",
            "left_operator_residual",
            "right_operator_residual",
            "analytic_residual_bound",
            "native_depth",
            "native_cx",
            "passed",
        ],
    ),
    "recurrent-convergence": (
        "recurrent-quantum/result.json",
        [
            "size",
            "family",
            "condition",
            "steps",
            "qubits",
            "augmented_fidelity",
            "conditional_solution_fidelity",
            "coherence_relative_residual",
            "solution_probability",
            "augmented_infidelity_upper_bound",
            "certificate_passed",
        ],
    ),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def rows_for(root, table):
    source, _ = TABLES[table]
    data = read_json(Path(root) / source, {})
    if table == "native-noise":
        return [
            {"size": row["size"], **profile}
            for row in data.get("cases", [])
            for profile in row["noise"]
        ]
    if table == "refresh-accuracy":
        return data.get("samples", [])
    if table == "refresh-intervals":
        return data.get("refresh_intervals", [])
    if table == "noise-controls":
        return data.get("pooled_results", [])
    if table == "optimization-resources":
        return data.get("resources", [])
    if table == "refresh-latency":
        return data.get("timings", [])
    return data.get("cases", [])


def csv_text(root, table):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    fields = TABLES[table][1]
    writer.writerow(fields)
    for row in rows_for(root, table):
        writer.writerow(
            [
                json.dumps(row.get(key), sort_keys=True, allow_nan=False)
                if isinstance(row.get(key), (list, dict))
                else row.get(key)
                for key in fields
            ]
        )
    return buffer.getvalue()


def plot_figures(root, run_id):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "svg.hashsalt": run_id,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    root = Path(root)
    figures = []

    def save(fig, name, inputs, caption):
        fig.tight_layout()
        for extension in ("svg", "png"):
            file = root / "figures" / f"{name}.{extension}"
            fig.savefig(
                file,
                dpi=160,
                metadata={"Date": None}
                if extension == "svg"
                else {"Software": "QCeNN-MI evidence report"},
            )
        figures.append(
            {
                "id": name,
                "files": [f"figures/{name}.svg", f"figures/{name}.png"],
                "inputs": inputs,
                "caption": caption,
            }
        )
        plt.close(fig)

    rows = rows_for(root, "recurrent-convergence")
    if rows:
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        for size in sorted({row["size"] for row in rows}):
            selected = [
                row
                for row in rows
                if row["size"] == size and row["family"] == "controlled_nonsymmetric"
            ]
            if selected:
                ax.semilogy(
                    [row["steps"] for row in selected],
                    [max(1 - row["augmented_fidelity"], 1e-15) for row in selected],
                    "o-",
                    label=f"order {size}",
                )
        ax.set(xlabel="Quantum channel updates", ylabel="Augmented-state infidelity")
        if ax.lines:
            ax.legend()
        save(
            fig,
            "recurrent-convergence",
            ["recurrent-quantum/result.json"],
            "Ideal complete reset-circuit convergence for the configured nonsymmetric "
            "fixtures. Coherent environment circuits are cross-checked separately. "
            "Dense encoding and final simulator diagnostics do not establish hardware quality.",
        )

    rows = rows_for(root, "refresh-accuracy")
    if rows:
        n = max(r["size"] for r in rows)
        condition = min(r["condition"] for r in rows)
        update = min(r["update_time"] for r in rows)
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        for family in sorted({r["family"] for r in rows}):
            selected = sorted(
                [
                    r
                    for r in rows
                    if r["size"] == n
                    and r["condition"] == condition
                    and r["family"] == family
                    and r["update_time"] == update
                    and r["noise_to_gap_ratio"] == 0
                    and r["refresh_age"] > 0
                ],
                key=lambda r: r["refresh_age"],
            )
            ax.loglog(
                [r["refresh_age"] for r in selected],
                [max(r["left_residual"], r["right_residual"]) for r in selected],
                marker="o",
                markersize=3,
                label=family,
            )
            if family == "fast_nonsymmetric":
                ax.loglog(
                    [r["refresh_age"] for r in selected],
                    [r["tracking_bound_eq24"] for r in selected],
                    "--",
                    color="#555555",
                    label="Eq. (24) bound: fast trajectory",
                )
        ax.axhline(1e-3, color="#b34b3b", linestyle=":", label="residual target 1e-3")
        ax.set(
            xlabel="Refresh age (dimensionless model time)",
            ylabel="Two-sided operator residual",
            title=f"Held-operator accuracy: N={n}, initial condition={condition:g}",
        )
        ax.legend(fontsize=8)
        save(
            fig,
            "refresh-accuracy",
            ["temporal-refresh/result.json"],
            (
                "Measured accuracy includes both useful refresh intervals and "
                "intentional stale-operator failures. Model time is not "
                "wall-clock latency."
            ),
        )
    rows = rows_for(root, "noise-controls")
    if rows:
        chosen = [
            r
            for r in rows
            if r["optimization_level"] == max(x["optimization_level"] for x in rows)
            and r["decomposition"] == "input_fused"
            and r["noise_scale"] == max(x["noise_scale"] for x in rows)
        ]
        fig, ax = plt.subplots(figsize=(9, max(4, len(chosen) * 0.28)))
        for i, row in enumerate(chosen):
            value = row["observed_echo_return"]
            if value is None:
                continue
            lo, hi = row["echo_return_95ci"]
            ax.errorbar(
                value,
                i,
                xerr=[[max(0.0, value - lo)], [max(0.0, hi - value)]],
                fmt="o",
                color="#176b81",
                capsize=2,
            )
        ax.set_yticks(
            range(len(chosen)),
            [f"N={r['size']} {r['profile'].split('_')[0]} / {r['control']}" for r in chosen],
            fontsize=8,
        )
        ax.axvline(0.9, color="#b34b3b", linestyle="--")
        ax.set(
            xlim=(0, 1.02),
            xlabel="Conditional reference-echo return with 95% Wilson interval",
            title="Solver, verifier and readout controls",
        )
        save(
            fig,
            "noise-controls",
            ["noise-verification/result.json"],
            (
                "Echo return includes the selected verifier and readout errors;"
                " it is not a direct state-fidelity measurement. Seeds are "
                "pooled only within identical configurations."
            ),
        )
    rows = [r for r in rows_for(root, "conditioning-resources") if r["status"] == "measured"]
    if rows:
        n = max(r["size"] for r in rows)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
        for tolerance in sorted({r["target_tolerance"] for r in rows}):
            chosen = sorted(
                [r for r in rows if r["size"] == n and r["target_tolerance"] == tolerance],
                key=lambda r: r["condition"],
            )
            for ax, key, label in zip(
                axes,
                ["degree", "raw_qubits", "minimum_success_probability"],
                ["Polynomial degree", "Raw qubits", "Worst-input success probability"],
                strict=True,
            ):
                ax.semilogx(
                    [r["condition"] for r in chosen],
                    [r[key] for r in chosen],
                    "o-",
                    label=f"target {tolerance:g}",
                )
                ax.set(xlabel="Condition number", ylabel=label)
                from matplotlib.ticker import NullFormatter, ScalarFormatter

                ax.set_xticks(sorted({r["condition"] for r in chosen}))
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.xaxis.set_minor_formatter(NullFormatter())
                if key == "minimum_success_probability":
                    ax.set_yscale("log")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Conditioning control at N={n}; resource-limited cases retained in CSV")
        save(
            fig,
            "conditioning-resources",
            ["resources/result.json"],
            (
                "Diagonal conditioning controls separate degree, qubit and "
                "repetition costs. Dense routed gate counts are separately "
                "reported by the native hardware stage."
            ),
        )
    rows = rows_for(root, "solver-baselines")
    if rows:
        n = max(r["size"] for r in rows)
        condition = min(r["condition"] for r in rows)
        family = sorted({r["family"] for r in rows})[0]
        selected = [
            r
            for r in rows
            if r["size"] == n and r["condition"] == condition and r["family"] == family
        ]
        labels = [
            r["method"].replace("chebyshev_lcu_structured_simulator", "LCU emulator")
            + (f" {r['target_tolerance']:g}" if "target_tolerance" in r else "")
            for r in selected
        ]
        times = [
            r.get("median_end_to_end_seconds", r.get("end_to_end_simulator_seconds"))
            for r in selected
        ]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.barh(labels, times, color=["#176b81" if "LCU" not in x else "#b8783b" for x in labels])
        ax.set_xscale("log")
        ax.set(
            xlabel="Measured software wall time (seconds)",
            title=(
                "Classical solve and emulator costs: N="
                f"{n}"
                ", condition="
                f"{condition:g}"
                "; output tasks differ"
            ),
        )
        save(
            fig,
            "solver-costs",
            ["baselines/result.json"],
            (
                "Classical times measure a complete vector solve; emulator "
                "times include dense encoding and structured simulation. These "
                "are different output tasks and establish no quantum speedup "
                "ratio."
            ),
        )
    return figures


def coverage(root, validation):
    root = Path(root)

    def available(name):
        data = read_json(root / name / "result.json", {})
        return bool(data) and data.get("passed", data.get("native_validation_passed", False))

    return [
        {
            "item": "P0-1",
            "status": "implemented"
            if (root / "historical-reconciliation.json").exists()
            else "missing",
            "evidence": "historical-reconciliation.json; publication-evidence/manifest.json",
            ("scope"): (
                "Both historical runs retained separately; Table III/IV "
                "discrepancies reconciled, no causal attribution invented."
            ),
        },
        {
            "item": "P0-2",
            ("status"): (
                "CI release workflow provided; release attachment requires a successful tagged run"
            ),
            "evidence": "validation.json; SHA256SUMS.json; PACKAGE_SHA256; source CITATION.cff",
            ("scope"): (
                "Only actual CI-generated assets are attached to the candidate "
                "tag; local execution is explicitly labeled."
            ),
        },
        {
            "item": "P1-1",
            "status": "implemented" if available("temporal-refresh") else "not run or failed",
            "evidence": "temporal-refresh/result.json; tables/refresh-intervals.csv",
            ("scope"): (
                "Positive and failed accuracy regimes, Eq. (24)/(25), "
                "acquisition error, drift and repetition cost; physical "
                "acquisition and deadlines not measured."
            ),
        },
        {
            "item": "P1-2",
            "status": "implemented for configured small hardware-ready orders"
            if available("noise-verification")
            else "not run or failed",
            "evidence": "noise-verification/result.json; hardware/audit/",
            ("scope"): (
                "Multiple seeds, noise grid, separate solver/verifier/readout "
                "controls and intervals. Order-100 statistical power remains "
                "unresolved at the baseline 64 shots."
            ),
        },
        {
            "item": "P1-3",
            "status": "implemented" if available("baselines") else "not run or failed",
            "evidence": "baselines/result.json",
            ("scope"): (
                "SciPy, LU, QR and SVD timings; HHL/QSVT analytical "
                "access-model comparison only; no quantum speedup ratio."
            ),
        },
        {
            "item": "P1-4",
            "status": "implemented with explicit native-synthesis budgets"
            if available("resources")
            else "not run or failed",
            "evidence": "resources/result.json; hardware/result.json; validation.json",
            ("scope"): (
                "Conditioning through 10000, preprocessing, basis/angle "
                "storage, attempts, readout and routing. Native counts omitted "
                "where budgets prevent synthesis."
            ),
        },
        {
            "item": "P1-5",
            "status": "implemented" if available("operator-audit") else "not run or failed",
            "evidence": "operator-audit/result.json; unit-tests.xml",
            ("scope"): (
                "Input-independent native full-state checks, unseen RHS, both "
                "residuals, failure sectors, singularity and runtime "
                "reference-solve guards."
            ),
        },
        {
            "item": "P2",
            "status": "implemented",
            ("evidence"): (
                "configuration.json; validation.json; "
                "publication-evidence/manifest.json; source "
                "docs/publication-evidence.md"
            ),
            ("scope"): (
                "Explicit profiles, CLI settings, schemas, failure semantics, "
                "provenance, preserved citation license, and an offline report."
            ),
        },
        {
            "item": "Recurrent quantum state updates",
            "status": "small-circuit simulator prototype passed"
            if available("recurrent-quantum")
            else "not run or failed",
            "evidence": "recurrent-quantum/result.json; recurrent-quantum/*.qpy",
            "scope": (
                "Complete coherent/reset quantum circuits without classical iterate updates; "
                "RHS-dependent residual-channel construction, separate from Chebyshev inverse "
                "action. Dense preprocessing, synthetic CX-only noise and preprogrammed "
                "snapshots do not establish live tracking or hardware performance."
            ),
        },
        {
            "item": "Physical device validation",
            "status": "not run",
            ("evidence"): (
                "No physical backend, calibration snapshot, job IDs or hardware counts supplied"
            ),
            ("scope"): (
                "Hardware readiness and synthetic emulation are not physical performance evidence."
            ),
        },
    ]


def generate(root):
    root = Path(root).resolve()
    validation = read_json(root / "validation.json")
    if validation is None:
        raise ValueError("validation.json is required")
    source_root = Path.cwd()
    legacy = source_root / "publication-evidence/historical-reconciliation.json"
    if legacy.exists():
        shutil.copyfile(legacy, root / "historical-reconciliation.json")
    for name in ("tables", "figures", "publication-evidence"):
        (root / name).mkdir(exist_ok=True)
    run_id = validation["publication_run_id"]
    table_manifest = []
    for table, (input_file, fields) in TABLES.items():
        file = root / "tables" / f"{table}.csv"
        file.write_text(csv_text(root, table))
        table_manifest.append(
            {
                "id": table,
                "file": str(file.relative_to(root)),
                "input": input_file,
                "fields": fields,
                "rows": len(rows_for(root, table)),
                "sha256": digest(file),
                "input_sha256": digest(root / input_file) if (root / input_file).exists() else None,
            }
        )
    figures = plot_figures(root, run_id)
    items = coverage(root, validation)
    temporal = read_json(root / "temporal-refresh/result.json", {})
    noise = read_json(root / "noise-verification/result.json", {})
    operator = read_json(root / "operator-audit/result.json", {})
    precision = read_json(root / "precision-1000/result.json", {})
    recurrent = read_json(root / "recurrent-quantum/result.json", {})
    findings = {
        "temporal_positive_accuracy_cases": temporal.get("positive_accuracy_cases"),
        "intentional_stale_operator_failures": temporal.get("intentional_accuracy_failures"),
        "seeded_noise_samples": len(noise.get("samples", [])) if noise else None,
        "failed_pooled_noisy_quality_gates": sum(
            not r["quality_gate"] for r in noise.get("pooled_results", [])
        )
        if noise
        else None,
        "strict_native_operator_audits": len(operator.get("cases", [])) if operator else None,
        "optimization_comparisons_failing_full_state_target": sum(
            not c["meets_strict_full_state_target"]
            for r in operator.get("cases", [])
            for c in r.get("optimization_comparison", [])
        )
        if operator
        else None,
        "worst_1000_operator_residual": max(
            (r["operator_relative_residual_2"] for r in precision.get("cases", [])), default=None
        ),
        "recurrent_quantum_circuit_cases": len(recurrent.get("cases", []))
        if recurrent
        else None,
        "recurrent_quantum_classical_iterate_updates": recurrent.get("classical_iterate_updates"),
        "recurrent_quantum_synthetic_quality_passed": recurrent.get(
            "all_synthetic_quality_gates_passed"
        ),
    }
    payload = {
        "schema_version": 1,
        "publication_run_id": run_id,
        "software_validation_passed_before_report": validation["passed"],
        "synthetic_noise_quality_passed": validation.get("all_noisy_quality_gates_passed"),
        "physical_hardware_tested": False,
        "provenance": validation["provenance"],
        "configuration": read_json(root / "configuration.json", {}),
        "coverage": items,
        "findings": findings,
        "tables": table_manifest,
        "figures": figures,
        "historical_reconciliation": read_json(root / "historical-reconciliation.json", {}),
        "image_provenance_available": (root / "base-image.json").exists(),
        "paper_sources_in_repository": False,
    }
    (root / "final-report.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = [
        "# QCeNN-MI final validation report",
        "",
        f"Run ID: `{run_id}`",
        "",
        (
            "Source: `"
            f"{validation['provenance']['source_commit']}"
            "`; source SHA-256: `"
            f"{validation['provenance']['source_sha256']}"
            "`."
        ),
        "",
        (
            "Execution context: "
            f"{validation['provenance']['execution_context']}"
            ". Software stages before report: "
            f"{('PASS' if validation['passed'] else 'FAIL / INCOMPLETE')}"
            "."
        ),
        "",
        (
            "Synthetic-noise quality: "
            f"{validation.get('all_noisy_quality_gates_passed')}"
            ". Physical-device validation: NOT RUN."
        ),
        "",
        (
            "The software verdict checks implementation and declared "
            "numerical acceptance criteria. It does not turn failed noise "
            "gates into successful hardware results."
        ),
        "",
        "## Requested implementation and evidence",
        "",
        "| Item | Status | Evidence and limits |",
        "|---|---|---|",
    ]
    for item in items:
        lines.append(f"| {item['item']} | {item['status']} | {item['evidence']}. {item['scope']} |")
    lines += [
        "",
        "## Historical discrepancy",
        "",
        (
            "The manuscript depth sequence 66/138/1032/41033 and order-100 "
            "accepted count 35/64 (echo 10/35) match the clean e6769dbe "
            "benchmark. The original public archive retains "
            "72/142/1033/41024 and 34/64 (echo 8/34). Both are separate "
            "simulator runs; neither is averaged or overwritten. Matching "
            "package versions do not prove identical source, compiled "
            "circuits or stochastic execution. The older archive lacks a "
            "complete immutable run manifest, so a unique root cause cannot"
            " be established."
        ),
        "",
        "## Measured results",
        "",
    ]
    lines += (
        ["## Key findings", ""] + [f"- {key}: {value}" for key, value in findings.items()] + [""]
    )
    html_sections = []
    for item in table_manifest:
        table = item["id"]
        fields = item["fields"]
        rows = rows_for(root, table)
        lines += [
            f"### {table}",
            "",
            f"{len(rows)} rows; source `{item['input']}`; CSV SHA-256 `{item['sha256']}`.",
            "",
            "| " + " | ".join(fields) + " |",
            "|" + "---|" * len(fields),
        ]
        for row in rows[:8]:
            lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
        lines += ["", "Full values and raw audit files are in the validation bundle.", ""]
        csv_data = base64.b64encode((root / item["file"]).read_bytes()).decode()
        heading = (
            "<h2>"
            f"{html.escape(table)}"
            "</h2><p>"
            f"{len(rows)}"
            " rows. <a download='"
            f"{table}"
            ".csv' href='data:text/csv;base64,"
            f"{csv_data}"
            "'>Download complete CSV</a></p>"
        )
        headers = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
        body = "".join(
            "<tr>"
            + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields)
            + "</tr>"
            for row in rows[:8]
        )
        html_sections.append(
            heading
            + (
                "<div class='table-scroll'><table><thead><tr>"
                f"{headers}"
                "</tr></thead><tbody>"
                f"{body}"
                "</tbody></table></div>"
            )
        )
    (root / "final-report.md").write_text("\n".join(lines) + "\n")
    figure_html = ""
    for figure in figures:
        data = base64.b64encode((root / figure["files"][0]).read_bytes()).decode()
        figure_html += (
            "<figure><img alt='"
            f"{html.escape(figure['id'])}"
            "' src='data:image/svg+xml;base64,"
            f"{data}"
            "'><figcaption>"
            f"{html.escape(figure['caption'])}"
            "</figcaption></figure>"
        )
    findings_html = (
        "<h2>Key findings from this run</h2><ul>"
        + "".join(
            f"<li><b>{html.escape(key.replace('_', ' '))}:</b> "
            f"{html.escape(str(value)) if value is not None else 'not run'}</li>"
            for key, value in findings.items()
        )
        + "</ul>"
    )
    cover = "".join(
        (
            "<tr><td>"
            f"{html.escape(item['item'])}"
            "</td><td>"
            f"{html.escape(item['status'])}"
            "</td><td>"
            f"{html.escape(item['scope'])}"
            "</td></tr>"
        )
        for item in items
    )
    css = (
        "body{font:15px/1.6 "
        "system-ui,sans-serif;color:#213047;background:#f0f3f6;margin:0}main{max-width:1180px;margin:32px"
        " auto;padding:36px;background:white;border-top:6px solid "
        "#176b81}h1,h2{color:#153b51}h1{font-size:32px}code{overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:left;padding:9px;border-bottom:1px"
        " solid "
        "#d7e0e7;vertical-align:top}th{background:#eaf1f5}.table-scroll{overflow:auto}figure{margin:28px"
        " "
        "0}img{max-width:100%;height:auto}figcaption{color:#52667a;font-size:13px}.status{padding:18px;background:#eaf1f5;border-left:4px"
        " solid #176b81}a{color:#176b81}"
    )
    page = (
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta "
        "name='viewport' "
        "content='width=device-width,initial-scale=1'><title>QCeNN-MI "
        "final validation report</title><style>"
        f"{css}"
        "</style><main><h1>QCeNN-MI final validation report</h1><p>Run "
        "<code>"
        f"{run_id}"
        "</code><br>Source <code>"
        f"{validation['provenance']['source_commit']}"
        "</code><br>Context: "
        f"{html.escape(validation['provenance']['execution_context'])}"
        "</p><div class='status'><b>Software: "
        f"{('PASS' if validation['passed'] else 'FAIL / INCOMPLETE')}"
        "</b> &nbsp; Synthetic noisy quality: "
        f"{validation.get('all_noisy_quality_gates_passed')}"
        " &nbsp; Physical hardware: NOT RUN.<p>Implementation checks, "
        "numerical fidelity, noise quality, and physical validation are"
        " separate verdicts.</p></div><h2>Checklist coverage and "
        "limits</h2><table><tr><th>Item</th><th>Status</th><th>Measured"
        " scope and remaining limits</th></tr>"
        f"{cover}"
        "</table><h2>Historical evidence</h2><p>The manuscript hardware"
        " values match the clean e6769dbe run. The older public archive"
        " remains separate. Both raw records and the exact "
        "discrepancies are preserved; their statistics are never "
        "pooled. See historical-reconciliation.json in the evidence "
        "bundle.</p>"
        f"{figure_html}"
        f"{''.join(html_sections)}"
        "<h2>Reproduction and scope</h2><p>The complete bundle includes"
        " source, locked dependencies, configuration, raw counts, QPY "
        "circuits, logs, tables, plots, checksums and a per-file "
        "evidence manifest. Run <code>python -m "
        "qcenn_mi.evidence_report --results PATH --verify-only</code> "
        "from that exact source checkout to detect altered data or "
        "table metrics. HHL/QSVT comparisons are analytical; no quantum"
        " speedup or physical-device result is claimed. Dense "
        "preprocessing, postselection and readout costs remain "
        "explicit.</p></main></html>"
    )
    page = page.replace(
        "<h2>Checklist coverage and limits</h2>",
        findings_html + "<h2>Checklist coverage and limits</h2>",
    )
    (root / "final-report.html").write_text(page)
    files = {}
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and path.name
            not in {
                "validation.json",
                "summary.md",
                "manifest.json",
                "SHA256SUMS.json",
                "PACKAGE_SHA256",
                "QCeNN_MI_Validation_Bundle.zip",
            }
            and "logs" not in path.relative_to(root).parts
        ):
            files[str(path.relative_to(root))] = digest(path)
    manifest = {
        "schema_version": 1,
        "publication_run_id": run_id,
        "source_commit": validation["provenance"]["source_commit"],
        "source_sha256": validation["provenance"]["source_sha256"],
        "producer": "src/qcenn_mi/evidence_report.py",
        "producer_sha256": digest(__file__),
        "tables": table_manifest,
        "figures": figures,
        "files": files,
        "mutable_aggregate_files_excluded": ["validation.json", "summary.md", "logs/"],
    }
    (root / "publication-evidence/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify(root)
    return payload


def verify(root):
    root = Path(root).resolve()
    manifest = read_json(root / "publication-evidence/manifest.json")
    if not manifest or manifest.get("schema_version") != 1:
        raise ValueError("Missing or invalid evidence manifest")
    if manifest.get("producer_sha256") and manifest["producer_sha256"] != digest(__file__):
        raise ValueError("Evidence producer source differs from the frozen run")
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Evidence checksum mismatch: {relative}")
    for item in manifest["tables"]:
        if (root / item["file"]).read_text() != csv_text(root, item["id"]):
            raise ValueError(f"Table metrics disagree with raw JSON: {item['id']}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.results)
    else:
        generate(args.results)
    print("Final report and evidence integrity: PASS")


if __name__ == "__main__":
    main()
