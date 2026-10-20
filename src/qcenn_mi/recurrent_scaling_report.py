"""Create scientific figures and auditable tables from recurrent trajectory data."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_data(rows, fields):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


SUMMARY_FIELDS = [
    "id",
    "size",
    "condition",
    "family",
    "rhs_kind",
    "logical_depolarizing_probability",
    "steps",
    "relative_residual",
    "relative_solution_error",
    "augmented_fidelity",
    "conditional_solution_fidelity",
    "solution_probability",
    "x1",
    "reference_x1",
    "absolute_x1_error",
    "trace",
    "purity",
    "minimum_density_eigenvalue",
    "residual_target",
    "accuracy_passed",
    "numerical_integrity_passed",
    "stress_probe",
]
TRACE_FIELDS = [
    "id",
    "size",
    "condition",
    "family",
    "rhs_kind",
    "logical_depolarizing_probability",
    "step",
    "x1",
    "reference_x1",
    "absolute_x1_error",
    "relative_residual",
    "relative_solution_error",
    "augmented_fidelity",
    "conditional_solution_fidelity",
    "solution_probability",
    "reference_probability",
    "padding_probability",
    "trace",
    "purity",
]
DYNAMIC_FIELDS = [
    "size",
    "condition",
    "logical_depolarizing_probability",
    "step",
    "theta",
    "x1",
    "reference_x1",
    "absolute_x1_error",
    "relative_residual",
    "relative_solution_error",
    "augmented_fidelity",
    "conditional_solution_fidelity",
    "solution_probability",
    "trace",
]


def tables(payload):
    summaries = [{**row, **row["final"]} for row in payload["cases"]]
    traces = [{**row, **point} for row in payload["cases"] for point in row["trajectory"]]
    dynamic = [{**row, **point} for row in payload["dynamic_cases"] for point in row["trajectory"]]
    return {
        "static_summary.csv": csv_data(summaries, SUMMARY_FIELDS),
        "static_trajectories.csv": csv_data(traces, TRACE_FIELDS),
        "dynamic_trajectories.csv": csv_data(dynamic, DYNAMIC_FIELDS),
    }


def figures(output, payload):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "savefig.facecolor": "white",
            "svg.hashsalt": "qcenn-recurrent-scaling",
        }
    )
    sizes = sorted(payload["settings"]["sizes"])
    probabilities = payload["settings"]["noise_probabilities"]
    colors = {0: "#1464b4", 1e-5: "#168a83", 1e-4: "#bd9600", 1e-3: "#df6b20", 1e-2: "#ba3046"}
    dynamic_colors = colors

    def label(p):
        return "Ideal" if p == 0 else f"p={p:g}"

    def save(figure, name, caption):
        folder = output / "figures"
        folder.mkdir(exist_ok=True)
        for suffix in ("png", "svg"):
            figure.savefig(folder / f"{name}.{suffix}", dpi=210, bbox_inches="tight")
        plt.close(figure)
        return {
            "name": name,
            "caption": caption,
            "files": [f"figures/{name}.png", f"figures/{name}.svg"],
        }

    figure, axes = plt.subplots(2, len(sizes), figsize=(4.6 * len(sizes), 6.8), squeeze=False)
    for column, size in enumerate(sizes):
        selected = [
            row
            for row in payload["cases"]
            if row["size"] == size
            and row["condition"] == 2
            and row["family"] == "controlled_nonsymmetric"
            and row["rhs_kind"] == "signed_random"
            and not row["stress_probe"]
        ]
        for row in selected:
            points = row["trajectory"]
            p = row["logical_depolarizing_probability"]
            steps = [point["step"] for point in points]
            axes[0, column].plot(
                steps,
                [point["x1"] for point in points],
                color=colors[p],
                label=label(p),
                linewidth=1.8,
            )
            axes[1, column].semilogy(
                steps,
                [max(point["relative_residual"], 1e-16) for point in points],
                color=colors[p],
                linewidth=1.8,
            )
        if selected:
            axes[0, column].axhline(
                selected[0]["final"]["reference_x1"],
                color="#202b35",
                linestyle="--",
                label="Reference solution",
                linewidth=1.5,
            )
        axes[0, column].set_title(rf"${size}\times {size}$, $\kappa=2$")
        axes[0, column].set_ylabel(r"Reconstructed $x_1$")
        axes[1, column].set(xlabel="Quantum channel updates", ylabel="Relative residual")
        axes[1, column].axhline(1e-6, color="#596675", linestyle=":", linewidth=1)
        for axis in axes[:, column]:
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    figure.suptitle(
        "Recurrent inverse action: convergence from the reference initial state", y=1.01
    )
    figure.tight_layout(rect=(0, 0.075, 1, 1))
    outputs = [
        save(
            figure,
            "static_x1_convergence",
            "Reduced-density simulations, nonsymmetric condition-2 fixtures and signed random "
            "right-hand sides. x1 is reconstructed from solution/reference coherences. p is global "
            "depolarization per update, not a gate error. Noisy lines are ensemble means.",
        )
    ]

    figure, axes = plt.subplots(2, len(sizes), figsize=(4.6 * len(sizes), 6.8), squeeze=False)
    for column, size in enumerate(sizes):
        selected = [row for row in payload["dynamic_cases"] if row["size"] == size]
        if selected:
            points = selected[0]["trajectory"]
            axes[0, column].plot(
                [p["step"] for p in points],
                [p["reference_x1"] for p in points],
                color="#202b35",
                linestyle="--",
                linewidth=2,
                label="Moving solution",
            )
        for row in selected:
            points = row["trajectory"]
            p = row["logical_depolarizing_probability"]
            color = dynamic_colors.get(p, "#168a83")
            axes[0, column].plot(
                [x["step"] for x in points],
                [x["x1"] for x in points],
                color=color,
                label=label(p),
                linewidth=1.7,
            )
            axes[1, column].semilogy(
                [x["step"] for x in points],
                [max(x["absolute_x1_error"], 1e-16) for x in points],
                color=color,
                linewidth=1.7,
            )
        for axis in axes[:, column]:
            axis.axvspan(0, 128, color="#dce3e8", alpha=0.45)
            if payload["settings"]["dynamic_steps"] >= 960:
                axis.axvline(960, color="#7a8794", linestyle=":", linewidth=1.2)
            axis.set_xlim(0, payload["settings"]["dynamic_steps"])
            axis.grid(alpha=0.2)
        axes[0, column].set_title(rf"${size}\times {size}$: same data state throughout")
        axes[0, column].set_ylabel(r"$x_1$ and its moving target")
        axes[1, column].set(
            xlabel="Quantum channel updates", ylabel=r"Absolute $x_1$ tracking error"
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    figure.suptitle(
        "Time-varying matrices: initial convergence, continuous drift, and a sudden change", y=1.01
    )
    figure.tight_layout(rect=(0, 0.075, 1, 1))
    outputs.append(
        save(
            figure,
            "dynamic_x1_tracking",
            "M(k)=M(0)G(theta(k)) rotates two columns at condition number 2. The state is not "
            "reinitialized. Shading marks initial settling; the dotted line "
            "marks a jump at update 960. Matrix values are preprogrammed. The final segment holds "
            "the matrix constant. x1 uses simulator coherences, not a measured signal.",
        )
    )

    conditions = payload["settings"]["conditions"]
    figure, axes = plt.subplots(
        1, len(conditions), figsize=(6 * len(conditions), 4.2), squeeze=False
    )
    for column, condition in enumerate(conditions):
        axis = axes[0, column]
        for size, color in zip(sizes, ["#1464b4", "#168a83", "#ba3046"], strict=False):
            values = []
            for p in probabilities:
                rows = [
                    row
                    for row in payload["cases"]
                    if row["size"] == size
                    and row["condition"] == condition
                    and row["rhs_kind"] == "signed_random"
                    and row["logical_depolarizing_probability"] == p
                ]
                values.append(max(row["final"]["relative_residual"] for row in rows))
            axis.semilogy(
                range(len(probabilities)),
                values,
                "o-",
                color=color,
                label=f"{size} x {size}",
                linewidth=1.8,
            )
        axis.axhline(1e-2, color="#596675", linestyle="--", label="Noisy residual target")
        axis.set_xticks(range(len(probabilities)), [label(p) for p in probabilities])
        axis.set(
            xlabel="Global depolarization per quantum update",
            ylabel="Worst sampled residual",
            title=rf"Condition number $\kappa={condition}$",
        )
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    outputs.append(
        save(
            figure,
            "noise_sensitivity",
            "Worst final residual over the three matrix families for the signed random RHS. "
            "Different iteration budgets are used at condition numbers 2 and 10. The profiles "
            "are logical depolarization, not native CX noise or calibrated hardware data.",
        )
    )
    return outputs


def generate(output):
    output = Path(output).resolve()
    payload = json.loads((output / "result.json").read_text())
    for name, content in tables(payload).items():
        (output / name).write_text(content)
    plot_records = figures(output, payload)
    main = [row for row in payload["cases"] if not row["stress_probe"]]
    ideal = [row for row in main if row["logical_depolarizing_probability"] == 0]
    noisy = [row for row in main if row["logical_depolarizing_probability"] != 0]
    orders = ", ".join(str(n) for n in payload["settings"]["sizes"])
    lines = [
        "# Recurrent quantum inverse-action scaling study",
        "",
        f"**Execution:** exact reduced-density quantum-channel simulation at orders {orders}, "
        "with independent complete-circuit checks through order 10. This is not physical "
        "hardware execution or a native-gate simulation of the order-100/1000 circuits.",
        "",
        f"Source commit: `{payload['provenance']['source_commit']}`.",
        f"Source SHA-256: `{payload['provenance']['source_sha256']}`.",
        "",
        f"Static/noisy trajectories: {len(payload['cases'])}; dynamical trajectories: "
        f"{len(payload['dynamic_cases'])}; complete-circuit comparisons: "
        f"{len(payload['full_circuit_crosschecks'])}.",
        "",
        f"Main ideal accuracy targets passed: {sum(row['accuracy_passed'] for row in ideal)}"
        f"/{len(ideal)}. Noisy accuracy targets passed: "
        f"{sum(row['accuracy_passed'] for row in noisy)}/{len(noisy)}. "
        "Failures are retained; the numerical-integrity verdict is a separate quantity.",
        "",
        "## Final residuals",
        "",
        "Worst sampled relative residual, not a guarantee over all matrices or right-hand sides.",
        "",
        "| Order | Condition | Ideal: all three RHS | p=0.0001, random RHS | "
        "p=0.001, random RHS | p=0.01, random RHS |",
        "|---|---|---|---|---|---|",
    ]
    for size in sorted(payload["settings"]["sizes"]):
        for condition in payload["settings"]["conditions"]:
            selected = [r for r in main if r["size"] == size and r["condition"] == condition]
            values = []
            for p in (0, 1e-4, 1e-3, 1e-2):
                rows = [r for r in selected if r["logical_depolarizing_probability"] == p]
                values.append(
                    "not run"
                    if not rows
                    else f"{max(r['final']['relative_residual'] for r in rows):.3e}"
                )
            lines.append(f"| {size} | {condition} | " + " | ".join(values) + " |")
    lines += ["", "## Figures", ""]
    for record in plot_records:
        lines += [f"![{record['name']}]({record['files'][0]})", "", record["caption"], ""]
    lines += [
        "## Conditioning stress probes",
        "",
        "| Order | Condition | Updates | Final residual | Accuracy target met |",
        "|---|---|---|---|---|",
    ]
    for row in payload["cases"]:
        if row["stress_probe"]:
            lines.append(
                f"| {row['size']} | {row['condition']} | {row['steps']} | "
                f"{row['final']['relative_residual']:.3e} | {row['accuracy_passed']} |"
            )
    lines += [
        "",
        "## Interpretation and reproduction",
        "",
        "The recurrent state is a density matrix of the augmented solution/reference register. "
        "Every update is the same quantum channel as the complete collision circuit. In the "
        "spectral basis, C is diagonal, so a reduced update needs O(P²) arithmetic/storage. "
        "This reduces classical simulator work; it is not a quantum speedup result.",
        "",
        "The input matrices and RHS vectors are archived as NPZ files with digests. Classical "
        "references are used only by read-only diagnostics. The simulator does not feed an "
        "estimated solution, fitted scale, or classical residual correction into its next state.",
        "",
        "Logical noise is global depolarization on the padded data register once per update: "
        "rho -> (1-p)rho + p I/P. Its p is not comparable with a CX gate error rate. The "
        "separately supplied small native CX-noise results remain a different experiment.",
        "",
        "The absolute x1 estimate uses solution/reference coherences and known input scaling. "
        "A physical implementation would need final measurements to estimate them. Recording "
        "these values along a simulator trajectory does not imply continuous, non-destructive "
        "readout on hardware.",
        "",
        "The dynamic family changes two columns at constant conditioning. A manufactured "
        "initial solution defines the benchmark RHS; it is not supplied to the channel. "
        "Independent linear solves check the reference. A preprogrammed angle sequence is "
        "not autonomous acquisition of unknown matrices.",
        "",
        "RHS-dependent inverse action is tested on basis, uniform and signed random inputs; "
        "the experiment does not reconstruct every column of a 1000×1000 inverse or certify "
        "all nonsingular inputs. Finite-step/noisy accuracy failures remain part of the data.",
        "",
        "Run: `python -m qcenn_mi.recurrent_scaling_benchmark --config configs/full.json "
        "--output-dir NEW_DIRECTORY`. Complete data are in `result.json` and the three CSVs.",
        "",
        "## Limits",
        "",
        *(f"- {item}" for item in payload["limitations"]),
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
    files = {
        str(path.relative_to(output)): digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.json"
    }
    manifest = {
        "schema_version": 1,
        "files": files,
        "figures": plot_records,
        "source_commit": payload["provenance"]["source_commit"],
        "source_sha256": payload["provenance"]["source_sha256"],
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify(output)
    return manifest


def verify(output):
    output = Path(output).resolve()
    manifest = json.loads((output / "MANIFEST.json").read_text())
    for relative, expected in manifest["files"].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Evidence checksum mismatch: {relative}")
    payload = json.loads((output / "result.json").read_text())
    for name, expected in tables(payload).items():
        if (output / name).read_text() != expected:
            raise ValueError(f"Table disagrees with raw trajectory evidence: {name}")
    return True
