"""Data-derived tables, exact scientific plots and integrity checks."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path


def tables(data):
    def render(rows):
        buffer = io.StringIO(newline="")
        fields = list(rows[0]) if rows else ["empty"]
        writer = csv.DictWriter(buffer, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()

    return {
        "matched_comparison.csv": render(
            [
                {
                    key: row[key]
                    for key in (
                        "size",
                        "condition",
                        "method",
                        "accuracy_passed",
                        "first_sustained_sampled_time",
                    )
                }
                | {
                    key: row["final"][key]
                    for key in (
                        "relative_residual",
                        "conditional_solution_fidelity",
                        "solution_probability",
                        "reference_probability",
                    )
                }
                for row in data["cases"]
            ]
        ),
        "condition_100_noise.csv": render(
            [
                {key: row[key] for key in ("size", "steps", "logical_depolarizing_probability")}
                | {
                    "ideal_residual": row["ideal_same_budget_residual"],
                    "noisy_transient_residual": row["transient"]["relative_residual"],
                    "stationary_residual": row["stationary"]["relative_residual"],
                    "stationary_equation_error": row["stationary_equation_maximum_error"],
                }
                for row in data["condition_100_noise"]
            ]
        ),
        "spectrum_tracking.csv": render(
            [
                {key: row[key] for key in ("size", "dwell", "logical_depolarizing_probability")}
                | point
                for row in data["dynamic_cases"]
                for point in row["trajectory"]
            ]
        ),
    }


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(output):
    output = Path(output)
    manifest = json.loads((output / "MANIFEST.json").read_text())
    for name, checksum in manifest["files"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or digest(path) != checksum:
            raise ValueError(f"Evidence checksum mismatch: {name}")
    data = json.loads((output / "result.json").read_text())
    for name, content in tables(data).items():
        if (output / name).read_text() != content:
            raise ValueError(f"Table disagrees with raw results: {name}")
    for row in data["cases"]:
        if digest(output / row["input_file"]) != row["input_sha256"]:
            raise ValueError("Matched fixture checksum mismatch")
    for group in data["dynamic_inputs"]:
        for row in group["snapshots"]:
            if digest(output / row["file"]) != row["sha256"]:
                raise ValueError("Dynamic matrix checksum mismatch")
    return True


def build_report(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    data = json.loads((output / "result.json").read_text())
    for name, content in tables(data).items():
        (output / name).write_text(content)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "dissipative-comparison",
        }
    )
    figures = output / "figures"
    figures.mkdir(exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        for suffix in ("pdf", "png", "svg"):
            fig.savefig(
                figures / f"{name}.{suffix}",
                dpi=180,
                metadata={"Date": None} if suffix == "svg" else None,
            )
        plt.close(fig)

    sizes = sorted({r["size"] for r in data["cases"]})
    conditions = sorted({r["condition"] for r in data["cases"]})
    names = {
        "discrete": "Discrete recurrence",
        "matched_continuous": "Matched-F continuous",
        "shang_rate_normalized": "Shang, rate normalized",
    }
    styles = {"discrete": "-", "matched_continuous": "--", "shang_rate_normalized": ":"}
    fig, axes = plt.subplots(
        len(sizes), len(conditions), figsize=(10.2, 2.25 * len(sizes)), squeeze=False
    )
    for i, size in enumerate(sizes):
        for j, condition in enumerate(conditions):
            ax = axes[i, j]
            for row in data["cases"]:
                if row["size"] == size and row["condition"] == condition:
                    ax.semilogy(
                        [p["time"] for p in row["trajectory"]],
                        [max(p["relative_residual"], 1e-14) for p in row["trajectory"]],
                        styles[row["method"]],
                        label=names[row["method"]],
                    )
            ax.axhline(1e-6, color="gray", linestyle="--", linewidth=0.6)
            ax.set(
                title=f"N={size}, condition={condition}",
                xlabel="Normalized model time / updates",
                ylabel="Relative residual",
                ylim=(1e-14, 2),
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "dissipative_comparison")

    if data["dynamic_cases"]:
        dwell = max(row["dwell"] for row in data["dynamic_cases"])
        fig, axes = plt.subplots(3, len(sizes), figsize=(10.2, 7.0), squeeze=False)
        for i, size in enumerate(sizes):
            selected = [
                r for r in data["dynamic_cases"] if r["size"] == size and r["dwell"] == dwell
            ]
            base = selected[0]["trajectory"]
            t = [p["time"] for p in base]
            axes[0, i].step(t, [p["condition"] for p in base], where="post", label="condition")
            axes[0, i].set(yscale="log", title=f"N={size}", ylabel="Condition number")
            axes[1, i].plot(t, [p["reference_x1"] for p in base], "k--", label="target")
            for row in selected:
                points = row["trajectory"]
                label = f"p={row['logical_depolarizing_probability']:g}"
                axes[1, i].plot([p["time"] for p in points], [p["x1"] for p in points], label=label)
                axes[2, i].semilogy(
                    [p["time"] for p in points],
                    [max(p["relative_residual"], 1e-14) for p in points],
                    label=label,
                )
            axes[1, i].set(ylabel="Absolute-scale x1")
            axes[2, i].set(ylabel="Relative residual", xlabel="Channel updates")
        axes[1, -1].legend(fontsize=7)
        save(fig, "variable_spectrum_tracking")

    if data["condition_100_noise"]:
        fig, ax = plt.subplots(figsize=(6.7, 3.8))
        for size in sizes:
            rows = [r for r in data["condition_100_noise"] if r["size"] == size]
            if rows:
                probabilities = [r["logical_depolarizing_probability"] for r in rows]
                ax.loglog(
                    probabilities,
                    [r["stationary"]["relative_residual"] for r in rows],
                    "o-",
                    label=f"N={size}: stationary",
                )
                ax.loglog(
                    probabilities,
                    [r["transient"]["relative_residual"] for r in rows],
                    "x--",
                    label=f"N={size}: budget {rows[0]['steps']}",
                )
        ax.axhline(0.01, color="gray", linestyle=":", label="noisy target")
        ax.set(xlabel="Logical depolarization per update p", ylabel="Relative coherence residual")
        ax.legend(fontsize=7, ncol=2)
        save(fig, "condition_100_diagnosis")

    lines = [
        "# Matched dissipative comparison and variable-spectrum tracking",
        "",
        "All results are reduced-density simulations. No physical gate noise was tested.",
        "Shang's generator is divided by two to bound the total jump rate by one.",
        "Model time and update counts are not hardware/query costs.",
        "",
        "| N | condition | method | residual | first sustained sampled time |",
        "|---:|---:|---|---:|---:|",
    ]
    for row in data["cases"]:
        lines.append(
            f"| {row['size']} | {row['condition']} | {row['method']} | "
            f"{row['final']['relative_residual']:.6g} | "
            f"{row['first_sustained_sampled_time']} |"
        )
    lines += [
        "",
        "Condition-100 transient error and exact logical-noise stationary error are",
        "reported separately in condition_100_noise.csv. Longer ideal N=10 evolution",
        "is a small-superoperator computation, not a native quantum execution.",
        "",
        "Dynamic matrices change every singular value, norm and condition number.",
        "The same density state survives every snapshot; only its simulator basis changes.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "schema_version": 1,
        "files": {
            str(p.relative_to(output)): digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "MANIFEST.json"
        },
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify(output)
