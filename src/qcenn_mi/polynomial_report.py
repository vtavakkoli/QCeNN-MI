"""Regenerate polynomial-study tables and scientific figures from raw records."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path


def tables(data):
    rows = []
    for row in data.get("static_cases", []):
        rows.append(
            {
                key: row[key]
                for key in [
                    "size",
                    "condition",
                    "family",
                    "polynomial_degree",
                    "initial_state",
                    "steps",
                    "first_sustained_sampled_update",
                    "accuracy_passed",
                    "polynomial_order_times_first_update",
                    "shared_preprocessing_seconds",
                    "simulation_seconds",
                ]
            }
            | {"initial_residual": row["trajectory"][0]["relative_residual"]}
            | row["final"]
        )
    dynamic = [
        {
            key: row[key]
            for key in [
                "size",
                "polynomial_degree",
                "logical_depolarizing_probability",
                "simulation_seconds",
                "peak_relative_residual",
                "peak_absolute_x1_error",
                "final_hold_first_sustained_sampled_time",
            ]
        }
        | row["final"]
        for row in data.get("dynamic_cases", [])
    ]
    noise = [
        {
            key: row[key]
            for key in [
                "size",
                "condition",
                "polynomial_degree",
                "noise_convention",
                "base_probability",
                "logical_probability_per_macro_update",
            ]
        }
        | {
            "transient_residual": row["final"]["relative_residual"],
            "stationary_residual": row["stationary"]["relative_residual"],
            "stationary_equation_error": row["stationary_equation_error"],
        }
        for row in data.get("logical_noise_cases", [])
    ]
    gate = [
        {
            key: row[key]
            for key in [
                "size",
                "condition",
                "polynomial_degree",
                "steps",
                "cx_error",
                "reset_error",
                "cx",
                "depth",
                "width",
                "reset_count",
            ]
        }
        | row["final"]
        for row in data.get("gate_noise_cases", [])
    ]
    prior = [
        {
            "size": r["size"],
            "condition": r["condition"],
            "method": r["method"],
            "polynomial_degree": r["polynomial_degree"],
            "first_sustained_sampled_update": r["first_sustained_sampled_update"],
            **r["final"],
        }
        for r in data.get("prior_comparison", {}).get("cases", [])
    ]
    result = {}
    for name, values in [
        ("static", rows),
        ("prior-comparison", prior),
        ("dynamic", dynamic),
        ("logical-noise", noise),
        ("gate-noise", gate),
        ("native-resources", data.get("native_resources", [])),
        ("circuit-comparisons", data.get("circuit_comparisons", [])),
    ]:
        buffer = io.StringIO(newline="")
        keys = list(dict.fromkeys(k for row in values for k in row)) or ["empty"]
        writer = csv.DictWriter(buffer, keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)
        result[name + ".csv"] = buffer.getvalue()
    return result


def verify(output):
    output = Path(output)
    data = json.loads((output / "result.json").read_text())
    manifest = json.loads((output / "MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        path = (output / name).resolve()
        if (
            not path.is_relative_to(output.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"Polynomial evidence checksum mismatch: {name}")
    for name, expected in tables(data).items():
        if (output / name).read_text() != expected:
            raise ValueError(f"Polynomial table differs from raw records: {name}")
    inputs = [(r["input_file"], r["input_sha256"]) for r in data.get("static_cases", [])]
    inputs += [(r["input_file"], r["input_sha256"]) for r in data.get("logical_noise_cases", [])]
    inputs += [
        (r["file"], r["sha256"]) for g in data.get("dynamic_inputs", []) for r in g["snapshots"]
    ]
    for name, expected in set(inputs):
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Polynomial input checksum mismatch: {name}")
    return True


def build_report(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    data = json.loads((output / "result.json").read_text())
    for name, content in tables(data).items():
        (output / name).write_text(content)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "pdf.fonttype": 42,
            "svg.hashsalt": "polynomial-recurrent",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def save(fig, name):
        fig.tight_layout()
        for extension in ["pdf", "png", "svg"]:
            buffer = io.BytesIO()
            fig.savefig(buffer, format=extension, dpi=180)
            path = figures / f"{name}.{extension}"
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_bytes(buffer.getvalue())
            os.replace(temp, path)
        plt.close(fig)

    sizes = [
        n for n in [10, 100, 1000] if any(r["size"] == n for r in data.get("static_cases", []))
    ]
    if not sizes:
        sizes = sorted({r["size"] for r in data.get("static_cases", [])})
    if sizes:
        plotted_condition = (
            100
            if 100 in data["configuration"]["conditions"]
            else data["configuration"]["conditions"][0]
        )
        fig, axes = plt.subplots(3, len(sizes), figsize=(3 * len(sizes), 6), squeeze=False)
        for j, size in enumerate(sizes):
            for degree, style in [(0, "-"), (4, "--"), (16, ":")]:
                matches = [
                    r
                    for r in data["static_cases"]
                    if r["size"] == size
                    and r["condition"] == plotted_condition
                    and r["family"] == "controlled_nonsymmetric"
                    and r["polynomial_degree"] == degree
                    and r["initial_state"] == "reference"
                ]
                if not matches:
                    continue
                rows = matches[0]["trajectory"]
                times = [r["time"] for r in rows]
                axes[0, j].plot(times, [r["x1"] for r in rows], style, label=f"$m={degree}$")
                axes[1, j].semilogy(
                    times, [max(r["relative_residual"], 1e-15) for r in rows], style
                )
                axes[2, j].semilogy(
                    times, [max(r["absolute_x1_error"], 1e-15) for r in rows], style
                )
            if matches:
                axes[0, j].axhline(
                    rows[-1]["reference_x1"], color="black", ls="-.", lw=0.7, label="Target"
                )
            axes[0, j].set_title(f"$N={size}$, $\\kappa={plotted_condition}$")
            axes[2, j].set_xlabel("Macro updates")
        for ax, label in zip(
            axes[:, 0], ["Signed $x_1$", "Relative residual", "$|x_1-x_1^*|$"], strict=True
        ):
            ax.set_ylabel(label)
        axes[0, 0].legend(fontsize=7)
        save(fig, "polynomial-static")
    dynamic_sizes = sorted({r["size"] for r in data.get("dynamic_cases", [])})
    if dynamic_sizes:
        fig, axes = plt.subplots(
            3, len(dynamic_sizes), figsize=(3 * len(dynamic_sizes), 6), squeeze=False
        )
        for j, size in enumerate(dynamic_sizes):
            for degree, style in [(0, "-"), (4, "--"), (16, ":")]:
                matches = [
                    r
                    for r in data["dynamic_cases"]
                    if r["size"] == size
                    and r["polynomial_degree"] == degree
                    and r["logical_depolarizing_probability"] == 1e-4
                ]
                if not matches:
                    continue
                rows = matches[0]["trajectory"]
                times = [r["time"] for r in rows]
                axes[0, j].plot(times, [r["x1"] for r in rows], style, label=f"$m={degree}$")
                axes[1, j].semilogy(
                    times, [max(r["relative_residual"], 1e-15) for r in rows], style
                )
                axes[2, j].semilogy(
                    times, [max(r["absolute_x1_error"], 1e-15) for r in rows], style
                )
            if matches:
                axes[0, j].plot(
                    times, [r["reference_x1"] for r in rows], "k-.", lw=0.8, label="Target"
                )
            axes[0, j].set_title(f"$N={size}$, logical $p=10^{{-4}}$")
            axes[2, j].set_xlabel("Macro updates")
        for ax, label in zip(
            axes[:, 0], ["Signed $x_1$", "Relative residual", "$|x_1-x_1^*|$"], strict=True
        ):
            ax.set_ylabel(label)
        axes[0, 0].legend(fontsize=7)
        save(fig, "polynomial-dynamic")
    native = [
        r
        for r in data.get("native_resources", [])
        if r["status"] == "compiled" and r["condition"] == 2
    ]
    if native:
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
        for size in sorted({r["size"] for r in native}):
            rows = [r for r in native if r["size"] == size]
            axes[0].plot(
                [r["polynomial_degree"] for r in rows],
                [r["cx"] for r in rows],
                "-o",
                ms=3,
                label=f"$N={size}$",
            )
            axes[1].plot(
                [r["polynomial_degree"] for r in rows], [r["depth"] for r in rows], "-o", ms=3
            )
        axes[0].set_ylabel("Compiled CX / collision")
        axes[1].set_ylabel("Compiled depth / collision")
        for ax in axes:
            ax.set_xlabel("Polynomial degree $m$")
        axes[0].legend(fontsize=7)
        save(fig, "polynomial-resources")
    if data.get("gate_noise_cases"):
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        for error in [0.0, 1e-4, 1e-3]:
            rows = [
                r
                for r in data["gate_noise_cases"]
                if r["condition"] == 2 and r["cx_error"] == error
            ]
            ax.semilogy(
                [r["polynomial_degree"] for r in rows],
                [max(r["final"]["relative_residual"], 1e-15) for r in rows],
                "-o",
                label=f"CX depol.={error:g}",
            )
        ax.set(xlabel="Polynomial degree $m$", ylabel="Relative residual after 16 updates")
        ax.legend(fontsize=7)
        save(fig, "polynomial-gate-noise")
    summary = {
        "static_cases": len(data.get("static_cases", [])),
        "static_accuracy_passed": sum(r["accuracy_passed"] for r in data.get("static_cases", [])),
        "dynamic_cases": len(data.get("dynamic_cases", [])),
        "logical_noise_cases": len(data.get("logical_noise_cases", [])),
        "gate_noise_cases": len(data.get("gate_noise_cases", [])),
        "native_compiled": sum(r["status"] == "compiled" for r in data.get("native_resources", [])),
        "native_resource_rejected": sum(
            r["status"] == "resource_rejected" for r in data.get("native_resources", [])
        ),
        "complete_circuit_comparisons": len(data.get("circuit_comparisons", [])),
        "maximum_complete_circuit_density_error": max(
            (r["maximum_density_error"] for r in data.get("circuit_comparisons", [])), default=None
        ),
        "physical_hardware_tested": False,
    }
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "REPORT.md").write_text(
        "# Polynomial recurrent-channel study\n\n"
        "All values derive from result.json. Large static cases use exact renewal-observable "
        "simulation; dynamic cases use reduced density matrices. Native gate counts and noise "
        "simulations are confined to explicitly compiled small circuits. No physical execution "
        "or quantum runtime speedup is claimed.\n\n```json\n"
        + json.dumps(summary, indent=2)
        + "\n```\n"
    )
