"""Build evidence-derived IEEE and EuroHPC PDFs without fabricated measurements."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#165D8C", "#C45A21", "#287D5A", "#76569A"]


def tex(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(c, c) for c in str(value))


def number(value, digits=2):
    if value == 0:
        return "$0$"
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"${mantissa}\!\times\!10^{{{int(exponent)}}}$"


def load_results(results):
    results = Path(results)
    names = (
        "assumption-a1",
        "a2-a4",
        "scaling",
        "coherent",
        "coherent-large",
        "compact",
        "precision-1000",
        "theory",
        "hardware",
    )
    data = {name: json.loads((results / name / "result.json").read_text()) for name in names}
    data["validation"] = json.loads((results / "validation.json").read_text())
    # A failed document build can be retried without rerunning numerical work.
    stages = [r for r in data["validation"]["stages"] if r["name"] != "publication"]
    expected = {"compile", "lint", "unit-tests", *names}
    if {r["name"] for r in stages} != expected or not all(r["passed"] for r in stages):
        raise ValueError("Publication requires a completed passing software benchmark suite")
    data["validation"]["stages"] = stages
    tree = ET.parse(results / "unit-tests.xml").getroot()
    data["test_count"] = sum(int(s.attrib.get("tests", 0)) for s in tree.iter("testsuite"))
    data["benchmark_bytes"] = sum(
        p.stat().st_size
        for p in results.rglob("*")
        if p.is_file() and "publication" not in p.relative_to(results).parts
    )
    return data


def write_tables(data, generated, metadata):
    precision = data["precision-1000"]["cases"]
    hardware = data["hardware"]["cases"]
    validation = data["validation"]
    worst = max(r["operator_relative_residual_2"] for r in precision)
    inverse = max(r["relative_inverse_error_frobenius"] for r in precision)
    times = sum(r.get("wall_seconds", 0) for r in validation["stages"])
    cpu = sum(r.get("cpu_seconds", 0) for r in validation["stages"])
    peak = max(r.get("peak_child_rss_bytes", 0) for r in validation["stages"]) / 1024**2
    macros = {
        "WorstOperatorResidual": number(worst),
        "WorstInverseError": number(inverse),
        "TestCount": str(data["test_count"]),
        "TheoryCaseCount": str(len(data["theory"]["cases"])),
        "PrecisionCaseCount": str(len(precision)),
        "SuiteWallSeconds": f"{times:.1f}",
        "SuiteCPUSeconds": f"{cpu:.1f}",
        "SuitePeakMiB": f"{peak:.1f}",
        "SourceCommit": tex(validation["provenance"]["source_commit"][:12]),
        "SourceFingerprint": validation["provenance"]["source_sha256"],
        "RunContext": tex(validation["provenance"]["execution_context"]),
        "PythonVersion": tex(validation["provenance"]["python"]),
        "RunDate": tex(validation["started_utc"][:10]),
        "ReportDate": tex(metadata.get("report_date", validation["started_utc"][:10])),
        "ArtifactMiB": f"{data['benchmark_bytes'] / 1024**2:.1f}",
        "ApplicationID": tex(metadata.get("application_id", "Not supplied in this run")),
        "AllocationNodeHours": tex(metadata.get("emulator_node_hours", "Not supplied")),
        "AllocationStorageGB": tex(metadata.get("storage_gb", "Not supplied")),
        "AllocationDays": tex(metadata.get("duration_days", "Not supplied")),
        "AllocationSource": tex(metadata.get("allocation_source", "No applicant notice supplied")),
    }
    for package, value in validation["provenance"]["packages"].items():
        macros["Version" + package.replace("-", "").title()] = tex(value)
    (generated / "metrics.tex").write_text(
        "\n".join(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items()) + "\n"
    )
    lines = []
    labels = {
        "dense_spd": "Dense SPD",
        "tridiagonal_spd": "Tridiagonal SPD",
        "dense_nonsymmetric": "Dense nonsym.",
        "signed_indefinite": "Signed indef.",
        "controlled_spd": "Controlled SPD",
        "controlled_nonsymmetric": "Controlled nonsym.",
        "controlled_indefinite": "Controlled indef.",
    }
    for row in precision:
        lines.append(
            f"{labels[row['family']]} & {row['condition_number']:.3g} & {row['degree']} & "
            f"{row['resources']['total_qubits']} & {number(row['operator_relative_residual_2'])} & "
            f"{number(row['relative_inverse_error_frobenius'])} & "
            f"{number(row['minimum_success_probability_all_inputs'])} \\\\"
        )
    (generated / "precision_rows.tex").write_text("\n".join(lines) + "\n")
    lines = [
        f"{r['size']} & {r['required_qubits']} & {r['native_action_operations'].get('cx', 0)} & "
        f"{r['native_action_depth']} & {number(r['native_ideal_residual'])} \\\\"
        for r in hardware
    ]
    (generated / "hardware_rows.tex").write_text("\n".join(lines) + "\n")
    lines = []
    for row in hardware:
        for noise in row["noise"]:
            lo, hi = noise["echo_return_95ci"]
            label = "Optimistic" if noise["profile"].startswith("optimistic") else "Stress"
            value = noise["observed_echo_return"]
            observed = "undefined" if value is None else f"{value:.3f}"
            lines.append(
                f"{row['size']} & {label} & {noise['shots']} & {noise['accepted']} & "
                f"{observed} & [{lo:.3f}, {hi:.3f}] & "
                f"{'Pass' if noise['quality_gate'] else 'Fail'} \\\\"
            )
    (generated / "noise_rows.tex").write_text("\n".join(lines) + "\n")
    lines = [
        f"{tex(r['name'])} & {'Pass' if r['passed'] else 'Fail'} & "
        f"{r.get('wall_seconds', 0):.2f} & {r.get('cpu_seconds', 0):.2f} & "
        f"{r.get('peak_child_rss_bytes', 0) / 1024**2:.1f} \\\\"
        for r in validation["stages"]
    ]
    (generated / "stage_rows.tex").write_text("\n".join(lines) + "\n")
    a2 = data["a2-a4"]["static"]
    lines = [
        f"{tex(r['case'])} & {r['depth']} & {number(r['ideal_residual'])} & "
        f"{number(r['shot_residual'])} & {number(r['noisy_residual'])} \\\\"
        for r in a2
    ]
    (generated / "a2_rows.tex").write_text("\n".join(lines) + "\n")


def figures(data, folder):
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(6.7, 3.2), layout="constrained")
    for index, condition in enumerate((2, 10, 100)):
        rows = [
            r
            for r in data["theory"]["cases"]
            if r["size"] == 4
            and r["family"] == "controlled_nonsymmetric"
            and r["condition_number"] == condition
        ]
        degrees = [r["degree"] for r in rows]
        ax.loglog(
            degrees,
            [max(r["operator_relative_residual_2"], 1e-16) for r in rows],
            "o-",
            ms=4,
            color=COLORS[index],
            label=rf"$\kappa={condition}$: measured",
        )
        ax.loglog(
            degrees,
            [max(r["residual_bound"], 1e-16) for r in rows],
            "--",
            color=COLORS[index],
            alpha=0.75,
            label=rf"$\kappa={condition}$: bound",
        )
    ax.set(
        xlabel="Polynomial degree d", ylabel=r"Operator residual $\|MX_d-I\|_2$", ylim=(1e-16, 1e3)
    )
    ax.grid(alpha=0.2, which="both")
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(folder / "convergence.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.7, 3.0), layout="constrained")
    for index, family in enumerate(
        ("controlled_spd", "controlled_nonsymmetric", "controlled_indefinite")
    ):
        rows = [r for r in data["precision-1000"]["cases"] if r["family"] == family]
        ax.loglog(
            [r["condition_number"] for r in rows],
            [r["operator_relative_residual_2"] for r in rows],
            "o-",
            color=COLORS[index],
            label=family.replace("controlled_", "").replace("_", " "),
        )
    ax.axhline(1e-12, ls="--", color="0.3", label="Polynomial target")
    ax.axhline(1e-10, ls=":", color="0.3", label="Numerical acceptance limit")
    ax.set(
        xlabel=r"Condition number $\kappa$",
        ylabel="Measured operator residual",
        ylim=(1e-15, 3e-10),
        xticks=[10, 100, 1000],
    )
    ax.grid(alpha=0.2, which="both")
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(folder / "precision.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.9), layout="constrained")
    rows = [r for r in data["precision-1000"]["cases"] if r["family"] == "controlled_nonsymmetric"]
    condition = [r["condition_number"] for r in rows]
    axes[0].semilogx(
        condition, [r["resources"]["total_qubits"] for r in rows], "o-", color=COLORS[0]
    )
    axes[0].set(xlabel=r"Condition number $\kappa$", ylabel="Total raw qubits", yticks=[17, 19, 21])
    axes[1].loglog(
        condition,
        [r["minimum_success_probability_all_inputs"] for r in rows],
        "o-",
        color=COLORS[1],
        label="Simulated minimum",
    )
    axes[1].loglog(condition, np.asarray(condition) ** -2, "k--", label=r"$\kappa^{-2}$")
    axes[1].set(xlabel=r"Condition number $\kappa$", ylabel="Worst-input success probability")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2, which="both")
    fig.savefig(folder / "resources.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.7, 3.0), layout="constrained")
    hardware = data["hardware"]["cases"]
    for index, profile in enumerate(("optimistic_synthetic", "stress_synthetic")):
        for position, row in enumerate(hardware):
            for noise in row["noise"]:
                if noise["profile"] == profile:
                    p = noise["observed_echo_return"]
                    if p is None:
                        ax.annotate("no accepted shots", (position, 0.08), rotation=90, fontsize=7)
                        continue
                    lo, hi = noise["echo_return_95ci"]
                    ax.errorbar(
                        position + (index - 0.5) * 0.12,
                        p,
                        yerr=[[p - lo], [hi - p]],
                        fmt="o",
                        capsize=3,
                        color=COLORS[index],
                        label=profile.replace("_synthetic", "") if position == 0 else None,
                    )
    ax.axhline(0.9, ls="--", color="0.4", label="Project echo threshold")
    ax.set(
        xticks=range(len(hardware)),
        xticklabels=[r["size"] for r in hardware],
        xlabel="Matrix dimension N",
        ylabel="Conditional reference-return fraction",
        ylim=(0, 1.06),
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.2, axis="y")
    fig.savefig(folder / "noise.pdf")
    plt.close(fig)


def build(results, output, metadata_file=None):
    data = load_results(results)
    metadata = json.loads(Path(metadata_file).read_text()) if metadata_file else {}
    output = Path(output).resolve()
    source = output / "source"
    shutil.copytree(ROOT / "paper", source, dirs_exist_ok=True)
    generated, plots = source / "generated", source / "figures"
    generated.mkdir(exist_ok=True)
    plots.mkdir(exist_ok=True)
    write_tables(data, generated, metadata)
    figures(data, plots)
    # Bundle the official class actually used so the source package is portable.
    located = subprocess.check_output(["kpsewhich", "IEEEtran.cls"], text=True).strip()
    if not located:
        raise RuntimeError("IEEEtran.cls not found; install texlive-publishers")
    shutil.copy2(located, source / "IEEEtran.cls")
    style = subprocess.run(
        ["kpsewhich", "IEEEtran.bst"], text=True, capture_output=True, check=False
    ).stdout.strip()
    style_path = Path(style) if style else Path(located).with_name("IEEEtran.bst")
    if not style_path.is_file():
        raise RuntimeError("IEEEtran.bst not found; install texlive-publishers")
    shutil.copy2(style_path, source / "IEEEtran.bst")
    outputs = []
    for stem, name in (
        ("manuscript", "QCeNN_MI_IEEE_Paper.pdf"),
        ("eurohpc_report", "QCeNN_MI_EuroHPC_Report.pdf"),
    ):
        subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"{stem}.tex"],
            cwd=source,
            check=True,
        )
        log = (source / f"{stem}.log").read_text(errors="replace")
        if "There were undefined references" in log or "Citation `" in log:
            raise RuntimeError(f"Unresolved references in {stem}")
        target = output / name
        shutil.copy2(source / f"{stem}.pdf", target)
        reader = PdfReader(target)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if len(text) < 5000:
            raise RuntimeError(f"Unexpectedly short {name}")
        outputs.append({"file": name, "pages": len(reader.pages)})
    (output / "publication.json").write_text(
        json.dumps(
            {
                "documents": outputs,
                "physical_hardware_tested": False,
                "source_commit": data["validation"]["provenance"]["source_commit"],
                "ai_assistance_disclosed": True,
            },
            indent=2,
        )
        + "\n"
    )
    if metadata:
        (output / "submission-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-metadata")
    args = parser.parse_args()
    print(build(args.results, args.output_dir, args.project_metadata))


if __name__ == "__main__":
    main()
