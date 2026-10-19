"""Run every simulator campaign and retain failures, timing and provenance.

One process per stage isolates native simulators and permits meaningful Linux
child CPU/peak-RSS measurements. Synthetic noisy quality is reported separately
from software validation. This command never connects to or submits to a QPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path


def stage_commands(output, *, profile="full", config=None):
    output = Path(output).resolve()
    python = sys.executable

    def campaign(name, module, *args):
        return name, [python, "-m", module, "--output-dir", str(output / name), *args]

    commands = [
        campaign("runtime-smoke", "qcenn_mi.runtime_smoke"),
        ("compile", [python, "-m", "compileall", "-q", "src", "tests"]),
        ("lint", [python, "-m", "ruff", "check", "src", "tests", "docker"]),
        (
            "unit-tests",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                f"--junitxml={output / 'unit-tests.xml'}",
            ],
        ),
        campaign("assumption-a1", "qcenn_mi.cli", "--shots", "8192", "--seed", "42"),
        campaign("a2-a4", "qcenn_mi.experiments"),
        campaign("scaling", "qcenn_mi.scaling"),
        campaign("coherent", "qcenn_mi.coherent_benchmark"),
        campaign("coherent-large", "qcenn_mi.coherent_large_benchmark"),
        campaign("compact", "qcenn_mi.compact_benchmark"),
        campaign("precision-1000", "qcenn_mi.precision_benchmark"),
        campaign("theory", "qcenn_mi.theory_benchmark"),
        campaign("time-varying", "qcenn_mi.time_varying_benchmark"),
        campaign(
            "hardware", "qcenn_mi.hardware", "--qubit-budget", "42", "--large-noise-shots", "64"
        ),
    ]
    if profile == "smoke":
        commands = [
            row
            for row in commands
            if row[0]
            in {"runtime-smoke", "compile", "lint", "unit-tests", "assumption-a1", "compact"}
        ]
    elif profile != "full":
        raise ValueError("profile must be full or smoke")
    for name, module in [
        ("temporal-refresh", "qcenn_mi.temporal_refresh_benchmark"),
        ("noise-verification", "qcenn_mi.noise_verification_benchmark"),
        ("baselines", "qcenn_mi.baseline_benchmark"),
        ("resources", "qcenn_mi.resource_benchmark"),
        ("operator-audit", "qcenn_mi.operator_audit"),
    ]:
        commands.append(
            campaign(
                name, module, "--config", str(Path(config or f"configs/{profile}.json").resolve())
            )
        )
    return commands


def is_source_file(path):
    return path.is_file() and not any(
        part == "__pycache__" or part.endswith(".egg-info") for part in path.parts
    )


def source_fingerprint(root):
    root = Path(root)
    paths = [
        p
        for folder in (
            "src",
            "tests",
            "docker",
            "configs",
            "docs",
            ".github",
            "publication-evidence",
        )
        for p in (root / folder).rglob("*")
        if is_source_file(p)
    ]
    paths += [
        root / name
        for name in (
            "Dockerfile",
            "docker-compose.yml",
            "requirements.lock",
            "pyproject.toml",
            "README.md",
            "CITATION.cff",
            "LICENSE",
        )
        if (root / name).is_file()
    ]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def create_bundle(root, output):
    """Archive a portable source tree and measured outputs, excluding recursion."""
    root, output = Path(root).resolve(), Path(output).resolve()
    bundle = output / "QCeNN_MI_Validation_Bundle.zip"
    source_files = [
        p
        for folder in (
            "src",
            "tests",
            "docs",
            ".github",
            "docker",
            "configs",
            "publication-evidence",
        )
        for p in (root / folder).rglob("*")
        if is_source_file(p)
    ]
    source_files += [
        root / name
        for name in (
            "Dockerfile",
            "docker-compose.yml",
            "requirements.lock",
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "CITATION.cff",
            ".dockerignore",
            ".gitignore",
        )
        if (root / name).is_file()
    ]
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_files):
            archive.write(path, "QCeNN-MI/" + str(path.relative_to(root)))
        for path in sorted(output.rglob("*")):
            if path.is_file() and path != bundle and path.name != "PACKAGE_SHA256":
                archive.write(path, "QCeNN-MI/evidence/" + str(path.relative_to(output)))
    checksum = hashlib.sha256(bundle.read_bytes()).hexdigest()
    (output / "PACKAGE_SHA256").write_text(f"{checksum}  {bundle.name}\n")
    return bundle


def installed_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def provenance(root):
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, cwd=root
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=root).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = os.environ.get("QCENN_SOURCE_COMMIT", "unknown"), None
    packages = (
        "numpy",
        "scipy",
        "qiskit",
        "qiskit-aer",
        "pytest",
        "ruff",
        "matplotlib",
        "qcenn-mi",
    )
    return {
        "source_commit": revision,
        "working_tree_dirty": dirty,
        "source_sha256": source_fingerprint(root),
        "source_fingerprint_policy": (
            "source, tests, docs, config, workflow and license; "
            "generated egg-info and caches excluded"
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "execution_context": os.environ.get("QCENN_EXECUTION_CONTEXT", "local-process"),
        "container_base_image": os.environ.get("QCENN_BASE_IMAGE"),
        "packages": {name: installed_version(name) for name in packages},
        "openmp_preload": os.environ.get("QCENN_OPENMP_PRELOAD"),
        "glibc_tunables": os.environ.get("GLIBC_TUNABLES"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "threads": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "eurohpc_allocated_node_execution_claimed": False,
        "physical_hardware_tested": False,
        "clock_source": "execution-environment UTC clock; not independently synchronized",
    }


def worker(metrics, command):
    started = time.perf_counter()
    try:
        code = subprocess.run(command, check=False).returncode
    except OSError as error:
        print(str(error), file=sys.stderr)
        code = 127
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    payload = {
        "returncode": code,
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
        "peak_child_rss_bytes": int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
    }
    Path(metrics).write_text(json.dumps(payload, indent=2) + "\n")
    return code


def execute_stage(name, command, output, *, env):
    logs = Path(output) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    metrics = logs / f"{name}.metrics.json"
    logfile = logs / f"{name}.log"
    wrapper = [
        sys.executable,
        "-m",
        "qcenn_mi.validation",
        "--worker-metrics",
        str(metrics),
        "--",
        *command,
    ]
    print(f"Starting {name}", flush=True)
    with logfile.open("w") as handle:
        process = subprocess.run(
            wrapper, stdout=handle, stderr=subprocess.STDOUT, check=False, env=env
        )
    data = (
        json.loads(metrics.read_text()) if metrics.exists() else {"returncode": process.returncode}
    )
    data.update(
        name=name,
        command=command,
        log=str(logfile.relative_to(output)),
        passed=process.returncode == 0,
    )
    print(f"{'PASS' if data['passed'] else 'FAIL'} {name}", flush=True)
    if not data["passed"]:
        print(logfile.read_text()[-6000:], flush=True)
    return data


def persist_summary(output, payload):
    output = Path(output)
    (output / "validation.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Complete QCeNN-MI validation",
        "",
        f"Run ID: `{payload['publication_run_id']}`; profile: `{payload['profile']}`.",
        "",
        (
            "Detailed evidence: [final-report.html](final-report.html), "
            "[final-report.md](final-report.md), and "
            "[manifest](publication-evidence/manifest.json)."
        ),
        "",
        f"Software validation: {'PASS' if payload['passed'] else 'FAIL / INCOMPLETE'}",
        "",
        f"Source: `{payload['provenance']['source_commit']}`; "
        f"tree SHA-256: `{payload['provenance']['source_sha256']}`.",
        "",
        "| Stage | Result | Wall seconds | CPU seconds | Peak child MiB |",
        "|---|---|---:|---:|---:|",
    ]
    for row in payload["stages"]:
        lines.append(
            f"| {row['name']} | {'PASS' if row['passed'] else 'FAIL'} | "
            f"{row.get('wall_seconds', 0):.2f} | {row.get('cpu_seconds', 0):.2f} | "
            f"{row.get('peak_child_rss_bytes', 0) / 1024**2:.1f} |"
        )
    lines += [
        "",
        f"All synthetic noisy quality gates: {payload.get('all_noisy_quality_gates_passed')}",
        "",
        "A software PASS does not certify noisy quality or physical execution.",
        "No EuroHPC allocated-node run or physical QPU run is claimed by this artifact.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def run_suite(output, *, profile="full", config=None, selected=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A completed run must never silently mix with stale output from a prior run.
    if (output / "validation.json").exists():
        raise FileExistsError("Output contains a validation run; choose a new directory")
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    payload = {
        "publication_run_id": "qcenn-" + uuid.uuid4().hex,
        "profile": profile,
        "schema_version": 2,
        "started_utc": datetime.now(UTC).isoformat(),
        "provenance": provenance(Path.cwd()),
        "stages": [],
        "passed": False,
    }
    config = Path(config or f"configs/{profile}.json").resolve()
    config_data = json.loads(config.read_text())
    (output / "configuration.json").write_text(
        json.dumps(
            {"profile": profile, "selected_stages": selected, "configuration": config_data},
            indent=2,
        )
        + "\n"
    )
    commands = stage_commands(output, profile=profile, config=config)
    if selected:
        unknown = set(selected) - {name for name, _ in commands}
        if unknown:
            raise ValueError(f"Unknown stages: {sorted(unknown)}")
        commands = [row for row in commands if row[0] in selected]
    for name, command in commands:
        payload["stages"].append(execute_stage(name, command, output, env=env))
        persist_summary(output, payload)
    hardware = output / "hardware" / "result.json"
    if hardware.exists():
        payload["all_noisy_quality_gates_passed"] = json.loads(hardware.read_text()).get(
            "all_synthetic_quality_gates_passed"
        )
    noise_study = output / "noise-verification/result.json"
    quality = {"native_hardware": payload.get("all_noisy_quality_gates_passed")}
    if noise_study.exists():
        quality["repeated_controls"] = json.loads(noise_study.read_text()).get(
            "all_synthetic_quality_gates_passed"
        )
    payload["synthetic_quality_by_study"] = quality
    observed_quality = [value for value in quality.values() if value is not None]
    payload["all_noisy_quality_gates_passed"] = all(observed_quality) if observed_quality else None
    payload["passed"] = all(row["passed"] for row in payload["stages"])
    persist_summary(output, payload)
    report_command = [sys.executable, "-m", "qcenn_mi.evidence_report", "--results", str(output)]
    payload["stages"].append(execute_stage("final-report", report_command, output, env=env))
    payload["passed"] = all(row["passed"] for row in payload["stages"])
    payload["evidence_integrity_passed"] = payload["stages"][-1]["passed"]
    payload["finished_utc"] = datetime.now(UTC).isoformat()
    persist_summary(output, payload)
    hashes = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "SHA256SUMS.json"
    }
    (output / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2) + "\n")
    if payload["passed"]:
        create_bundle(Path.cwd(), output)
    return payload


def main():
    if "--worker-metrics" in sys.argv:
        marker = sys.argv.index("--")
        raise SystemExit(worker(sys.argv[2], sys.argv[marker + 1 :]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--profile", choices=["full", "smoke"], default="full")
    parser.add_argument("--config")
    parser.add_argument("--stages", nargs="+")
    parser.add_argument("--list-stages", action="store_true")
    args = parser.parse_args()
    if args.list_stages:
        print(
            "\n".join(
                name
                for name, _ in stage_commands(
                    args.output_dir, profile=args.profile, config=args.config
                )
            )
        )
        return
    result = run_suite(
        args.output_dir, profile=args.profile, config=args.config, selected=args.stages
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
