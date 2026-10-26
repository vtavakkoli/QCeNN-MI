#!/usr/bin/env python3
"""Docker-free QCeNN-MI publication verification using the selected Python venv.

Example:
    .venv-qcenn/bin/python scripts/run_publication_final.py --profile full
    .venv-qcenn/bin/python scripts/run_publication_final.py --profile full \
        --paper-dir ../qcenn-paper-assets/QCeNN_MI_Revision --require-paper

A passing software suite never implies QPU advantage or journal approval.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import venv
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "requirements.lock"


def fail(message: str) -> None:
    raise RuntimeError(message)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args],
        text=True,
    ).strip()


def source_revision() -> str:
    """Record Git HEAD even if the checkout contains local changes.

    The exact source bytes are independently frozen and fingerprinted.
    A dirty tree is not a scientific failure, and must not block execution.
    """
    return git("rev-parse", "HEAD")


def source_fingerprint() -> str:
    path = ROOT / "src" / "qcenn_mi" / "source_provenance.py"
    spec = importlib.util.spec_from_file_location("qcenn_source_provenance", path)
    if spec is None or spec.loader is None:
        fail("Cannot read source inventory")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.source_fingerprint(ROOT)


RESUME_ALLOWED_CHANGES = frozenset({
    "scripts/run_publication_final.py",
    ".github/workflows/native-python-publication.yml",
    "src/qcenn_mi/publication_comparison.py",
    "tests/test_publication_comparison.py",
    "tests/test_publication_runner_interpreter.py",
    "README.md",
})


def verify_resumable_source(output: Path, prior_run: dict, fingerprint: str) -> list[str]:
    """Reuse completed scientific stages only when their source is unchanged.

    A narrow list of runner/comparison/test/documentation edits may differ.
    The frozen original scientific source and its identity remain recorded;
    resumed stages explicitly record the new orchestrator source identity.
    """
    index_path = output / "suite" / "source-files.json"
    snapshot = output / "suite" / "source-snapshot.zip"
    if not index_path.is_file() or not snapshot.is_file():
        fail("Cannot resume: frozen source inventory/snapshot is missing")
    index = load_json(index_path)
    if (index.get("source_sha256") != prior_run.get("source_sha256")
            or index.get("snapshot_sha256") != sha(snapshot)):
        fail("Cannot resume: previous source snapshot integrity mismatch")
    if fingerprint == prior_run["source_sha256"]:
        return []
    spec = importlib.util.spec_from_file_location(
        "qcenn_resumed_source_provenance", ROOT / "src/qcenn_mi/source_provenance.py"
    )
    if spec is None or spec.loader is None:
        fail("Cannot load source inventory to verify safe resume")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in module.source_paths(ROOT)
    }
    original = index.get("files")
    if not isinstance(original, dict) or not original:
        fail("Cannot resume: previous file inventory is incomplete")
    changes = sorted(
        key for key in (set(original) | set(files))
        if original.get(key) != files.get(key)
    )
    forbidden = sorted(set(changes) - RESUME_ALLOWED_CHANGES)
    if forbidden:
        fail(
            "Cannot reuse a previous scientific campaign after changes to: "
            + ", ".join(forbidden)
            + ". Use a fresh output directory for a complete scientific rerun."
        )
    print(
        "RESUME WARNING: runner/comparison source differs from the original "
        "frozen campaign; recording both source identities. The scientific "
        "validation and temporal/phase results retain their original source "
        "provenance, not an identical single-source replay.",
        flush=True,
    )
    return changes


def completed_stage(output: Path, folder: str, *, manifest: str = "MANIFEST.json") -> bool:
    directory = output / folder
    if not directory.exists():
        return False
    if not (directory / manifest).is_file():
        fail(
            f"Cannot safely resume incomplete stage {directory}: "
            "manifest absent; use a new output directory or inspect it manually"
        )
    verify_manifest(directory, name=manifest)
    print(f"Reusing checksum-verified {folder} evidence", flush=True)
    return True


def run(command: list[str], *, log: Path | None = None, env: dict | None = None,
        cwd: Path = ROOT) -> None:
    print("Running:", " ".join(str(x) for x in command), flush=True)
    if log is None:
        subprocess.run(command, cwd=cwd, env=env, check=True)
        return
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        if process.wait() != 0:
            fail(f"Command failed ({process.returncode}); inspect {log}")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(folder: Path, name: str = "MANIFEST.json",
                    field: str = "sha256") -> None:
    manifest = folder / name
    if not manifest.is_file():
        fail(f"Missing evidence manifest: {manifest}")
    hashes = load_json(manifest)
    if field in hashes:
        hashes = hashes[field]
    elif "files" in hashes:
        hashes = hashes["files"]
    if not hashes or not isinstance(hashes, dict):
        fail(f"Invalid manifest: {manifest}")
    for relative, expected in hashes.items():
        source = folder / relative
        if not source.is_file() or sha(source) != expected:
            fail(f"Evidence checksum mismatch: {source}")


def select_environment(explicit_venv: Path | None) -> tuple[Path, Path]:
    """Use the running virtualenv by default, including .venv-qcenn."""
    if explicit_venv is not None:
        directory = explicit_venv.expanduser().resolve()
        interpreter = directory / "bin" / "python"
        if Path(sys.prefix).resolve() == directory:
            interpreter = Path(sys.executable)
        return directory, interpreter
    if sys.prefix != sys.base_prefix:
        return Path(sys.prefix).resolve(), Path(sys.executable)
    directory = ROOT / ".venv"
    return directory, directory / "bin" / "python"


def interpreter_version(executable: Path) -> tuple[int, int]:
    output = subprocess.check_output(
        [str(executable), "-c",
         "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        text=True,
    ).strip()
    major, minor = (int(part) for part in output.split("."))
    if (major, minor) < (3, 11):
        fail(f"QCeNN-MI requires Python >=3.11, selected {output}")
    return major, minor


def verify_dependencies(executable: Path, *, strict_lock: bool) -> None:
    """Exact 3.12 replay, or compatibility-checked alternate Python env."""
    run([str(executable), "-m", "pip", "check"])
    if strict_lock:
        checker = (
            "import importlib.metadata as md, pathlib, sys; "
            "lines=pathlib.Path(sys.argv[1]).read_text().splitlines(); "
            "pins=(line.strip().split('==',1) for line in lines "
            "if '==' in line and not line.lstrip().startswith('#')); "
            "bad=[(name,version,md.version(name)) for name,version in pins "
            "if md.version(name)!=version]; "
            "assert not bad, f'Non-pinned dependency versions: {bad}'"
        )
        run([str(executable), "-c", checker, str(PACKAGES)])
        return
    # All scientific and validation tools must be present; versions must
    # satisfy project ranges, not arbitrary equality to the Python-3.12 lock.
    checker = """
import importlib.metadata as md
from packaging.requirements import Requirement
from pathlib import Path
import tomllib
config = tomllib.loads(Path('pyproject.toml').read_text())
project = config['project']
requirements = (
    project['dependencies'] +
    project['optional-dependencies']['dev'] +
    project['optional-dependencies']['analysis']
)
missing = []
incompatible = []
for expression in requirements:
    req = Requirement(expression)
    try:
        installed = md.version(req.name)
    except md.PackageNotFoundError:
        missing.append(req.name)
        continue
    if installed not in req.specifier:
        incompatible.append(f'{req.name} {installed} does not satisfy {req.specifier}')
if missing or incompatible:
    raise SystemExit(f'Install missing packages: {missing}; incompatible: {incompatible}')
"""
    run([str(executable), "-c", checker])


def python_environment(venv_dir: Path, *, executable: Path, skip_install: bool
                       ) -> tuple[Path, str]:
    if not executable.is_file():
        if skip_install:
            fail(f"No Python interpreter exists at {executable}")
        if executable != venv_dir / "bin" / "python":
            fail(f"Requested interpreter does not exist: {executable}")
        print("Creating Python virtual environment:", venv_dir, flush=True)
        venv.EnvBuilder(with_pip=True, symlinks=True).create(venv_dir)
    version = interpreter_version(executable)
    strict_lock = version == (3, 12)
    if strict_lock:
        mode = "exact-python312-lock"
    else:
        mode = "compatible-unpinned-python"
        print(
            f"Python {version[0]}.{version[1]} selected from {executable}. "
            "The recorded requirements.lock was tested on Python 3.12. "
            "This run checks package compatibility instead of claiming an "
            "identical dependency replay.",
            flush=True,
        )
    if not skip_install:
        if strict_lock:
            run([str(executable), "-m", "pip", "install", "-r", str(PACKAGES)])
            run([str(executable), "-m", "pip", "install", "--no-deps", "-e", str(ROOT)])
        else:
            run([str(executable), "-m", "pip", "install", "-e", ".[dev,analysis]"])
    try:
        verify_dependencies(executable, strict_lock=strict_lock)
    except subprocess.CalledProcessError:
        fail(
            f"Missing or incompatible packages in {executable}. "
            f"To install compatible dependencies, run: "
            f"{executable} -m pip install -e '.[dev,analysis]' "
            "(on Python 3.12, prefer requirements.lock)."
        )
    return executable, mode


def native_environment(venv_dir: Path, fingerprint: str, revision: str) -> dict:
    env = dict(os.environ)
    env.update({
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "QCENN_EXECUTION_CONTEXT": "native-python",
        "QCENN_EXPECTED_SOURCE_SHA256": fingerprint,
        "QCENN_SOURCE_COMMIT": revision,
    })
    # Some Linux Aer wheels need their bundled OpenMP runtime preloaded.
    libraries = sorted(venv_dir.glob("lib/python*/site-packages/qiskit_aer.libs/libgomp*.so*"))
    if libraries:
        previous = env.get("LD_PRELOAD", "")
        if str(libraries[0]) not in previous.split(":"):
            env["LD_PRELOAD"] = ":".join(x for x in (str(libraries[0]), previous) if x)
        env["QCENN_OPENMP_PRELOAD"] = str(libraries[0])
    return env


def manuscript_check(python: Path, args: argparse.Namespace, output: Path,
                     env: dict) -> dict:
    evidence = output / "temporal-full"
    command = [
        str(python), "-m", "qcenn_mi.publication_preflight",
        "--repository", str(ROOT), "--evidence", str(evidence),
        "--output", str(output / "submission-preflight.json"),
    ]
    result = {"manuscript_supplied": False, "compiled": False}
    if args.paper_dir:
        folder = args.paper_dir.expanduser().resolve()
        required = (
            "QCeNN_MI.tex", "QCeNN_MI.pdf",
            "QCeNN_MI_Supplement.tex", "QCeNN_MI_Supplement.pdf",
        )
        missing = [name for name in required if not (folder / name).is_file()]
        if missing:
            fail(f"Missing manuscript files in {folder}: {missing}")
        for name in ("QCeNN_MI.tex", "QCeNN_MI_Supplement.tex"):
            if (ROOT / "manuscript" / name).read_bytes() != (folder / name).read_bytes():
                fail(f"{name} differs from the pinned GitHub manuscript source")
        if args.compile_paper:
            if shutil.which("pdflatex") is None:
                fail("--compile-paper requires pdflatex and the licensed IEEE template fonts")
            destination = output / "paper-build"
            shutil.copytree(folder, destination)
            for name in ("QCeNN_MI", "QCeNN_MI_Supplement"):
                for _ in range(2):
                    run([
                        "pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        f"{name}.tex",
                    ], cwd=destination, log=output / f"{name}.compile.log", env=env)
            folder = destination
            result["compiled"] = True
        if shutil.which("pdfinfo") is None:
            fail("PDF validation needs pdfinfo (typically poppler-utils)")
        pdf_output = subprocess.check_output(
            ["pdfinfo", str(folder / "QCeNN_MI.pdf")], text=True,
        )
        pages = next((int(line.split(":", 1)[1].strip()) for line
                      in pdf_output.splitlines() if line.startswith("Pages:")), None)
        if pages is None or pages > 20:
            fail(f"Main paper exceeds 20 pages or page count unknown: {pages}")
        result.update({"manuscript_supplied": True, "main_pages": pages})
        command += [
            "--manuscript-source", str(folder / "QCeNN_MI.tex"),
            "--manuscript-pdf", str(folder / "QCeNN_MI.pdf"),
        ]
        if args.require_paper:
            command.append("--strict")
    elif args.require_paper or args.compile_paper:
        fail("--require-paper/--compile-paper requires --paper-dir")
    run(command, env=env)
    if args.require_paper and not load_json(output / "submission-preflight.json")[
        "automatic_integrity_passed"
    ]:
        fail("Manuscript source/PDF integrity check failed")
    return result


def archive_results(output: Path, *, revision: str, fingerprint: str,
                    manuscript_status: dict) -> None:
    names = [
        "RUN.json", "suite/validation.json", "suite/final-report.json",
        "suite/final-report.md", "suite/source-files.json",
        "temporal-full/result.json", "temporal-full/REPORT.md",
        "temporal-full/MANIFEST.json", "phase-bias/result.json",
        "phase-bias/SHA256SUMS.json", "comparison/comparison.json",
        "comparison/COMPARISON.md", "comparison/MANIFEST.json",
        "submission-preflight.json",
        "extended-tracking/result.json", "extended-tracking/summary.csv",
        "extended-tracking/REPORT.md", "extended-tracking/MANIFEST.json",
        "native-limitations/result.json", "native-limitations/REPORT.md",
        "native-limitations/MANIFEST.json",
    ]
    absent = [name for name in names if not (output / name).is_file()]
    if absent:
        fail(f"Publication archive incomplete: {absent}")
    files = {name: sha(output / name) for name in names}
    run_record = load_json(output / "RUN.json")
    manifest = {
        "schema_version": 2,
        "source_commit": revision,
        "source_sha256": fingerprint,
        "resumed_stage_provenance": run_record.get("resume"),
        "single_source_replay": not bool(
            run_record.get("resume", {}).get("allowlisted_source_changes")
        ),
        "execution": "native-python-no-docker",
        "manuscript": manuscript_status,
        "files_sha256": files,
        "hardware_speedup_demonstrated": False,
        "journal_submission_approved": False,
    }
    destination = output / "FINAL_SHA256SUMS.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with tarfile.open(output / "FINAL_EVIDENCE.tar.gz", "w:gz") as archive:
        for name in [*names, "FINAL_SHA256SUMS.json"]:
            archive.add(output / name, arcname=name)
    # Standalone reproducibility evidence for each new experiment, not a
    # second copy of the potentially gigabyte-scale complete validation bundle.
    for folder in ("temporal-full", "phase-bias", "comparison",
                   "extended-tracking", "native-limitations"):
        target = output / f"{folder}.zip"
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in sorted((output / folder).rglob("*")):
                if file.is_file():
                    archive.write(file, arcname=file.relative_to(output).as_posix())
    print("Final evidence:", output / "FINAL_EVIDENCE.tar.gz")
    print("Separate experiment ZIPs:", *(output / f"{name}.zip" for name in
                                        ("temporal-full", "phase-bias", "comparison",
                                         "extended-tracking", "native-limitations")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--venv", type=Path, help="Defaults to current virtualenv, then .venv")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse verified completed stages in an existing --output-dir.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--paper-dir", type=Path)
    parser.add_argument("--compile-paper", action="store_true")
    parser.add_argument("--require-paper", action="store_true")
    args = parser.parse_args()

    if platform.system() != "Linux":
        parser.error("The repository validation.py uses Linux resource metrics; use Linux/WSL2")
    revision = source_revision()
    venv_dir, executable = select_environment(args.venv)
    output = (args.output_dir or ROOT / "artifacts" / (
        "publication-python-" + revision[:12] + "-" +
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )).expanduser().resolve()
    if args.resume:
        if not (output / "RUN.json").is_file():
            fail("--resume requires an existing output directory with RUN.json")
        if (output / "FINAL_EVIDENCE.tar.gz").exists():
            fail("Evidence archive already exists: run was completed")
    else:
        if output.exists():
            fail(f"Run output already exists: {output}; choose a new directory")
        output.mkdir(parents=True)
    started = time.monotonic()
    changes = git("status", "--porcelain", "--untracked-files=normal")
    if changes:
        print(
            "Notice: local checkout changes detected; proceeding without "
            "modifying them. Their source bytes will be recorded in the "
            "SHA-256 evidence snapshot.",
            flush=True,
        )
    print(f"Repository: {ROOT}\nCommit: {revision}\nOutput: {output}", flush=True)
    try:
        python, dependency_mode = python_environment(
            venv_dir, executable=executable, skip_install=args.skip_install
        )
        fingerprint = source_fingerprint()
        prior_run = load_json(output / "RUN.json") if args.resume else None
        if prior_run is not None:
            if prior_run.get("profile") != args.profile:
                fail("Cannot resume: the profile differs from the original run")
            if Path(prior_run.get("python", "")).resolve() != python.resolve():
                fail("Cannot resume: select the original Python interpreter")
            changed = verify_resumable_source(output, prior_run, fingerprint)
            initial_revision = prior_run["commit"]
            initial_fingerprint = prior_run["source_sha256"]
            prior_run["resume"] = {
                "original_commit": initial_revision,
                "original_source_sha256": initial_fingerprint,
                "current_commit": revision,
                "current_source_sha256": fingerprint,
                "allowlisted_source_changes": changed,
                "stage_1_to_4_provenance": "original frozen source snapshot",
                "remaining_stage_provenance": "current source, separately recorded",
                "single_source_reproduction_if_changes": False,
            }
            (output / "RUN.json").write_text(
                json.dumps(prior_run, indent=2) + "\n", encoding="utf-8"
            )
        else:
            initial_revision, initial_fingerprint = revision, fingerprint
        env = native_environment(venv_dir, fingerprint, revision)
        if not args.resume:
            (output / "RUN.json").write_text(json.dumps({
            "commit": revision, "source_sha256": fingerprint, "profile": args.profile,
            "python": str(python), "environment": "native-python",
            "python_version": ".".join(map(str, interpreter_version(python))),
            "dependency_mode": dependency_mode,
            "git_working_tree_dirty_at_start": bool(changes),
            "git_working_tree_status_at_start": changes.splitlines(),
            "source_snapshot_note": (
                "Source SHA-256 and the suite source-snapshot.zip describe the "
                "actual checkout, including local modifications."
            ),
            "started_utc": datetime.now(UTC).isoformat(),
        }, indent=2) + "\n")
        if args.resume:
            print("[1/7] Reusing the completed full/smoke simulator campaign")
        else:
            print("[1/7] Full/smoke simulator campaign")
            run([str(python), "-m", "qcenn_mi.validation", "--profile", args.profile,
                 "--output-dir", str(output / "suite")],
                log=output / "suite-run.log", env=env)
        print("[2/7] Independently verify full campaign evidence")
        run([str(python), "-m", "qcenn_mi.evidence_report",
             "--results", str(output / "suite"), "--verify-only"], env=env)
        state = load_json(output / "suite" / "validation.json")
        if not (state.get("passed") and state.get("evidence_integrity_passed")
                and state.get("source_unchanged_during_run")
                and state.get("profile") == args.profile
                and state["provenance"]["source_commit"] == initial_revision
                and state["provenance"]["source_sha256"] == initial_fingerprint):
            fail("Full campaign status does not match its pinned source or profile")
        print("[3/7] 96-case matched temporal comparison")
        if not (args.resume and completed_stage(output, "temporal-full")):
            run([str(python), "-m", "qcenn_mi.temporal_chebyshev_publication",
                 "--output-dir", str(output / "temporal-full")], env=env)
            verify_manifest(output / "temporal-full")
        print("[4/7] Finite QSP phase and logical noise bias")
        if not (args.resume and completed_stage(
                output, "phase-bias", manifest="SHA256SUMS.json")):
            run([str(python), "-m", "qcenn_mi.phase_bias_audit",
                 "--conditions", "2", "10", "--output-dir",
                 str(output / "phase-bias")], env=env)
            verify_manifest(output / "phase-bias", name="SHA256SUMS.json")
        print("[5/7] Prior results and manuscript-table audit")
        if not (args.resume and completed_stage(output, "comparison")):
            comparison_command = [
                str(python), "-m", "qcenn_mi.publication_comparison",
                "--temporal", str(output / "temporal-full" / "result.json"),
                "--phase", str(output / "phase-bias" / "result.json"),
                "--manuscript", str(ROOT / "manuscript" / "QCeNN_MI.tex"),
                "--output-dir", str(output / "comparison"),
            ]
            previous = ROOT / "results" / "polynomial-access-validation" / "polynomial-summary.json"
            if previous.is_file():
                comparison_command += ["--previous", str(previous)]
            else:
                print(
                    "Optional historical static polynomial summary absent; "
                    "auditing matched temporal/QSP/manuscript normally and recording "
                    "previous_static.status=not_available.",
                    flush=True,
                )
            run(comparison_command, env=env)
            verify_manifest(output / "comparison")
        print("[5a] Stress-test extended trajectories and moving-target theorem")
        extended_args = [
            str(python), "-m", "qcenn_mi.extended_tracking_benchmark",
            "--output-dir", str(output / "extended-tracking"),
        ]
        if args.profile == "smoke":
            extended_args.append("--smoke")
        if not (args.resume and completed_stage(output, "extended-tracking")):
            run(extended_args, env=env, log=output / "extended-tracking.log")
            verify_manifest(output / "extended-tracking")
        print("[5b] Audit actual native gates and synthetic gate-local noise")
        native_args = [
            str(python), "-m", "qcenn_mi.native_limitation_audit",
            "--output-dir", str(output / "native-limitations"),
        ]
        if args.profile == "smoke":
            native_args.append("--smoke")
        if not (args.resume and completed_stage(output, "native-limitations")):
            run(native_args, env=env, log=output / "native-limitations.log")
            verify_manifest(output / "native-limitations")
        print("[6/7] Manuscript preflight")
        status = manuscript_check(python, args, output, env)
        if source_fingerprint() != fingerprint or source_revision() != revision:
            fail("Source changed during the publication run")
        print("[7/7] Produce SHA-256 indexed evidence archives")
        archive_results(output, revision=initial_revision, fingerprint=initial_fingerprint,
                        manuscript_status=status)
    except BaseException:
        print(f"FAILED: partial evidence retained at {output}", file=sys.stderr)
        raise
    print(f"Completed in {time.monotonic() - started:.1f}s")
    print("Review", output / "comparison" / "COMPARISON.md")
    print("This checks reproducibility, not physical quantum advantage or editorial approval.")


if __name__ == "__main__":
    main()
