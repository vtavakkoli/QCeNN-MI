#!/usr/bin/env python3
"""Docker-free, pinned QCeNN-MI publication verification (Linux/Python 3.12).

Example:
    python3.12 scripts/run_publication_final.py --profile full
    python3.12 scripts/run_publication_final.py --profile full \
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


def clean_revision() -> str:
    if git("status", "--porcelain", "--untracked-files=normal"):
        fail("Commit or remove non-ignored repository changes before a pinned publication run.")
    return git("rev-parse", "HEAD")


def source_fingerprint() -> str:
    path = ROOT / "src" / "qcenn_mi" / "source_provenance.py"
    spec = importlib.util.spec_from_file_location("qcenn_source_provenance", path)
    if spec is None or spec.loader is None:
        fail("Cannot read source inventory")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.source_fingerprint(ROOT)


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


def python_environment(venv_dir: Path, *, skip_install: bool) -> Path:
    executable = venv_dir / "bin" / "python"
    if not executable.is_file():
        if skip_install:
            fail(f"No Python virtual environment in {venv_dir}")
        print("Creating Python virtual environment:", venv_dir, flush=True)
        venv.EnvBuilder(with_pip=True, symlinks=True).create(venv_dir)
    if not skip_install:
        run([str(executable), "-m", "pip", "install", "-r", str(PACKAGES)])
        run([str(executable), "-m", "pip", "install", "--no-deps", "-e", str(ROOT)])
    run([str(executable), "-m", "pip", "check"])
    # Do not silently change exact pinned scientific software between runs.
    checker = (
        "import importlib.metadata as md, pathlib, sys; "
        "lines=pathlib.Path(sys.argv[1]).read_text().splitlines(); "
        "pins=(line.strip().split('==',1) for line in lines "
        "if '==' in line and not line.lstrip().startswith('#')); "
        "bad=[(name, version, md.version(name)) for name,version in pins "
        "if md.version(name)!=version]; "
        "assert not bad, f'Non-pinned dependency versions: {bad}'"
    )
    run([str(executable), "-c", checker, str(PACKAGES)])
    return executable


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
    ]
    absent = [name for name in names if not (output / name).is_file()]
    if absent:
        fail(f"Publication archive incomplete: {absent}")
    files = {name: sha(output / name) for name in names}
    manifest = {
        "schema_version": 2,
        "source_commit": revision,
        "source_sha256": fingerprint,
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
    for folder in ("temporal-full", "phase-bias", "comparison"):
        target = output / f"{folder}.zip"
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in sorted((output / folder).rglob("*")):
                if file.is_file():
                    archive.write(file, arcname=file.relative_to(output).as_posix())
    print("Final evidence:", output / "FINAL_EVIDENCE.tar.gz")
    print("Separate experiment ZIPs:", *(output / f"{name}.zip" for name in
                                        ("temporal-full", "phase-bias", "comparison")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--paper-dir", type=Path)
    parser.add_argument("--compile-paper", action="store_true")
    parser.add_argument("--require-paper", action="store_true")
    args = parser.parse_args()

    if platform.system() != "Linux":
        parser.error("The repository validation.py uses Linux resource metrics; use Linux/WSL2")
    if sys.version_info[:2] != (3, 12):
        parser.error("Use Python 3.12 for the pinned requirements.lock campaign")
    revision = clean_revision()
    venv_dir = args.venv.expanduser().resolve()
    if venv_dir.is_relative_to(ROOT) and venv_dir != ROOT / ".venv":
        parser.error("Use the ignored .venv directory inside the Git checkout")
    output = (args.output_dir or ROOT / "artifacts" / (
        "publication-python-" + revision[:12] + "-" +
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )).expanduser().resolve()
    if output.exists():
        fail(f"Run output already exists: {output}; choose a new directory")
    output.mkdir(parents=True)
    started = time.monotonic()
    print(f"Repository: {ROOT}\nCommit: {revision}\nOutput: {output}", flush=True)
    try:
        python = python_environment(venv_dir, skip_install=args.skip_install)
        fingerprint = source_fingerprint()
        env = native_environment(venv_dir, fingerprint, revision)
        (output / "RUN.json").write_text(json.dumps({
            "commit": revision, "source_sha256": fingerprint, "profile": args.profile,
            "python": str(python), "environment": "native-python",
            "started_utc": datetime.now(UTC).isoformat(),
        }, indent=2) + "\n")
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
                and state["provenance"]["source_commit"] == revision
                and state["provenance"]["source_sha256"] == fingerprint):
            fail("Full campaign status does not match its pinned source or profile")
        print("[3/7] 96-case matched temporal comparison")
        run([str(python), "-m", "qcenn_mi.temporal_chebyshev_publication",
             "--output-dir", str(output / "temporal-full")], env=env)
        verify_manifest(output / "temporal-full")
        print("[4/7] Finite QSP phase and logical noise bias")
        run([str(python), "-m", "qcenn_mi.phase_bias_audit",
             "--conditions", "2", "10", "--output-dir",
             str(output / "phase-bias")], env=env)
        verify_manifest(output / "phase-bias", name="SHA256SUMS.json")
        print("[5/7] Prior results and manuscript-table audit")
        run([str(python), "-m", "qcenn_mi.publication_comparison",
             "--previous", str(ROOT / "results" / "polynomial-access-validation" /
                               "polynomial-summary.json"),
             "--temporal", str(output / "temporal-full" / "result.json"),
             "--phase", str(output / "phase-bias" / "result.json"),
             "--manuscript", str(ROOT / "manuscript" / "QCeNN_MI.tex"),
             "--output-dir", str(output / "comparison")], env=env)
        verify_manifest(output / "comparison")
        print("[6/7] Manuscript preflight")
        status = manuscript_check(python, args, output, env)
        if source_fingerprint() != fingerprint or clean_revision() != revision:
            fail("Source changed during the publication run")
        print("[7/7] Produce SHA-256 indexed evidence archives")
        archive_results(output, revision=revision, fingerprint=fingerprint,
                        manuscript_status=status)
    except BaseException:
        print(f"FAILED: partial evidence retained at {output}", file=sys.stderr)
        raise
    print(f"Completed in {time.monotonic() - started:.1f}s")
    print("Review", output / "comparison" / "COMPARISON.md")
    print("This checks reproducibility, not physical quantum advantage or editorial approval.")


if __name__ == "__main__":
    main()
