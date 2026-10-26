"""Submission preflight: inventory/manifest checks; no fabricated scientific approval.

This checker cannot validate manuscript prose, DOI correctness, or the
source-to-PDF typesetting equivalence. Those require a human sign-off.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(folder):
    folder = Path(folder)
    manifest = folder / "MANIFEST.json"
    if not manifest.is_file():
        return {"status": "missing", "path": str(manifest)}
    raw = json.loads(manifest.read_text())
    hashes = raw.get("sha256", raw.get("files"))
    if not isinstance(hashes, dict) or not hashes:
        return {"status": "invalid_schema", "path": str(manifest)}
    differences = []
    for name, expected in hashes.items():
        path = folder / name
        if not path.is_file() or sha256(path) != expected:
            differences.append(name)
    return {"status": "passed" if not differences else "failed",
            "files_checked": len(hashes), "mismatches": differences}


def pdf_check(pdf):
    if pdf is None:
        return {"status": "not_supplied"}
    path = Path(pdf)
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    data = path.read_bytes()
    signature = data.startswith(b"%PDF-") and b"%%EOF" in data[-2048:]
    result = {"status": "header_trailer_ok" if signature else "malformed",
              "sha256": sha256(path), "bytes": len(data)}
    try:
        output = subprocess.run(["pdfinfo", str(path)], text=True,
                                capture_output=True, timeout=30, check=True)
        for line in output.stdout.splitlines():
            if line.startswith("Pages:"):
                result["pages"] = int(line.partition(":")[2].strip())
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        result["page_count_verified"] = False
    else:
        result["page_count_verified"] = "pages" in result
    return result


def inspect(*, repository, evidence, manuscript_source=None, manuscript_pdf=None):
    repository, evidence = Path(repository), Path(evidence)
    status = {"repository": str(repository), "evidence": str(evidence)}
    status["evidence_manifest"] = verify_manifest(evidence)
    source = Path(manuscript_source) if manuscript_source else None
    status["editable_manuscript"] = {
        "status": "present" if source is not None and source.is_file() else "not_supplied",
        "sha256": sha256(source) if source is not None and source.is_file() else None,
    }
    status["pdf"] = pdf_check(manuscript_pdf)
    if (repository / ".git").is_dir():
        process = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"],
                                 capture_output=True, text=True, check=False)
        status["source_commit"] = process.stdout.strip() if process.returncode == 0 else None
    else:
        status["source_commit"] = None
    status["automatic_integrity_passed"] = bool(
        status["evidence_manifest"]["status"] == "passed" and
        status["editable_manuscript"]["status"] == "present" and
        status["pdf"]["status"] == "header_trailer_ok" and
        status["pdf"].get("page_count_verified", False))
    status["manual_verifications_outstanding"] = [
        "Rebuild the manuscript PDF from the exact submitted source and inspect every page",
        "Confirm every result/table/figure against the manifest-matched frozen data",
        "Confirm every reference and DOI against publisher records",
        "Confirm author names, corresponding author and all author approvals",
        "Confirm acknowledgments, disclosures, AI-assistance statement, funding and permissions",
        "Confirm current IEEE Access article template, length rules and portal requirements",
    ]
    status["submission_signed_off"] = False
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--manuscript-source", type=Path)
    parser.add_argument("--manuscript-pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = inspect(repository=args.repository, evidence=args.evidence,
                     manuscript_source=args.manuscript_source,
                     manuscript_pdf=args.manuscript_pdf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Automatic integrity status: {report['automatic_integrity_passed']}")
    print("Manual sign-off required; checker never declares submission approved.")
    if args.strict and not report["automatic_integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
