"""Pin a clean Git source revision and its bytes before invoking Docker Compose.

Run from any directory: python docker/reproduce.py --profile full
No third-party Python packages are required on the host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import subprocess
from pathlib import Path


def pinned_environment(root, profile):
    root = Path(root).resolve()
    api = runpy.run_path(str(root / "src/qcenn_mi/source_provenance.py"))
    if not (root / ".git").exists():
        metadata = json.loads((root / "SOURCE_REVISION.json").read_text())
        fingerprint = api["source_fingerprint"](root)
        if fingerprint != metadata["source_sha256"] or metadata["source_commit"] == "unknown":
            raise ValueError("Source export does not match a known source revision")
        return dict(
            os.environ,
            SOURCE_COMMIT=metadata["source_commit"],
            SOURCE_SHA256=fingerprint,
            QCENN_PROFILE=profile,
        )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    paths = [path.relative_to(root).as_posix() for path in api["source_paths"](root)]
    changed = subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only", "--", *api["source_scope"]()], cwd=root, text=True
    )
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *paths], cwd=root, text=True
    )
    if changed.strip() or untracked.strip():
        raise ValueError("Commit the source changes before a pinned publication run")
    # Git stores canonical LF bytes even on a CRLF Windows checkout; Docker's
    # existing normalize_source.py restores those bytes before validation.
    fingerprint = hashlib.sha256()
    for path in paths:
        content = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=root)
        fingerprint.update(path.encode() + b"\0" + content + b"\0")
    return dict(
        os.environ,
        SOURCE_COMMIT=revision,
        SOURCE_SHA256=fingerprint.hexdigest(),
        QCENN_PROFILE=profile,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    environment = pinned_environment(root, args.profile)
    print(f"Pinned source: {environment['SOURCE_COMMIT']}", flush=True)
    print(f"Source SHA-256: {environment['SOURCE_SHA256']}", flush=True)
    raise SystemExit(
        subprocess.run(
            [
                "docker",
                "compose",
                "up",
                "--build",
                "--abort-on-container-exit",
                "--exit-code-from",
                "validation",
            ],
            cwd=root,
            env=environment,
            check=False,
        ).returncode
    )


if __name__ == "__main__":
    main()
