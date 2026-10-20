"""Portable canonical source inventory, hashing and immutable run snapshots."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

SOURCE_DIRECTORIES = (
    "src",
    "tests",
    "docker",
    "configs",
    "docs",
    ".github",
    "publication-evidence",
)
ROOT_FILES = (
    "Dockerfile",
    "docker-compose.yml",
    "requirements.lock",
    "pyproject.toml",
    "README.md",
    "CITATION.cff",
    "LICENSE",
    ".dockerignore",
    ".gitignore",
    ".gitattributes",
)


def source_scope():
    return [*SOURCE_DIRECTORIES, *ROOT_FILES]


def is_source_file(path):
    return path.is_file() and not any(
        part == "__pycache__" or part.endswith(".egg-info") for part in path.parts
    )


def source_paths(root):
    """Single canonical source inventory for fingerprinting and frozen exports."""
    root = Path(root)
    paths = [
        p for folder in SOURCE_DIRECTORIES for p in (root / folder).rglob("*") if is_source_file(p)
    ]
    paths += [root / name for name in ROOT_FILES if (root / name).is_file()]
    return sorted(paths)


def source_fingerprint(root):
    root = Path(root)
    digest = hashlib.sha256()
    for path in source_paths(root):
        digest.update(
            path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0"
        )
    return digest.hexdigest()


def freeze_source(root, output):
    """Freeze actual run bytes before execution, even without a Git checkout."""
    root, output = Path(root).resolve(), Path(output).resolve()
    inventory = {}
    with zipfile.ZipFile(output / "source-snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths(root):
            relative = path.relative_to(root).as_posix()
            content = path.read_bytes()
            inventory[relative] = hashlib.sha256(content).hexdigest()
            archive.writestr("QCeNN-MI/" + relative, content)
    payload = {
        "schema_version": 1,
        "source_sha256": source_fingerprint(root),
        "snapshot_sha256": hashlib.sha256(
            (output / "source-snapshot.zip").read_bytes()
        ).hexdigest(),
        "files": inventory,
    }
    (output / "source-files.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
