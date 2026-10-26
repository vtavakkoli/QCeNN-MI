"""Restore canonical LF source bytes when Docker copies an older CRLF checkout.

Git attributes protect new checkouts. Existing Windows working trees may still
contain CRLF until each file is checked out again, so the image canonicalizes
only declared source text before installation and byte-level evidence checks.
Binary circuit archives, generated outputs, and checksum values are untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SOURCE_DIRECTORIES = (
    "src",
    "tests",
    "docker",
    "configs",
    "docs",
    "manuscript",
    ".github",
    "publication-evidence",
)
ROOT_TEXT_FILES = (
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
TEXT_SUFFIXES = {".py", ".json", ".md", ".toml", ".yml", ".yaml", ".txt", ".tex", ".bib", ".log"}


def normalize_source(root):
    """Replace CRLF with LF in source text; retain strict downstream SHA checks."""
    root = Path(root)
    paths = [root / name for name in ROOT_TEXT_FILES]
    paths.extend(
        path
        for directory in SOURCE_DIRECTORIES
        for path in (root / directory).rglob("*")
        if path.suffix in TEXT_SUFFIXES or path.name == "ORIGINAL_LICENSE"
    )
    changed = []
    for path in sorted(paths):
        if (
            not path.is_file()
            or path.is_symlink()
            or any(part == "__pycache__" or part.endswith(".egg-info") for part in path.parts)
        ):
            continue
        original = path.read_bytes()
        canonical = original.replace(b"\r\n", b"\n")
        if canonical != original:
            path.write_bytes(canonical)
            changed.append(path.relative_to(root).as_posix())
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    changed = normalize_source(args.root)
    print(f"Canonical LF source: normalized {len(changed)} file(s)")


if __name__ == "__main__":
    main()
