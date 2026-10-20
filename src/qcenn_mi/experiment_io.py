"""Small, explicit configuration and JSON helpers for reproducible experiments."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np


def save_npz_atomic(path, **arrays):
    """Publish a complete NPZ at once, before downstream hashing or transfer."""
    path = Path(path)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(buffer.getvalue())
    os.replace(temporary, path)


def configuration(path, section, defaults):
    result = dict(defaults)
    if path:
        data = json.loads(Path(path).read_text())
        if data.get("schema_version") != 1:
            raise ValueError("Experiment configuration requires schema_version=1")
        supplied = data.get(section, {})
        unknown = set(supplied) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown {section} settings: {sorted(unknown)}")
        result.update(supplied)
    return result


def save_result(output, payload):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "result.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload
