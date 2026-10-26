#!/usr/bin/env bash
# Compatibility entry point. Publication execution is native Python, not Docker.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v python3.12 >/dev/null 2>&1; then
  PYTHON=python3.12
else
  PYTHON=python3
fi
exec "$PYTHON" "$ROOT/scripts/run_publication_final.py" "$@"
