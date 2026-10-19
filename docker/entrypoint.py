"""Load Aer's wheel-specific OpenMP library before the target Python process."""

from __future__ import annotations

import glob
import os
import sys
import sysconfig
from pathlib import Path


def aer_openmp_library():
    roots = {sysconfig.get_path("platlib"), sysconfig.get_path("purelib")}
    # Python 3.11/3.12 -S does not initialize a virtual environment's prefix.
    prefix = Path(sys.executable).absolute().parent.parent
    roots.add(
        str(
            prefix
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
    )
    matches = sorted(
        {
            p
            for root in roots
            if root
            for p in glob.glob(str(Path(root) / "qiskit_aer.libs" / "libgomp*.so*"))
        }
    )
    return matches[0] if matches else None


def main():
    command = sys.argv[1:]
    if not command:
        raise SystemExit("An executable and its arguments are required")
    env = os.environ.copy()
    library = aer_openmp_library() if sys.platform.startswith("linux") else None
    if library:
        current = env.get("LD_PRELOAD", "").replace(":", " ").split()
        env["LD_PRELOAD"] = ":".join(dict.fromkeys([library, *current]))
        env["QCENN_OPENMP_PRELOAD"] = library
    elif env.get("QCENN_REQUIRE_AER_PRELOAD") == "1":
        raise SystemExit(
            "Aer wheel OpenMP library not found; rebuild the image with requirements.lock"
        )
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
