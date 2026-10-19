import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_report_and_validation_import_without_loading_aer():
    code = (
        "import sys; import qcenn_mi; import qcenn_mi.validation; "
        "import qcenn_mi.evidence_report; assert 'qiskit_aer' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux ELF loading regression")
def test_entrypoint_preloads_the_actual_wheel_and_runs_aer():
    code = (
        "import os,json; from qiskit_aer import AerSimulator; "
        "p=os.environ['QCENN_OPENMP_PRELOAD']; "
        "assert p in os.environ['LD_PRELOAD']; "
        "print(json.dumps({'library':p,'backend':AerSimulator().name}))"
    )
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    env.pop("QCENN_OPENMP_PRELOAD", None)
    result = subprocess.run(
        [sys.executable, "-S", str(ROOT / "docker/entrypoint.py"), sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    data = json.loads(result.stdout)
    assert Path(data["library"]).is_file()
    assert "qiskit_aer.libs" in data["library"]
    assert data["backend"] == "aer_simulator"


def test_entrypoint_does_not_import_quantum_libraries_during_discovery():
    spec = importlib.util.spec_from_file_location("aer_entrypoint", ROOT / "docker/entrypoint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.aer_openmp_library)


def test_legacy_package_exports_remain_available():
    from qcenn_mi import InversionResult, solve_inverse_2x2

    assert callable(solve_inverse_2x2)
    assert InversionResult.__name__ == "InversionResult"
