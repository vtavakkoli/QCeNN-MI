"""Exercise real Aer statevector and shot execution before expensive campaigns."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path


def campaign(output):
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    state = circuit.copy()
    state.save_statevector()
    backend = AerSimulator(max_parallel_threads=1)
    vector = np.asarray(backend.run(state).result().get_statevector())
    circuit.measure_all()
    counts = backend.run(circuit, shots=256, seed_simulator=1729).result().get_counts()
    passed = np.allclose(abs(vector) ** 2, [0.5, 0, 0, 0.5]) and set(counts) <= {"00", "11"}
    payload = {
        "passed": bool(passed),
        "machine": platform.machine(),
        "openmp_preload": os.environ.get("QCENN_OPENMP_PRELOAD"),
        "statevector_probabilities": (abs(vector) ** 2).tolist(),
        "raw_counts": counts,
        "physical_hardware_tested": False,
    }
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/runtime-smoke")
    args = parser.parse_args()
    raise SystemExit(0 if campaign(args.output_dir)["passed"] else 1)


if __name__ == "__main__":
    main()
