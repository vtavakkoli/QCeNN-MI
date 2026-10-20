"""Budgeted complete-circuit checks of deferred residual reset composition."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import multiprocessing as mp
import queue as queue_module
import time
from pathlib import Path

import numpy as np

from .experiment_io import configuration
from .polynomial_benchmark import fixture, save_json
from .polynomial_composition import composition_resources, polynomial_composition_circuit
from .validation import provenance

DEFAULTS = {
    "sizes": [2, 4, 10, 100, 1000],
    "conditions": [2, 100],
    "degrees": [0, 1, 2, 4, 8, 16],
    "timeout_seconds": 45,
    "maximum_controlled_unitary_qubits": 8,
    "seed": 1729,
}


def composition_table(data):
    rows = data["cases"]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def build_report(output):
    output = Path(output)
    data = json.loads((output / "result.json").read_text())
    (output / "resources.csv").write_text(composition_table(data))
    summary = {
        "complete": sum(r["status"] == "complete" for r in data["cases"]),
        "timeout": sum(r["status"] == "timeout" for r in data["cases"]),
        "resource_rejected": sum(r["status"] == "resource_rejected" for r in data["cases"]),
        "implementation_failure": sum(
            r["status"] == "implementation_failure" for r in data["cases"]
        ),
        "maximum_density_error": max(
            (r["maximum_density_error"] for r in data["cases"] if r["status"] == "complete"),
            default=None,
        ),
        "native_gate_execution": False,
        "physical_hardware_tested": False,
    }
    save_json(output / "SUMMARY.json", summary)
    save_json(
        output / "MANIFEST.json",
        {
            "files": {
                str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(output.rglob("*"))
                if p.is_file() and p.name != "MANIFEST.json" and p.suffix != ".tmp"
            }
        },
    )


def verify(output):
    output = Path(output).resolve()
    data = json.loads((output / "result.json").read_text())
    if (output / "resources.csv").read_text() != composition_table(data):
        raise ValueError("Composition resources table differs from raw records")
    manifest = json.loads((output / "MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        path = (output / name).resolve()
        if (
            not path.is_relative_to(output)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"Composition evidence checksum mismatch: {name}")
    return True


def worker(queue, size, condition, degree, output, settings):
    from qiskit import qpy, transpile
    from qiskit.quantum_info import Statevector, partial_trace

    from .polynomial_recurrent import build_polynomial_solver
    from .recurrent_quantum import build_recurrent_solver

    start = time.perf_counter()
    try:
        matrix = fixture(size, condition, "controlled_nonsymmetric", settings["seed"])
        rhs = np.random.default_rng(settings["seed"] + size).normal(size=size)
        base = build_recurrent_solver(matrix, rhs, max_encoding_dimension=16)
        prep = time.perf_counter() - start
        circuit = polynomial_composition_circuit(
            base,
            degree,
            1,
            max_controlled_unitary_qubits=settings["maximum_controlled_unitary_qubits"],
        )
        actual = partial_trace(
            Statevector(circuit), list(range(base.data_qubits, circuit.num_qubits))
        ).data
        polynomial = build_polynomial_solver(matrix, rhs, degree)
        initial = np.zeros(1 << base.data_qubits)
        initial[size] = 1
        rho = np.outer(initial, initial)
        c, f = polynomial.no_jump, polynomial.residual_operator
        expected = c @ rho @ c.T + np.trace(f @ rho @ f.T) * rho
        native = transpile(
            circuit,
            basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=1,
            seed_transpiler=42,
        )
        content = io.BytesIO()
        qpy.dump(native, content)
        name = f"n{size}-k{condition}-m{degree}.qpy"
        path = Path(output) / name
        temp = path.with_suffix(".tmp")
        temp.write_bytes(content.getvalue())
        temp.replace(path)
        queue.put(
            {
                "status": "complete",
                "preprocessing_seconds": prep,
                "wall_seconds": time.perf_counter() - start,
                "maximum_density_error": float(np.max(abs(actual - expected))),
                "trace": float(np.trace(actual).real),
                "width": native.num_qubits,
                "cx": native.count_ops().get("cx", 0),
                "depth": native.depth(),
                "qpy_file": name,
                "native_gate_execution": False,
                "simulation": "complete coherent circuit at residual-unitary level",
            }
        )
    except MemoryError as error:
        queue.put({"status": "resource_rejected", "reason": str(error)})
    except Exception as error:
        queue.put({"status": "implementation_failure", "reason": repr(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    settings = configuration(args.config, "polynomial-composition", DEFAULTS)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    records = []
    for size in settings["sizes"]:
        for condition in settings["conditions"]:
            for degree in settings["degrees"]:
                row = {"size": size, "condition": condition, "polynomial_degree": degree}
                row.update(composition_resources(size, degree, 1, "reset"))
                if (
                    size > 10
                    or row["largest_controlled_unitary_qubits"]
                    > settings["maximum_controlled_unitary_qubits"]
                ):
                    row.update(
                        status="resource_rejected",
                        reason="Input/control-unitary dense synthesis budget",
                    )
                else:
                    print(f"Composed N={size} kappa={condition} m={degree}", flush=True)
                    queue = context.Queue()
                    process = context.Process(
                        target=worker, args=(queue, size, condition, degree, output, settings)
                    )
                    start = time.perf_counter()
                    process.start()
                    process.join(settings["timeout_seconds"])
                    if process.is_alive():
                        process.terminate()
                        process.join()
                        row.update(
                            status="timeout",
                            timeout_seconds=settings["timeout_seconds"],
                            wall_seconds=time.perf_counter() - start,
                        )
                    else:
                        try:
                            row.update(queue.get(timeout=2))
                        except queue_module.Empty:
                            row.update(
                                status="implementation_failure",
                                reason=f"Worker exit {process.exitcode}",
                            )
                    queue.close()
                records.append(row)
                save_json(output / "progress.json", records)
    passed = all(
        r["maximum_density_error"] < 1e-9 for r in records if r["status"] == "complete"
    ) and all(r["status"] != "implementation_failure" for r in records)
    save_json(
        output / "result.json",
        {
            "configuration": settings,
            "provenance": provenance(Path.cwd()),
            "cases": records,
            "passed": passed,
            "physical_hardware_tested": False,
        },
    )
    build_report(output)
    verify(output)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
