import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Operator

from qcenn_mi.experiment_io import audit_qpy, save_qpy_atomic


def test_archived_circuit_is_independently_readable(tmp_path):
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    path = tmp_path / "complete.qpy"
    save_qpy_atomic(path, circuit)
    with path.open("rb") as handle:
        recovered = qpy.load(handle)[0]
    assert np.allclose(Operator(circuit).data, Operator(recovered).data)
    assert audit_qpy(tmp_path) == {"complete.qpy": 1}


def test_failed_serialization_keeps_previously_published_bytes(tmp_path, monkeypatch):
    path = tmp_path / "circuit.qpy"
    path.write_bytes(b"previous immutable payload")

    def fail(circuits, buffer):
        buffer.write(b"QISKIT partial")
        raise RuntimeError("Interrupted serializer")

    monkeypatch.setattr(qpy, "dump", fail)
    with pytest.raises(RuntimeError):
        save_qpy_atomic(path, QuantumCircuit(1))
    assert path.read_bytes() == b"previous immutable payload"


def test_integrity_audit_rejects_empty_and_truncated_circuits(tmp_path):
    path = tmp_path / "bad.qpy"
    for broken in [b"", b"QISKIT truncated"]:
        path.write_bytes(broken)
        with pytest.raises(ValueError, match="Unreadable circuit artifact"):
            audit_qpy(tmp_path)
