"""A2: local real quantum circuits with interaction weights shared by columns.

Each inverse column is amplitude encoded in log2(n) qubit cells. Classical
scales retain magnitude. This is a small hybrid variational experiment, not a
quantum implementation of dense matrix multiplication or a speed-up claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import least_squares


def validate_matrix(matrix):
    raw = np.asarray(matrix)
    if np.iscomplexobj(raw):
        raise ValueError("Only real matrices are supported")
    m = np.asarray(raw, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] not in (2, 4):
        raise ValueError("Simulator prototype supports 2x2 or 4x4 matrices")
    if not np.isfinite(m).all() or np.linalg.matrix_rank(m) < len(m):
        raise ValueError("Matrix must be finite and nonsingular")
    return m


def local_template(rotations, interactions):
    """RY layers and nearest-neighbor controlled-RY edges on an open chain.

    rotations has shape (depth+1, qubits); interactions (depth, qubits-1).
    The latter is the shared template, reused for every inverse column.
    Qiskit little-endian basis ordering is used throughout.
    """
    rotations = np.asarray(rotations, dtype=float)
    interactions = np.asarray(interactions, dtype=float)
    if rotations.ndim != 2 or min(rotations.shape) < 1:
        raise ValueError("rotations must have shape (depth+1, qubits)")
    depth, qubits = rotations.shape[0] - 1, rotations.shape[1]
    if interactions.shape != (depth, qubits - 1):
        raise ValueError("Invalid interaction template shape")
    if not np.isfinite(rotations).all() or not np.isfinite(interactions).all():
        raise ValueError("Circuit parameters must be finite")
    qc = QuantumCircuit(qubits, name="qcenn_local")
    for layer in range(depth + 1):
        for q in range(qubits):
            qc.ry(float(rotations[layer, q]), q)
        if layer < depth:
            for q in range(qubits - 1):
                qc.cry(float(interactions[layer, q]), q, q + 1)
    return qc


def direction(circuit):
    return np.asarray(Statevector.from_instruction(circuit).data.real)


def readout_direction(circuit, *, backend=None, shots=8192, seed=42, postselect=False, audit=None):
    """Full Pauli tomography; principal eigenvector gives a pure-state estimate.

    Includes Y settings because real entangled states can have YY correlations.
    No statevector or target inverse enters measurement reconstruction. Cost is
    3**qubits settings; mixed/noisy states are projected onto a pure real state.
    Returns direction, raw density minimum eigenvalue and raw purity.
    With postselect=True the highest-index qubit is measured in Z and only
    ancilla=0 counts enter data-qubit tomography. Optional audit captures raw
    counts, job ID, retained shots, backend name and compiled resource counts.
    Backend must implement the Qiskit Backend.run interface. A simulator seed
    is passed only to Aer; physical backends receive no simulator options.
    """
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ValueError("shots must be a positive integer")
    backend = AerSimulator() if backend is None else backend
    n = circuit.num_qubits - int(postselect)
    if n < 1:
        raise ValueError("Readout requires at least one data qubit")
    settings = list(product("XYZ", repeat=n))
    circuits = []
    for setting in settings:
        qc = circuit.copy()
        for q, basis in enumerate(setting):
            if basis == "X":
                qc.h(q)
            elif basis == "Y":
                qc.sdg(q)
                qc.h(q)
        qc.measure_all()
        circuits.append(qc)
    compiled = transpile(circuits, backend, optimization_level=1, seed_transpiler=seed)
    options = {"shots": shots}
    if isinstance(backend, AerSimulator):
        options["seed_simulator"] = seed
    job = backend.run(compiled, **options)
    result = job.result()
    raw_counts = [result.get_counts(i) for i in range(len(settings))]
    selected = []
    for counts in raw_counts:
        if postselect:
            counts = {
                bits: count
                for bits, count in counts.items()
                if bits.replace(" ", "")[-1 - n] == "0"
            }
        if sum(counts.values()) == 0:
            raise RuntimeError("No accepted postselection shots; increase shot budget")
        selected.append(counts)
    if audit is not None:
        audit.update(
            job_id=job.job_id(),
            backend=str(backend.name),
            shots=shots,
            settings=["".join(setting) for setting in settings],
            raw_counts=raw_counts,
            retained_shots=[sum(c.values()) for c in selected],
            compiled_depths=[qc.depth() for qc in compiled],
            compiled_operations=[dict(qc.count_ops()) for qc in compiled],
            postselect_ancilla_zero=bool(postselect),
        )
    rho = np.zeros((2**n, 2**n), dtype=complex)
    for labels in product("IXYZ", repeat=n):
        setting = tuple("Z" if p == "I" else p for p in labels)
        counts = selected[settings.index(setting)]
        active = [q for q, p in enumerate(labels) if p != "I"]
        total = sum(counts.values())
        expectation = (
            sum(
                count * (-1) ** sum(int(bits.replace(" ", "")[-1 - q]) for q in active)
                for bits, count in counts.items()
            )
            / total
        )
        rho += expectation * Pauli("".join(reversed(labels))).to_matrix() / 2**n
    eigenvalues, eigenvectors = np.linalg.eigh(rho.real)
    state = eigenvectors[:, -1]
    return state, float(eigenvalues[0]), float(np.trace(rho @ rho).real)


@dataclass
class CoupledResult:
    matrix: np.ndarray
    estimate: np.ndarray
    rotations: np.ndarray
    interactions: np.ndarray
    evaluations: int
    residual: float
    success: bool

    def circuits(self):
        return [local_template(r, self.interactions) for r in self.rotations]

    def measure(self, *, backend=None, shots=8192, seed=42, audit=None):
        columns, diagnostics = [], []
        for j, circuit in enumerate(self.circuits()):
            record = {}
            state, minimum, purity = readout_direction(
                circuit, backend=backend, shots=shots, seed=seed + j, audit=record
            )
            if audit is not None:
                audit.append(record)
            transformed = self.matrix @ state
            scale = transformed[j] / (transformed @ transformed)
            columns.append(scale * state)
            diagnostics.append({"raw_min_eigenvalue": minimum, "raw_purity": purity})
        estimate = np.column_stack(columns)
        return estimate, diagnostics


def solve_coupled(matrix, *, depth=2, initial=None, max_nfev=400, seed=42, tolerance=1e-7):
    """Joint residual solve with shared interactions and per-column cell rotations.

    ``initial`` warm starts circuit parameters, never an exact inverse. The
    optimizer evaluates genuine Qiskit statevectors; classical scales are
    eliminated analytically. Failure to reach tolerance is reported explicitly.
    """
    m = validate_matrix(matrix)
    if not isinstance(depth, int) or depth < 1 or max_nfev < 1 or tolerance <= 0:
        raise ValueError("Invalid solver budget, depth or tolerance")
    n = len(m)
    qubits = int(np.log2(n))
    shape = (n, depth + 1, qubits)
    edge_shape = (depth, qubits - 1)
    split = int(np.prod(shape))
    if initial is None:
        rng = np.random.default_rng(seed)
        rotations = rng.normal(0, 0.15, shape)
        # Start near basis-vector columns without computing the inverse.
        for j in range(n):
            for q in range(qubits):
                rotations[j, 0, q] += np.pi * ((j >> q) & 1)
        edges = np.full(edge_shape, np.pi / 2)
    else:
        rotations, edges = initial.rotations.copy(), initial.interactions.copy()
        if rotations.shape != shape or edges.shape != edge_shape:
            raise ValueError("Warm start dimensions/depth do not match")
    x0 = np.concatenate([rotations.ravel(), edges.ravel()])

    def unpack(parameters):
        return parameters[:split].reshape(shape), parameters[split:].reshape(edge_shape)

    def estimate(parameters):
        rotations, edges = unpack(parameters)
        columns = []
        for j in range(n):
            state = direction(local_template(rotations[j], edges))
            transformed = m @ state
            columns.append(state * transformed[j] / (transformed @ transformed))
        return np.column_stack(columns)

    opt = least_squares(
        lambda p: (m @ estimate(p) - np.eye(n)).ravel(),
        x0,
        max_nfev=max_nfev,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    value = estimate(opt.x)
    residual = float(np.linalg.norm(m @ value - np.eye(n)))
    rotations, edges = unpack(opt.x)
    return CoupledResult(
        m.copy(), value, rotations.copy(), edges.copy(), opt.nfev, residual, residual < tolerance
    )
