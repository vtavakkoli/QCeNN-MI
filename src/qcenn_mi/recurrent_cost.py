"""Explicit assumed-latency and confidence costs, not measured device timing."""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate


def coherence_readout_rotations(size):
    """Small explicit basis rotations for N coherences and reference population."""
    if not 2 <= size <= 10:
        raise ValueError("dense readout synthesis is restricted to small explicit circuits")
    n = size.bit_length()
    circuits = []
    for index in range(size):
        rotation = np.eye(1 << n)
        rotation[np.ix_([index, size], [index, size])] = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        circuit = QuantumCircuit(n)
        circuit.append(UnitaryGate(rotation, label=f"coherence {index}"), range(n))
        circuits.append(circuit)
    circuits.append(QuantumCircuit(n))
    return circuits


def certified_readout_budget(
    size, singular_ratio, purified_error, *, residual_target, failure=0.05
):
    """Hoeffding budget from the paper's conservative coherence-readout bound.

    Returns a rejection when state bias already consumes the precision budget.
    Shots include all N real coherences plus the reference population.
    """
    if size < 2 or not 0 < singular_ratio <= 1 or not 0 <= purified_error < 1:
        raise ValueError("invalid dimension, singular ratio or state error")
    if residual_target <= 0 or not 0 < failure < 1:
        raise ValueError("positive tolerance and failure probability in (0,1) required")
    e, gamma = purified_error, singular_ratio
    available = residual_target * gamma * (0.5 - e) - 2 * e
    if available <= 0:
        return {"status": "state_error_exceeds_readout_certificate", "total_executions": None}
    precision = available / (np.sqrt(size) + 1 + residual_target * gamma)
    shots = math.ceil(math.log(2 * (size + 1) / failure) / (2 * precision**2))
    return {
        "status": "bounded",
        "measurement_settings": size + 1,
        "absolute_observable_precision": float(precision),
        "shots_per_setting": shots,
        "total_executions": (size + 1) * shots,
        "confidence": 1 - failure,
        "residual_target": residual_target,
        "model": "independent bounded measurements; ideal readout settings; no device evidence",
    }


def scheduled_latency(circuit, *, cx_seconds, one_qubit_seconds, reset_seconds):
    """Dependency-respecting ASAP schedule with declared durations.

    Resets on distinct qubits may overlap. rz is virtual. No crosstalk, routing,
    classical latency or calibration is silently included in this model.
    """
    if min(cx_seconds, one_qubit_seconds, reset_seconds) < 0:
        raise ValueError("gate durations must be nonnegative")
    ready = np.zeros(circuit.num_qubits)
    durations = {
        "cx": cx_seconds,
        "sx": one_qubit_seconds,
        "x": one_qubit_seconds,
        "rz": 0.0,
        "reset": reset_seconds,
        "barrier": 0.0,
    }
    for item in circuit.data:
        name = item.operation.name
        if name not in durations:
            raise ValueError(f"unsupported native instruction: {name}")
        indices = [circuit.find_bit(q).index for q in item.qubits]
        if indices:
            ready[indices] = float(max(ready[indices])) + durations[name]
    return float(max(ready, default=0))


def amortized_cost(
    *, compile_seconds, reuse, executions, circuit_seconds, prepare_seconds=0.0, readout_seconds=0.0
):
    if (
        reuse < 1
        or executions < 1
        or min(compile_seconds, circuit_seconds, prepare_seconds, readout_seconds) < 0
    ):
        raise ValueError("positive reuse/executions and nonnegative times required")
    return compile_seconds / reuse + executions * (
        prepare_seconds + circuit_seconds + readout_seconds
    )


def reset_break_even(baseline_updates, filtered_updates, order):
    """Equal bulk management cost per macro only; excludes actual reset topology."""
    if min(baseline_updates, filtered_updates, order) < 1:
        raise ValueError("update counts and order must be positive")
    difference = baseline_updates - filtered_updates
    if difference <= 0:
        return {"status": "no_positive_reset_saving"}
    return {
        "status": "conditional_threshold",
        "primitive_time_coefficient": (order * filtered_updates - baseline_updates) / difference,
        "extra_control_time_coefficient": filtered_updates / difference,
        "assumption": "same bulk reset/management latency per macro; actual ancilla resets differ",
    }
