import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qcenn_mi.recurrent_cost import (
    amortized_cost,
    certified_readout_budget,
    coherence_readout_rotations,
    reset_break_even,
    scheduled_latency,
)


def test_asymmetric_reset_topology_is_scheduled_and_precision_charges_all_settings():
    circuit = QuantumCircuit(3)
    circuit.reset([0, 1])
    circuit.cx(0, 2)
    circuit.cx(1, 2)
    assert scheduled_latency(circuit, cx_seconds=2, one_qubit_seconds=1, reset_seconds=10) == 14
    with pytest.raises(ValueError):
        scheduled_latency(circuit, cx_seconds=-1, one_qubit_seconds=1, reset_seconds=1)
    budget = certified_readout_budget(10, 0.1, 0, residual_target=0.01)
    assert budget["total_executions"] == 11 * budget["shots_per_setting"]
    assert certified_readout_budget(10, 0.1, 0.01, residual_target=0.01)["total_executions"] is None
    assert amortized_cost(compile_seconds=12, reuse=3, executions=10, circuit_seconds=2) == 24


def test_notes_break_even_is_conditional_and_not_a_physical_latency_measurement():
    threshold = reset_break_even(99, 13, 17)
    assert threshold["primitive_time_coefficient"] == pytest.approx(122 / 86)
    assert threshold["extra_control_time_coefficient"] == pytest.approx(13 / 86)
    assert reset_break_even(5, 10, 2)["status"] == "no_positive_reset_saving"


def test_readout_basis_measures_the_signed_coherence_observable():
    circuits = coherence_readout_rotations(2)
    assert len(circuits) == 3
    for i, circuit in enumerate(circuits[:2]):
        unitary = Operator(circuit).data
        outcomes = np.zeros(4)
        outcomes[i] = 0.5
        outcomes[2] = -0.5
        target = np.zeros((4, 4))
        target[i, 2] = target[2, i] = 0.5
        assert np.max(abs(unitary.conj().T @ np.diag(outcomes) @ unitary - target)) < 1e-12
