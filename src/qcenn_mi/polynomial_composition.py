"""Compose the polynomial map from controlled baseline residual unitaries.

No F_m factorization is performed. Coherent flags record the first residual
branch; resetting the data is deferred until all ell baseline calls finish.
The construction charges ell calls and their extra controls explicitly.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, UnitaryGate

from .recurrent_quantum import _integer, _prepare_reference


def composition_resources(size, degree, steps, workspace="reset"):
    _integer(size, "size", 2)
    _integer(degree, "polynomial_degree")
    _integer(steps, "steps")
    if workspace not in ["reset", "coherent"]:
        raise ValueError("workspace must be reset or coherent")
    n, ell = size.bit_length(), degree + 1
    width = 2 * n + ell if workspace == "reset" else n + steps * (n + ell)
    return {
        "data_qubits": n,
        "total_qubits": width,
        "polynomial_degree": degree,
        "macro_updates": steps,
        "residual_unitary_calls": ell * steps,
        "extra_control_incidents": steps * ell * (ell - 1) // 2,
        "controlled_swaps": steps * ell * n,
        "auxiliary_resets": max(0, steps - 1) * (n + ell) if workspace == "reset" else 0,
        "largest_controlled_unitary_qubits": n + ell,
        "intermediate_data_measurements": 0,
    }


def polynomial_composition_circuit(
    baseline,
    degree,
    steps,
    *,
    workspace="coherent",
    initial_state=None,
    max_qubits=24,
    max_controlled_unitary_qubits=8,
):
    """Realize E_m without classically computing G_m or the inverse solution.

    Calls to the supplied baseline residual U_F are controlled on all previous
    flags being zero. At most one flag is raised. Each flagged branch swaps its
    data with a known reference environment only after the final residual call.
    Reset mode reuses flags/environment on the device between macro updates.
    """
    resources = composition_resources(baseline.size, degree, steps, workspace)
    if resources["total_qubits"] > max_qubits:
        raise MemoryError("Composed circuit exceeds the declared qubit budget")
    if resources["largest_controlled_unitary_qubits"] > max_controlled_unitary_qubits:
        raise MemoryError("Controlled-unitary synthesis exceeds the declared dense budget")
    n, ell = baseline.data_qubits, degree + 1
    circuit = QuantumCircuit(resources["total_qubits"])
    data = list(range(n))
    if initial_state is None:
        _prepare_reference(circuit, data, baseline.reference_index)
    else:
        state = np.asarray(initial_state, dtype=complex)
        if state.shape != (1 << n,) or not np.isclose(np.linalg.norm(state), 1):
            raise ValueError("initial state must be normalized and match the register")
        circuit.append(StatePreparation(state), data)
    residual = UnitaryGate(baseline.collision_unitary, label="input U_F")
    for macro in range(steps):
        start = n if workspace == "reset" else n + macro * (ell + n)
        flags = list(range(start, start + ell))
        environment = list(range(start + ell, start + ell + n))
        if workspace == "reset" and macro:
            circuit.reset(flags + environment)
        _prepare_reference(circuit, environment, baseline.reference_index)
        for j, flag in enumerate(flags):
            if j:
                gate = residual.control(j, ctrl_state=0)
                circuit.append(gate, flags[:j] + data + [flag])
            else:
                circuit.append(residual, data + [flag])
        for flag in flags:
            for first, second in zip(data, environment, strict=True):
                circuit.cswap(flag, first, second)
    circuit.metadata = {"polynomial_composition": resources, "target_solution_loaded": False}
    return circuit
