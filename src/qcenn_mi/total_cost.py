"""Transparent all-snapshot/final-only cost accounting for temporal QCeNN channels.

Costs are *assumption-driven resource equivalents*, not hardware measurements.
Destructive readout consumes a state; a resident path needs complete replay
(or independently propagated copies) for each measured checkpoint. The same
number of settings/shots is charged to each contender. Classical preparation
and oracle synthesis are charged once per *unique input snapshot* as a
declared optimistic amortization assumption, not per replayed copy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CostWeights:
    oracle_call: float = 1.0
    reference_preparation: float = 5.0
    macro_reset: float = 2.0
    measurement: float = 0.1
    compile_snapshot: float = 10.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative cost")
        if not (self.oracle_call or self.macro_reset
                or self.reference_preparation or self.measurement):
            raise ValueError("At least one physical operation cost must be positive")


def checkpoints_for(horizon: int, every: int) -> tuple[int, ...]:
    """0-based sampled checkpoints, always including the final state.

    every=0 means final-only; otherwise sample after every 'every' snapshots
    (counted from the start) and always include the final one.
    """
    if isinstance(horizon, bool) or horizon < 2:
        raise ValueError("horizon must be at least two")
    if isinstance(every, bool) or every < 0:
        raise ValueError("readout interval must be a nonnegative integer")
    if every == 0:
        return (horizon - 1,)
    return tuple(sorted(set(range(every - 1, horizon, every)) | {horizon - 1}))


def resource_cost(
    method: str, *, degree: int, updates: int, checkpoints: tuple[int, ...],
    copies_per_checkpoint: int, weights: CostWeights,
    accepted_probabilities: dict[int, float] | None = None,
    native_macro_equivalent: float | None = None,
) -> dict:
    """Cost of obtaining measurements from *independent accepted copies*.

    Resident: each readout checkpoint requires its own path from t=0.
    Fresh: only construct the current snapshot, not all preceding snapshots.
    Direct: one fresh postselected state per accepted output, with geometric
    retries expected; only oracle-call mode is supported for direct filtering.

    When native_macro_equivalent is supplied, cost the complete *compiled
    one-macro circuit* per repeated macro, as an explicit gate-weighted
    extrapolation. Do not double count individual oracle and reset costs;
    this is not a transpiled K-step changing-input native circuit.
    """
    if method not in {"resident", "fresh", "direct"}:
        raise ValueError("method must be resident, fresh or direct")
    if (isinstance(degree, bool) or not isinstance(degree, int) or degree < 1
            or isinstance(updates, bool) or not isinstance(updates, int) or updates < 1):
        raise ValueError("degree and updates must be positive integers")
    if not checkpoints or sorted(set(checkpoints)) != list(checkpoints) or min(checkpoints) < 0:
        raise ValueError("checkpoints must be sorted nonnegative distinct indices")
    if isinstance(copies_per_checkpoint, bool) or copies_per_checkpoint < 1:
        raise ValueError("copies_per_checkpoint must be positive")
    if native_macro_equivalent is not None and (
        not math.isfinite(native_macro_equivalent) or native_macro_equivalent < 0
        or method == "direct"
    ):
        raise ValueError("native macro proxy supports recurrent channels only")

    copies = len(checkpoints) * copies_per_checkpoint
    measurements = copies
    if method == "resident":
        macro_updates = updates * copies_per_checkpoint * sum(t + 1 for t in checkpoints)
        expected_preparations = float(copies)
        compilation_snapshots = max(checkpoints) + 1
        calls = float(2 * degree * macro_updates)
    elif method == "fresh":
        macro_updates = updates * copies
        expected_preparations = float(copies)
        compilation_snapshots = len(checkpoints)
        calls = float(2 * degree * macro_updates)
    else:
        macro_updates = 0
        compilation_snapshots = len(checkpoints)
        if not accepted_probabilities:
            raise ValueError("Direct postselection success probabilities are required")
        if any(t not in accepted_probabilities or
               not math.isfinite(accepted_probabilities[t]) or
               not 0 < accepted_probabilities[t] <= 1 for t in checkpoints):
            raise ValueError("Direct success probabilities must be in (0,1]")
        retries = sum(1 / accepted_probabilities[t] for t in checkpoints)
        expected_preparations = copies_per_checkpoint * retries
        calls = float(2 * degree * expected_preparations)

    readout_cost = measurements * weights.measurement
    compilation_cost = compilation_snapshots * weights.compile_snapshot
    prep_cost = expected_preparations * weights.reference_preparation
    if native_macro_equivalent is None:
        quantum_work = calls * weights.oracle_call + macro_updates * weights.macro_reset
        model = "weighted_primitive_oracle_calls"
    else:
        quantum_work = macro_updates * native_macro_equivalent
        model = "single_macro_compilation_gate_weight_proxy"
    total = quantum_work + prep_cost + readout_cost + compilation_cost
    return {
        "method": method,
        "model": model,
        "total_equivalent_units": float(total),
        "quantum_work_units": float(quantum_work),
        "preparation_units": float(prep_cost),
        "readout_units": float(readout_cost),
        "compilation_units": float(compilation_cost),
        "primitive_oracle_calls": calls,
        "macro_updates": macro_updates,
        "expected_reference_preparations": float(expected_preparations),
        "destructive_measurement_executions": measurements,
        "compiled_input_snapshots": compilation_snapshots,
        "copies_per_checkpoint": copies_per_checkpoint,
        "readout_checkpoints": list(checkpoints),
        "cost_assumptions": (
            "Readout consumes each prepared state; resident trajectories "
            "are independently replayed from t=0 for every readout copy. "
            "Previously prepared unknown states are never cloned. "
            "Classical input preparation and channel compilation are cached "
            "per unique snapshot across copies, not measured. "
            "No classical wall-clock speedup or physical QPU benchmark."
        ),
    }


def native_macro_lookup(
    evidence: dict, *, size: int, condition: float, degree: int,
    cx_equivalent: float = 1.0, single_equivalent: float = 0.1,
    reset_equivalent: float = 1.0,
) -> float | None:
    """Optional assumption-driven gate proxy for matching *small* QSP audits.

    A count compiled for a diagonal input is NOT an exact count for a
    varying-input schedule. Never infer counts for unsupported degrees.
    """
    for item in evidence.get("records", []):
        if (item["size"] == size and abs(item["condition"] - condition) < 1e-9
                and item["filter_degree"] == degree
                and item.get("status") == "compiled_small_complete_circuit"):
            counts = item["native_counts_line_topology"]
            return float(
                counts.get("cx", 0) * cx_equivalent
                + (counts.get("sx", 0) + counts.get("x", 0)) * single_equivalent
                + counts.get("reset", 0) * reset_equivalent
            )
    return None
