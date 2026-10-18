# Compact inverse hardware-simulation preparation

Native ideal correctness: PASS

All synthetic-noise quality gates passed: False

This is synthetic hardware simulation, not physical or calibration-derived execution.

| Matrix | Data / total qubits | Ideal residual | Success | Routed CX | Depth |
|---|---:|---:|---:|---:|---:|
| 2 | 1 / 5 | 1.242e-07 | 68.1% | 22 | 72 |
| 4 | 2 / 6 | 1.572e-07 | 62.7% | 79 | 142 |
| 10 | 4 / 8 | 1.126e-07 | 60.2% | 406 | 1033 |
| 100 | 7 / 11 | 1.033e-07 | 60.0% | 14625 | 41024 |

## Synthetic native-gate noise

| Matrix | Profile | Shots | Accepted | Echo return | 95% interval | Gate |
|---|---|---:|---:|---:|---|---|
| 2 | optimistic_synthetic | 32768 | 21946 | 0.9932561742458762 | [0.9920836930920269, 0.9942560051065583] | PASS |
| 2 | stress_synthetic | 32768 | 20519 | 0.9715385740045811 | [0.969173567104536, 0.9737270560293508] | PASS |
| 4 | optimistic_synthetic | 32768 | 20176 | 0.9859238699444886 | [0.9842033677383701, 0.9874593700493572] | PASS |
| 4 | stress_synthetic | 32768 | 18324 | 0.9346758349705304 | [0.9310062311612347, 0.9381632253616076] | PASS |
| 10 | optimistic_synthetic | 32768 | 19180 | 0.9531282586027111 | [0.9500451798454567, 0.9560298644730413] | PASS |
| 10 | stress_synthetic | 32768 | 16348 | 0.7010031808172253 | [0.6939386952571533, 0.7079732249721649] | FAIL |
| 100 | optimistic_synthetic | 64 | 34 | 0.23529411764705882 | [0.12436985532639575, 0.3999613792253638] | FAIL |

A passing native simulation does not imply noisy or physical hardware accuracy.
Poor noisy quality is reported, not hidden by classical correction or a passing CLI exit.
The verification circuit appends an independently computed reference-state projection.
QPY contains action and verifier circuits; a real backend needs recompilation with its target/calibration.

Provisional quality gate: at least 20 accepted shots and 95% Wilson lower bound >= 0.9.
