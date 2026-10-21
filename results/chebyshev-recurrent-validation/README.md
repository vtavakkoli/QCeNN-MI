# Source-pinned Chebyshev recurrent validation

The corrected full campaign passed 26 stages and 281 unit tests. This is a software-integrity result; deliberately difficult solver cases and resource-limit rejections remain in the record. No physical hardware was executed.

`review-record.json` is a compact derivation from the full raw study, linked by SHA-256. It retains exact lower-bound-family call evaluations, small complete native cost cases, phase and density checks, synthetic native noise, aggregate static outcomes, logical-noise outcomes and controlled-baseline timeouts. The full inputs, trajectories, QPY files, logs and source snapshot are in the accompanying reproduction archive. Large raw payloads are not duplicated in this repository.

The executed local commit and its public counterpart have independently verified identical complete Git trees; their commit metadata differ. The source pin is the public commit `95605dbf79371860f4ea57bd1e6a5ffc910d41ac`. This data-only record is added after execution and does not retroactively change that pin.

Reproduce from the source pin:

```sh
python -m pip install -r requirements.lock
python -m pip install -e .
python -m qcenn_mi.validation --profile full --output-dir artifacts/fresh-chebyshev
```

## Interpretation

The signed Chebyshev response improves ideal augmented-state query scaling relative to the original recurrence in the specified real-Julia access model. It does not improve the best-known QLSP bound or establish a classical/quantum end-to-end speedup. The diagonal analytical family is classically trivial. Degree budgets reject native phase synthesis at conditions 100 and 1000; large-size results are reduced-channel simulations.

Cost examples select the update count offline, include local preprocessing and assembly/compilation, and report confidence-dependent ideal shot workloads under explicit assumed durations. Diagnostic simulation is excluded from compilation time. Reference-preparation gates are counted; zero-state qubits are assumed supplied at the start of an execution. These are not calibrated hardware times. Fixed-macro synthetic noise comparisons charge different gates and queries. The original high-conditioning power failures are preserved.
