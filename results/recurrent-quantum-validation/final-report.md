# QCeNN-MI final validation report

Run ID: `qcenn-933a9b1ce82241e9803927d8a08cec11`

Source: `82dcfeca1d88fccfb94ee4cca45adf01af00d601`; source SHA-256: `1f1b7412762d8240a6a99158a4a9c73e326705efbc4a301dbcfbff5f8155ee0c`.

Execution context: local-process. Software stages before report: PASS.

Synthetic-noise quality: True. Physical-device validation: NOT RUN.

The software verdict checks implementation and declared numerical acceptance criteria. It does not turn failed noise gates into successful hardware results.

## Requested implementation and evidence

| Item | Status | Evidence and limits |
|---|---|---|
| P0-1 | implemented | historical-reconciliation.json; publication-evidence/manifest.json. Both historical runs retained separately; Table III/IV discrepancies reconciled, no causal attribution invented. |
| P0-2 | CI release workflow provided; release attachment requires a successful tagged run | validation.json; SHA256SUMS.json; PACKAGE_SHA256; source CITATION.cff. Only actual CI-generated assets are attached to the candidate tag; local execution is explicitly labeled. |
| P1-1 | not run or failed | temporal-refresh/result.json; tables/refresh-intervals.csv. Positive and failed accuracy regimes, Eq. (24)/(25), acquisition error, drift and repetition cost; physical acquisition and deadlines not measured. |
| P1-2 | not run or failed | noise-verification/result.json; hardware/audit/. Multiple seeds, noise grid, separate solver/verifier/readout controls and intervals. Order-100 statistical power remains unresolved at the baseline 64 shots. |
| P1-3 | not run or failed | baselines/result.json. SciPy, LU, QR and SVD timings; HHL/QSVT analytical access-model comparison only; no quantum speedup ratio. |
| P1-4 | not run or failed | resources/result.json; hardware/result.json; validation.json. Conditioning through 10000, preprocessing, basis/angle storage, attempts, readout and routing. Native counts omitted where budgets prevent synthesis. |
| P1-5 | not run or failed | operator-audit/result.json; unit-tests.xml. Input-independent native full-state checks, unseen RHS, both residuals, failure sectors, singularity and runtime reference-solve guards. |
| P2 | implemented | configuration.json; validation.json; publication-evidence/manifest.json; source docs/publication-evidence.md. Explicit profiles, CLI settings, schemas, failure semantics, provenance, preserved citation license, and an offline report. |
| Recurrent quantum state updates | small-circuit simulator prototype passed | recurrent-quantum/result.json; recurrent-quantum/*.qpy. Complete coherent/reset quantum circuits without classical iterate updates; RHS-dependent residual-channel construction, separate from Chebyshev inverse action. Dense preprocessing, synthetic CX-only noise and preprogrammed snapshots do not establish live tracking or hardware performance. |
| Physical device validation | not run | No physical backend, calibration snapshot, job IDs or hardware counts supplied. Hardware readiness and synthetic emulation are not physical performance evidence. |

## Historical discrepancy

The manuscript depth sequence 66/138/1032/41033 and order-100 accepted count 35/64 (echo 10/35) match the clean e6769dbe benchmark. The original public archive retains 72/142/1033/41024 and 34/64 (echo 8/34). Both are separate simulator runs; neither is averaged or overwritten. Matching package versions do not prove identical source, compiled circuits or stochastic execution. The older archive lacks a complete immutable run manifest, so a unique root cause cannot be established.

## Measured results

## Key findings

- temporal_positive_accuracy_cases: None
- intentional_stale_operator_failures: None
- seeded_noise_samples: None
- failed_pooled_noisy_quality_gates: None
- strict_native_operator_audits: None
- optimization_comparisons_failing_full_state_target: None
- worst_1000_operator_residual: None
- recurrent_quantum_circuit_cases: 60
- recurrent_quantum_classical_iterate_updates: 0
- recurrent_quantum_synthetic_quality_passed: True

### native-resources

0 rows; source `hardware/result.json`; CSV SHA-256 `2faad0696670d1deaa6493d9a84943c13ca7f55f9d5e6642b850572aa0996356`.

| size | required_qubits | native_action_depth | native_action_operations | native_ideal_residual |
|---|---|---|---|---|

Full values and raw audit files are in the validation bundle.

### native-noise

0 rows; source `hardware/result.json`; CSV SHA-256 `e673f3f4b2811f25d34b12927cc6a961eeb8b1a2b0706633f35f980481e12745`.

| size | profile | shots | accepted | observed_echo_return | echo_return_95ci | quality_gate |
|---|---|---|---|---|---|---|

Full values and raw audit files are in the validation bundle.

### precision

0 rows; source `precision-1000/result.json`; CSV SHA-256 `52837d356e4b2fd162f42e38fb3f5dd566f622018f8d08e4a7e030e5daee22d2`.

| size | family | condition_number | degree | operator_relative_residual_2 | minimum_success_probability_all_inputs |
|---|---|---|---|---|---|

Full values and raw audit files are in the validation bundle.

### refresh-accuracy

0 rows; source `temporal-refresh/result.json`; CSV SHA-256 `ae5a659b103f4a104b61fb10352569b819a91a44603545d300a252474362cc48`.

| size | condition | family | update_time | noise_to_gap_ratio | refresh_age | left_residual | right_residual | tracking_bound_eq24 |
|---|---|---|---|---|---|---|---|---|

Full values and raw audit files are in the validation bundle.

### refresh-intervals

0 rows; source `temporal-refresh/result.json`; CSV SHA-256 `65c5f87511f7761218bade5a2e7939e945c4ddeeef3a671210ff5668d1679d52`.

| size | condition | family | noise_to_gap_ratio | target | sufficient_refresh_age_eq25 | empirical_maximum_sampled_feasible_age | sufficient_condition_validated |
|---|---|---|---|---|---|---|---|

Full values and raw audit files are in the validation bundle.

### noise-controls

0 rows; source `noise-verification/result.json`; CSV SHA-256 `505b883ce4b3d406000580019a3c19880f56860c3125a6f8489582d943183dd0`.

| size | decomposition | optimization_level | profile | noise_scale | control | shots | accepted | observed_echo_return | echo_return_95ci | effective_good_output_probability | quality_gate |
|---|---|---|---|---|---|---|---|---|---|---|---|

Full values and raw audit files are in the validation bundle.

### optimization-resources

0 rows; source `noise-verification/result.json`; CSV SHA-256 `00701c61a944748fd6ec80c8e18e3f3d024ce951a34a84b37ad4292fb15a1651`.

| size | decomposition | optimization_level | native_depth | single_qubit_gates | two_qubit_gates | synthesis_and_transpile_seconds | native_ideal_residual | passed |
|---|---|---|---|---|---|---|---|---|

Full values and raw audit files are in the validation bundle.

### refresh-latency

0 rows; source `temporal-refresh/result.json`; CSV SHA-256 `259ae1f5adff07b5e8b736d58d3b71ddf645269de414dd1f0c3e1edbafae9465`.

| size | condition | family | noise_to_gap_ratio | update_time | encoding_and_polynomial_seconds | isolated_svd_seconds_not_added_twice | native_synthesis_seconds | native_transpile_seconds | native_gate_simulation_seconds | physical_acquisition_seconds |
|---|---|---|---|---|---|---|---|---|---|---|

Full values and raw audit files are in the validation bundle.

### solver-baselines

0 rows; source `baselines/result.json`; CSV SHA-256 `df68ac196a55b49d4ddb8a301402c4afdb96689b4de00799e2ed78735ae15173`.

| size | condition | family | method | target_tolerance | median_solve_seconds | median_end_to_end_seconds | end_to_end_simulator_seconds | relative_residual |
|---|---|---|---|---|---|---|---|---|

Full values and raw audit files are in the validation bundle.

### conditioning-resources

0 rows; source `resources/result.json`; CSV SHA-256 `674ac4230663660e164a997f83b9700417a5083af5789f56444e78a5ebb82cbe`.

| size | condition | target_tolerance | status | degree | raw_qubits | encoded_operator_residual | minimum_success_probability | native_cx | native_depth |
|---|---|---|---|---|---|---|---|---|---|

Full values and raw audit files are in the validation bundle.

### operator-correctness

0 rows; source `operator-audit/result.json`; CSV SHA-256 `d867a0bfe52df2d7714aaa8dc62aa8ef4af0c6e7e74acf5cf1152a0d0b198f34`.

| size | family | left_operator_residual | right_operator_residual | analytic_residual_bound | native_depth | native_cx | passed |
|---|---|---|---|---|---|---|---|

Full values and raw audit files are in the validation bundle.

### recurrent-convergence

60 rows; source `recurrent-quantum/result.json`; CSV SHA-256 `743ae8b0cc3b5d9c9f402e30ae8242c1fbc81dddcea153958a87a0721aa6999c`.

| size | family | condition | steps | qubits | augmented_fidelity | conditional_solution_fidelity | coherence_relative_residual | solution_probability | augmented_infidelity_upper_bound | certificate_passed |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | controlled_spd | 2.0 | 0 | 5 | 0.693975090232497 | None | 1.0 | 0.0 | 0.5 | True |
| 2 | controlled_spd | 2.0 | 1 | 5 | 0.8327701082789966 | 0.6246438118107247 | 0.550957044581581 | 0.08976792187020649 | 0.44999999999999996 | True |
| 2 | controlled_spd | 2.0 | 4 | 5 | 0.92741342150247 | 0.8105626298785344 | 0.31695677204427164 | 0.18595475936460326 | 0.3280499999999999 | True |
| 2 | controlled_spd | 2.0 | 16 | 5 | 0.995026106831225 | 0.9909702039773587 | 0.0762710472638119 | 0.2729445745116788 | 0.0926510094425919 | True |
| 2 | controlled_spd | 2.0 | 64 | 5 | 0.9999998890784698 | 0.9999998203653007 | 0.00011612871411635409 | 0.30597190538372904 | 0.0005895092288869255 | True |
| 2 | controlled_nonsymmetric | 2.0 | 0 | 5 | 0.693975090232497 | None | 1.0 | 0.0 | 0.5 | True |
| 2 | controlled_nonsymmetric | 2.0 | 1 | 5 | 0.8327701082789967 | 0.6246438118107239 | 0.5509570445815807 | 0.08976792187020666 | 0.45 | True |
| 2 | controlled_nonsymmetric | 2.0 | 4 | 5 | 0.927413421502471 | 0.8105626298785334 | 0.3169567720442723 | 0.1859547593646033 | 0.32805 | True |

Full values and raw audit files are in the validation bundle.

