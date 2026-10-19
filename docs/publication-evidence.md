# Publication evidence and claim traceability

The manuscript is maintained separately. This repository contains software,
experiments and independently verifiable evidence; no `paper/` directory is
required. The current research attribution and citation license is retained.
Previously distributed third-party and MIT-licensed copies retain their existing
rights as described in `LICENSE`.

## Reproduce the complete study

```bash
docker compose up --build --abort-on-container-exit --exit-code-from validation
```

The default `full` profile retains the original campaigns and adds runtime
startup, practical refresh intervals, repeated noise controls, fair classical
baselines, conditioning/resources, and independent full-state operator audits.
The final stage writes `artifacts/final-report.html`, `.md`, and `.json`, plus
CSV tables, SVG/PNG plots, raw results, per-stage logs, circuit exports, an
evidence manifest, checksums and `QCeNN_MI_Validation_Bundle.zip`.

Each run receives one `publication_run_id`. All table and plot inputs, producer
source, source commit/fingerprint, versions, image provenance when available,
configuration, seeds and per-file digests are tied to that run. The bundle is
created only after the software stages and evidence integrity check pass.
A passing suite can contain intentional stale-operator failures, failed noisy
quality gates and resource-limit rejections. Their verdicts remain separate.

For an inexpensive integration run:

```bash
QCENN_PROFILE=smoke QCENN_ARTIFACTS_DIR=./artifacts/smoke \
  docker compose up --build --abort-on-container-exit --exit-code-from validation
```

A fresh directory is required for each run. `configs/full.json` and
`configs/smoke.json` specify the additional experiments. The full profile runs
20 stages including report verification. The smoke profile is intentionally
smaller and is never evidence that the full study passed.

## Claim, implementation, experiment, and archive mapping

| Claim or question | Implementation / mathematical check | Experiment input | Generated table or figure |
|---|---|---|---|
| Coherent inverse action and two-sided error | `compact.py`, analytic Chebyshev tail; independently reconstructed native operator | `operator-audit/result.json`, `precision-1000/result.json` | `operator-correctness.csv`, `precision.csv` |
| Useful refresh interval and Eq. (24)/(25) | `temporal_refresh_benchmark.py`, explicit positive gap and Lipschitz bound | `temporal-refresh/result.json` | `refresh-accuracy.csv`, `refresh-intervals.csv`, refresh figure |
| Matrix acquisition error | `tracking_bound`; separate from gate noise | Refresh samples with nonzero acquisition error | Refresh tables and full raw JSON |
| Quantum gate/readout and verifier effects | `noise_verification_benchmark.py`, five controlled noise locations | `noise-verification/result.json` | `noise-controls.csv`, echo-control figure |
| Native dense gate costs | `hardware.py`, line routing and final permutation restoration | `hardware/result.json`, `hardware/audit/`, QPY | `native-resources.csv`, `native-noise.csv`; manuscript Tables III/IV require this run's values |
| Classical and emulator wall times | `baseline_benchmark.py`; SciPy/LU/QR/SVD with matched matrix/RHS | `baselines/result.json` | `solver-baselines.csv`, solver-cost figure |
| Conditioning, storage and postselection cost | `resource_benchmark.py`; explicit degree, width, memory and synthesis budgets | `resources/result.json` | `conditioning-resources.csv`, resource figure |
| Historical manuscript/archive discrepancy | Preserved separate raw counts, source revisions, logs and circuit files | `publication-evidence/historical-runs/`, `historical-reconciliation.json` | Explicit discrepancy record; no pooled statistics |

Strict full-state audits use optimization level 1 to retain tiny endpoint
rotations at repeated singular values. Level 2 comparisons are measured and
retained even when they fail that stricter full-state target. This does not
change the main hardware study's level-2 baseline defaults.

Full-state native audits include all unsuccessful ancilla sectors, not just the
reconstructed success amplitudes. Unseen right-hand sides use the same unfused
input-independent operator. Regression tests forbid classical solves, inverse
construction and residual optimization in the coherent runtime path. References
are permitted only in independent audits and final projection verifiers.

## Output schema and failure contract

`validation.json` schema version 2 records run ID, profile, stage commands,
return codes, logs, wall/CPU times, process peak RSS and provenance. Experiment
`result.json` files record explicit configuration and `passed` or
`native_validation_passed`; synthetic noisy quality has its own field. Absent
native synthesis timings/counts are `null` with a resource-limit explanation.
Missing physical acquisition latency is `null`, never substituted by model time.

`publication-evidence/manifest.json` schema version 1 maps each table and figure
to input JSON, its digest, producer code digest, columns/row count and output
file digest. The mutable aggregate status/log files are excluded from this
inner manifest and covered by the completed-run `SHA256SUMS.json`. A separate
`PACKAGE_SHA256` covers the final ZIP. `verify-only` checks raw file digests and
recomputes every CSV table from its source metrics, so updating a forged table's
checksum does not make it pass.

```bash
docker compose run --rm validation python -m qcenn_mi.evidence_report \
  --results /artifacts --verify-only
```

A failed scientific or software stage exits nonzero and retains its logs.
Synthetic quality failures are reported as scientific outcomes unless the
separate hardware CLI's `--require-noisy-quality` is explicitly selected.

## Release and archive policy

The first successful main-branch CI run creates the immutable
`v0.2.0-publication-candidate` tag and attaches its actual CI-generated bundle
and report. Candidate-tag runs also execute the full suite. Existing tags and
release assets are retained; future candidates require new tag names. The
workflow records image digests and semantically verifies report metrics before
creating the candidate. Tag/release names must
not be force-moved to conceal a changed result. The report retains local/CI/HPC
execution context; CI success is not an allocated EuroHPC or physical QPU run.
Historical QPY circuits are stored losslessly as `.qpy.gz`; `CIRCUITS.json`
records their raw and compressed hashes. Decompress before loading with Qiskit.
Their original source license is preserved alongside the historical evidence.

Citation metadata and package version identify the release candidate separately
from the historical manuscript benchmark.

## Statistical and comparison scope

The noise study uses fixed, configurable independent shot seeds, Wilson 95%
intervals, accepted/total/returned counts, and effective good-output probability.
Noiseless, readout-only, verifier-only, solver-only and solver-plus-verifier
controls identify what the observable includes. Multiple optimization levels
and fused/unfused input preparation are retained, including regressions.
These are synthetic density-matrix channels; calibration uncertainty is not
modeled. The historical N=100 64-shot result remains statistically limited.
Use larger budgets only when feasible and retain skipped/limited cases.

Classical baselines output a full classical vector. The coherent target is an
inverse-applied state, and its emulator exposes amplitudes for audit. Their
wall times are reported with task differences, never as quantum speedup ratios.
HHL and QSVT are analytical access-model comparisons only:
[HHL](https://doi.org/10.1103/PhysRevLett.103.150502),
[QSVT](https://arxiv.org/abs/1806.01838). Dense spectral encoding, preparation,
postselection, routing and readout remain costs of the implemented method.

Native timing is measured on a declared subset of the actual refresh trajectories
(default order <=10, condition <=10, unperturbed initial update). Other rows keep
null native timings with the budget reason. Encoding time includes its actual
classical decomposition and polynomial construction. Isolated SVD diagnostics
are separate reference measurements; SPD encoding may use eigendecomposition.
No timing is interpreted as physical quantum-device latency.

The source fingerprint excludes generated `*.egg-info` and bytecode caches so
an editable install and a normal wheel install identify the same source bytes.
Both original hardware JSON records and their raw-count audits are included in
`publication-evidence/historical-runs/`; the original `results/` files remain
unchanged in GitHub. This makes historical verification work after extracting
the portable source/evidence bundle without needing an external results folder.
