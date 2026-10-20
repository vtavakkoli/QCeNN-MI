# Validation and reproducibility

## Complete Docker run

From a clean checkout, Docker Engine and Compose v2 run the complete default
simulator suite:

```bash
docker compose up --build --abort-on-container-exit --exit-code-from validation
```

The image includes the source, pinned Python dependencies, and test tools.
Only the output directory is mounted from the host. The workflow submits no
cloud jobs or physical QPU jobs. The first image build needs package-download
access.

Git attributes require LF line endings for source text. During the image build,
`docker/normalize_source.py` also restores LF for declared source text copied
from existing Windows CRLF checkouts, before installation and fingerprinting.
This leaves binary QPY/gzip archives and recorded checksum values unchanged.
Historical evidence still passes strict byte-level SHA-256 verification; other
content changes are not repaired or accepted by newline normalization.

Defaults are four CPUs, 4 GiB memory, and one thread per numerical-library pool.
Adjust `QCENN_CPUS`, `QCENN_MEMORY_LIMIT`, and `QCENN_ARTIFACTS_DIR` as needed.
A prior `validation.json` causes an explicit error, preventing completed runs
from being silently overwritten or mixed.

For a separate run with recorded source provenance:

```bash
SOURCE_COMMIT=$(git rev-parse HEAD) \
QCENN_ARTIFACTS_DIR=./artifacts/run-02 \
docker compose up --build --abort-on-container-exit --exit-code-from validation
```

To pin the base image, build with
`docker compose build --build-arg PYTHON_IMAGE=python@sha256:YOUR_VERIFIED_DIGEST`.
The default `python:3.12-slim-bookworm` tag can change. GitHub Actions resolves
and records its image digest before building. Dependency pins and source hashes
improve repeatability; results need not be bitwise identical across CPUs or
BLAS implementations.

## Startup compatibility and profiles

The container entrypoint discovers the hashed `libgomp` file bundled in the
installed Aer wheel without importing NumPy, Qiskit or Aer. It restarts the
requested process with that exact library in `LD_PRELOAD`, preserving other
preloads. This reserves its static TLS during process startup. The approach is
architecture independent; it does not assume an x86 library path or increase
the container RAM limit. See the [glibc loader documentation](https://sourceware.org/glibc/manual/latest/html_node/Dynamic-Linking-Tunables.html)
and [OpenMP loading notes](https://lightgbm.readthedocs.io/en/stable/FAQ.html).

The default is `QCENN_PROFILE=full`. Set `QCENN_PROFILE=smoke` for the smaller
configuration. For Python selection use `--profile`, `--config`, `--stages`, and
`--list-stages` on `qcenn_mi.validation`. New experiment CLIs expose sizes,
conditions, ages, targets, seeds, shots, profiles and optimization settings;
configuration files are recorded in each output and never change old baseline
defaults silently. A full run has 21 stages, including its final report.

## Validation stages

| Stage | Coverage |
|---|---|
| `runtime-smoke` | Real Aer Bell-state and finite-shot startup check |
| `compile` | Compile all package and test modules |
| `lint` | Ruff checks |
| `unit-tests` | Complete unit suite with JUnit output |
| `assumption-a1` | Ideal and finite-shot 2×2 quantum-cell checks |
| `a2-a4` | Coupled cells, noisy tomography, polynomial correction, and tracking |
| `scaling` | Block-hybrid 10×10/100×100 and dynamic 2×2 checks |
| `coherent` | Small coherent Neumann-LCU inverse action |
| `coherent-large` | Structured 10×10/100×100/1000×1000 simulations and smaller native checks |
| `compact` | Compact Chebyshev-LCU accuracy and qubit resources |
| `precision-1000` | Full-operator audit for thirteen 1000×1000 matrices |
| `theory` | 162 degree, condition-number, and matrix-family bound checks |
| `time-varying` | 40 snapshots, 32 delayed-operator pairs, 16 acquisition-error cases, and 8 rank-loss rejections |
| `recurrent-quantum` | Complete quantum channel circuits, convergence, coherent/reset equivalence, state reuse, QPY exports, final counts and separate CX-only synthetic noise |
| `hardware` | Native ideal and synthetic-noise 2×2/4×4/10×10/100×100 checks |

Additional full-profile stages are `temporal-refresh`, `noise-verification`,
`baselines`, `resources`, `operator-audit`, and `final-report`. Their scope,
raw schemas, provenance and limits are described in
[publication evidence](publication-evidence.md).

Independent stages continue after a failure so their logs remain available.
The final exit code is nonzero if any software stage fails. Synthetic noisy
quality is a separate result: expected quality failures remain visible in JSON
and the aggregate summary even when software validation passes.

The default coupled-cell depth sweep is 1/2/4. Additional depths can be requested
with `python -m qcenn_mi.experiments --depths 1 2 4 8 16 32 64 150 --output-dir NEW_PATH`.
Long sweeps may be substantially more expensive and must be recorded separately.

The temporal method recompiles each snapshot. Agreement with a drift or
acquisition-error bound does not imply a small residual. Acquisition error is
not quantum gate noise, and the suite does not test a real-time deadline.

## Outputs and provenance

| Output | Contents |
|---|---|
| `validation.json` | Stage commands, return codes, wall/CPU times, peak child RSS, and source/environment provenance |
| `summary.md` | Aggregate software status and separate noisy-quality outcome |
| `final-report.html`, `.md`, `.json` | Offline report with checklist status, measured tables/plots and limits |
| `publication-evidence/manifest.json` | Input, producer and per-table/figure SHA-256 provenance |
| `unit-tests.xml` | JUnit test results |
| `logs/` | Per-stage stdout/stderr and resource measurements |
| Campaign directories | JSON results, Markdown summaries, raw-count audits, and QPY circuit exports |
| `SHA256SUMS.json` | SHA-256 digests of completed run outputs |
| `QCeNN_MI_Validation_Bundle.zip` | Source and run evidence, created after all software stages pass |
| `PACKAGE_SHA256` | Checksum of the validation bundle |

The bundle places source under `QCeNN-MI/` and measured outputs under
`QCeNN-MI/evidence/`. Extract it and run Docker Compose from the source directory
to reproduce into a fresh `artifacts/` directory.

Source provenance includes commit, dirty-tree status, source fingerprint,
Python/package versions, thread settings, and execution context. Peak child RSS
uses a fresh wrapper process per stage. These are process measurements, not HPC
allocation charges. On an allocated system, separately retain scheduler job IDs,
node types, charged node-hours, image digests, and calibration context. Follow
the host's supported container policy.

## Local Python run

On Linux or macOS, use the locked environment and run from the repository root:

```bash
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m qcenn_mi.validation --output-dir artifacts/local-run
```

The aggregate runner uses Unix process-resource accounting. Docker provides the
supported path on Windows; individual simulator commands remain available
through the Python package.

## Continuous integration

[Quantum simulator validation](../.github/workflows/quantum-feasibility.yml) runs
with AMD64/ARM64 startup checks, a smoke profile on pull requests, and full
validation on main pushes and candidate tags. A Windows job checks Git CRLF
checkout behavior and the same source canonicalization used in the image.
It can also be started manually. It checks Compose configuration, builds the
locked image, runs all stages, records
image details and Compose logs, publishes the summary, and uploads evidence
with 30-day retention. Generated results are not committed automatically.
