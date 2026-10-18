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

## Validation stages

| Stage | Coverage |
|---|---|
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
| `hardware` | Native ideal and synthetic-noise 2×2/4×4/10×10/100×100 checks |

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
on pushes and pull requests to `main`, and can also be started manually. It
checks Compose configuration, builds the locked image, runs all stages, records
image details and Compose logs, publishes the summary, and uploads evidence
with 30-day retention. Generated results are not committed automatically.
