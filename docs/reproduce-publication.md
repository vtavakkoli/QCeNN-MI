# Run every test and build the publication package

From a clean checkout, Docker Engine with Compose v2 runs the complete default
suite and builds both PDFs in one command:

```bash
docker compose up --build --abort-on-container-exit --exit-code-from validation
```

All outputs appear in `artifacts/`. The service includes the numerical libraries,
test tools, pinned Python dependency closure and LaTeX tooling. It runs no cloud
jobs or QPU submissions. The first image build needs package-download access.
The image includes all source needed at runtime; the source is not mounted from
the host. Only the artifact directory is mounted for output.

The default resource limits are four CPUs and 4 GiB memory. Numerical library
thread pools are restricted to one for repeatability. Adjust `QCENN_CPUS`,
`QCENN_MEMORY_LIMIT` and `QCENN_ARTIFACTS_DIR` where needed. Always select a new
output directory for a new run; a prior `validation.json` is an explicit error
instead of silently mixing stale results.

```bash
# Example of a separate run; provide the source commit for the manifest.
SOURCE_COMMIT=$(git rev-parse HEAD) \
QCENN_ARTIFACTS_DIR=./artifacts/run-02 \
docker compose up --build --abort-on-container-exit --exit-code-from validation
```

For an immutable base image, build with
`docker compose build --build-arg PYTHON_IMAGE=python@sha256:YOUR_VERIFIED_DIGEST`.
The default `python:3.12-slim-bookworm` tag may change its patch release. Record
`docker image inspect qcenn-mi-validation:local` with the evidence. Dependency
pins and a content fingerprint improve repeatability; they are not a claim of
bitwise-identical results across CPUs, BLAS builds or future base images.

## What the command runs

1. Compilation, Ruff and the complete unit-test suite with JUnit output.
2. A1 ideal and finite-shot 2x2 checks.
3. A2-A4 static, noisy tomography, polynomial and tracking campaigns.
4. Block-hybrid 10/100 scaling and dynamic 2x2 checks.
5. Small coherent Neumann-LCU inverse-action checks.
6. Coherent 10/100/1000 structured statevector study and manageable native checks.
7. Compact degree/qubit and accuracy study.
8. Full-operator precision audit for thirteen 1000x1000 matrices.
9. 162 degree/condition/family checks of the theoretical error and success bounds.
10. Native ideal and synthetic-noise 2/4/10/100 hardware-preparation campaign.
11. Evidence-derived IEEE manuscript and EuroHPC technical report PDFs.

Individual software failures are retained while other independent numerical
stages continue. The final exit code is nonzero if any software stage fails.
Publication is built only after numerical software validation passes. **Noisy
quality has its own decision:** its expected failures are visible in JSON and
the aggregate summary; software PASS never means hardware qualification.

The default A2 depth sweep is 1/2/4, not an unreported 150-layer experiment.
Longer optimization sweeps can be requested separately with
`python -m qcenn_mi.experiments --depths 1 2 4 8 16 32 64 150 --output-dir NEW_PATH`.
They may be substantially more expensive and must be labeled as additional runs.

## Artifacts and provenance

`validation.json` contains every command, return code, wall/CPU time, per-stage
peak child RSS, source commit, dirty-tree flag, source-tree SHA-256, package
versions and thread controls. Linux RSS is measured by a fresh wrapper process
for each stage. This is process accounting, not an HPC allocation charge.
`SHA256SUMS.json` records final output digests. Campaign subdirectories contain
result JSON, summaries, raw-count audits and QPY circuits. `publication/` contains
the PDFs and a portable LaTeX source tree with generated tables, vector figures
and the official IEEE class/style actually used.

The report never infers that a local or CI run was an allocated EuroHPC run.
On the hosting system, additionally record scheduler job IDs, allocated nodes,
charged node-hours, node type, image digest and calibration context. Follow its
supported container/runtime policy rather than assuming a Docker daemon exists.

## Without Docker and with applicant-specific metadata

```bash
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m qcenn_mi.validation --output-dir artifacts/local-run --publication \
  --project-metadata /absolute/path/to/private-submission-metadata.json
```

This path needs TeX Live with IEEEtran and latexmk. The private JSON uses the
fields documented in `paper/README.md`. It is optional; without it the public
report prints that applicant allocation values were not supplied. Keep it out of
the public repository. On Docker, mount it read-only and pass its container path:

```bash
docker compose run --rm \
  -v /absolute/path/to/private-submission-metadata.json:/submission.json:ro \
  validation python -m qcenn_mi.validation --output-dir /artifacts --publication \
  --project-metadata /submission.json
```

The manuscript follows generic IEEEtran journal formatting. A final venue may
require a different template or page count. Author affiliations and venue-specific
copyright/submission data must be supplied by the authors. The package is never
submitted automatically to a journal or to EuroHPC.

A successful publication run also creates `QCeNN_MI_Publication_Package.zip`
and `PACKAGE_SHA256`. The archive contains the source tree plus a separate
`evidence/` directory with the recorded run, leaving `artifacts/` available for a
fresh reproduction after extraction. The archive includes the portable LaTeX
source tree, measured data, raw-count/QPY evidence and resource logs.
