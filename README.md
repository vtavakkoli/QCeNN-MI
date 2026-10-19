# QCeNN-MI

**Quantum matrix inverse action, Chebyshev-LCU circuits, and QCeNN simulator benchmarks.**

[![Quantum simulator validation](https://github.com/vtavakkoli/QCeNN-MI/actions/workflows/quantum-feasibility.yml/badge.svg)](https://github.com/vtavakkoli/QCeNN-MI/actions/workflows/quantum-feasibility.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![License: Research Attribution and Citation 1.0](https://img.shields.io/badge/License-Research%20Citation%201.0-orange.svg)](LICENSE)

QCeNN-MI is a research toolkit for static and time-varying matrix inversion.
It provides coherent quantum inverse-action circuits, hybrid quantum cellular
nonlinear network experiments, and reproducible accuracy and resource audits.
The coherent circuit produces a state proportional to `M⁻¹b`, conditioned on
ancilla success; a complete classical inverse requires additional readout.

**License and citation:** This project uses the [QCeNN-MI Research Attribution and Citation License 1.0](LICENSE). **Attribution is required before use**, including internal research use; public software or products must credit the project, and scholarly/public reports of results must cite it. See [Citation and license](#citation-and-license) and [CITATION.cff](CITATION.cff).

**Status:** validated on simulators for the documented test families. Physical
quantum-device execution and end-to-end quantum advantage remain unverified.

## Capabilities

| Component | Implemented scope | Execution model |
|---|---|---|
| Local QCeNN cells and shared interaction templates | Real 2×2 and 4×4 matrices | Variational quantum circuits with classical optimization and readout scaling |
| Stateful tracking and polynomial correction | Changing small matrices; block-hybrid 10×10 and 100×100 cases | Quantum circuit parameters plus classical prediction and correction |
| Coherent inverse action | Neumann-LCU and compact Chebyshev-LCU constructions | Postselected quantum operation with classical dense spectral input encoding |
| Large ideal-simulation audits | Matrix orders 10, 100, and 1000; full-operator precision checks at order 1000 | Structured statevector simulation with smaller native-circuit cross-checks |
| Temporal robustness audits | Orders 2, 10, 100, and 1000 | Refreshed snapshots, delayed updates, acquisition error, and rank-loss rejection |
| Hardware preparation | Orders 2, 4, 10, and 100 | Native-gate compilation, QPY exports, and synthetic-noise simulation |

## Run the complete validation suite

Docker Engine and Compose v2 run all unit tests and default simulator campaigns:

```bash
git clone https://github.com/vtavakkoli/QCeNN-MI.git
cd QCeNN-MI
docker compose up --build --abort-on-container-exit --exit-code-from validation
```

The 20 full-profile stages cover compilation, lint, unit tests, hybrid solvers, coherent
circuits, scaling, precision, theoretical bounds, temporal dynamics, and hardware
simulation. Outputs include JSON results, Markdown summaries, raw measurement
audits, QPY circuits, timing, provenance, and a validation bundle in `artifacts/`. The final stage generates an offline
HTML report, Markdown/JSON reports, CSV tables, SVG/PNG figures, and a verified
evidence manifest.
Each run must use a fresh output directory.

See [validation and reproducibility](docs/validation.md) for resource limits,
individual stages, evidence formats, and repeat runs.

## Test profiles and final report

```bash
# Faster integration profile; each run uses a fresh output folder
QCENN_PROFILE=smoke QCENN_ARTIFACTS_DIR=./artifacts/smoke \
  docker compose up --build --abort-on-container-exit --exit-code-from validation

# Independently check table metrics and evidence digests after a run
docker compose run --rm validation python -m qcenn_mi.evidence_report \
  --results /artifacts --verify-only
```

Open `artifacts/final-report.html` for checklist coverage, measured results,
plots, explicit limitations, and complete CSV downloads. The report separates
software validation, ideal numerical error, synthetic noisy quality, and
physical-device validation. See [publication evidence](docs/publication-evidence.md)
for experiment settings, claim mappings, historical discrepancies and releases.

The image preloads Aer’s actual wheel OpenMP library before starting Python,
which avoids the Linux `cannot allocate memory in static TLS block` startup
failure. Startup and real simulator execution are checked on AMD64 and ARM64 CI.

## Python installation

Use Python 3.11 or later; the locked validation environment targets Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

For development with supported dependency ranges, use
`python -m pip install -e ".[dev]"` instead of the locked installation.

Run a basic 2×2 quantum-cell example:

```bash
qcenn-mi --matrix 2.0 0.5 0.5 1.5 --shots 8192 --output-dir results/assumption-a1
```

Run individual campaigns:

```bash
# Compact coherent inverse action and circuit resources
python -m qcenn_mi.compact_benchmark --output-dir results/compact

# Complete inverse-action operator audit for 1000×1000 fixtures
python -m qcenn_mi.precision_benchmark --output-dir results/precision-1000

# Time-varying matrices, delayed updates, and acquisition error
python -m qcenn_mi.time_varying_benchmark --output-dir results/time-varying

# Native-gate and synthetic-noise checks; no physical QPU submission
python -m qcenn_mi.hardware --qubit-budget 42 --output-dir results/hardware

# New practical tracking, repeated noise controls, baselines and resource studies
python -m qcenn_mi.temporal_refresh_benchmark --config configs/full.json
python -m qcenn_mi.noise_verification_benchmark --config configs/full.json
python -m qcenn_mi.baseline_benchmark --config configs/full.json
python -m qcenn_mi.resource_benchmark --config configs/full.json
```

For repeatable timings, set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`MKL_NUM_THREADS=1`. The complete validation runner applies these settings to
each stage automatically.

## Interpreting the results

The order-1000 precision study audits 13 real matrices, including SPD,
nonsymmetric, and indefinite inputs with controlled condition numbers up to
1000. Its polynomial target is `1e-12`, and the separate numerical acceptance
limit is `1e-10`. The archived worst operator residual is approximately
`7.3e-13`; see the [precision protocol and recorded results](docs/precision-1000.md).
These are ideal-simulation results for specified fixtures.

Amplitude encoding uses `ceil(log2(N))` data qubits, plus walk and selector
ancillas. The recorded order-1000 precision cases use 15–21 total qubits.
Dense classical eigendecomposition/SVD, circuit depth, postselection, and
measurement costs remain substantial. Worst-input postselection success falls
to approximately `1e-6` at condition number 1000.

Software validation and noisy-circuit quality are reported separately. The
existing 10×10 optimistic synthetic-noise case passes its quality gate, while
the 10×10 stress case and dense 100×100 optimistic case fail. A passing software
suite therefore does not establish hardware accuracy. Use
`--require-noisy-quality` with the hardware campaign to enforce its noisy gates.

The temporal method recompiles matrix snapshots. Its bounds require a positive
singular-value gap; acquisition error is distinct from quantum gate noise.
Finite-dimensional theory does not remove numerical, resource, or update-latency
limits. Rectangular, complex, singular, and numerically singular inputs are
outside the current coherent implementation's supported domain.

## Documentation

- [Validation, Docker, CI, and evidence formats](docs/validation.md)
- [Hybrid QCeNN experiments and physical-experiment protocol](docs/simulator-to-hardware.md)
- [Small coherent inverse-action construction](docs/coherent-inverse.md)
- [Large structured-statevector simulations](docs/coherent-large.md)
- [Compact circuits and hardware preparation](docs/compact-hardware.md)
- [1000×1000 precision and conditioning](docs/precision-1000.md)
- [Archived simulator results](https://github.com/vtavakkoli/QCeNN-MI/tree/main/results)

## Development

```bash
python -m ruff check src tests docker
python -m pytest -q
```

GitHub Actions runs AMD64/ARM64 startup checks, a smoke suite on pull requests,
and the complete Docker Compose suite on pushes to `main` and candidate tags. Each run uploads its evidence and validation bundle as a
workflow artifact and publishes the aggregate summary on the workflow run.

| Path | Purpose |
|---|---|
| `src/qcenn_mi/` | Solvers, quantum circuits, simulation campaigns, and validation runner |
| `tests/` | Numerical, circuit, tracking, and validation regression tests |
| `docs/` | Reproduction instructions and experiment protocols |
| `results/` | Selected archived simulator evidence |
| `.github/workflows/` | Continuous validation |

## Citation and license

**Citation and attribution are mandatory conditions of reuse under the
[QCeNN-MI Research Attribution and Citation License 1.0](LICENSE).**

Before first use in a project, add the following credit to its README,
NOTICE, or research documentation (for unmodified standalone evaluation,
keep the distributed `LICENSE` and `CITATION.cff` files):

> QCeNN-MI — Vahid Tavakkoli and Kabeh Mohsenzadegan (2026).
> https://github.com/vtavakkoli/QCeNN-MI

For scholarly papers, preprints, theses, presentations, technical reports,
public benchmarks, or other public research results using this project,
**cite the software** in the references or acknowledgments:

```bibtex
@software{tavakkoli_mohsenzadegan_qcenn_mi_2026,
  author  = {Vahid Tavakkoli and Kabeh Mohsenzadegan},
  title   = {QCeNN-MI: Quantum Cellular Nonlinear Network for Matrix Inversion},
  year    = {2026},
  version = {0.1.1},
  url     = {https://github.com/vtavakkoli/QCeNN-MI}
}
```

The GitHub **Cite this repository** feature draws its citation information
from [CITATION.cff](CITATION.cff). The hybrid dynamics are motivated by
Tavakkoli, Chedjou, and Kyamakya's
[2019 time-varying matrix-inversion solver](https://doi.org/10.3390/s19184002);
also cite that paper if its scientific contributions are used.

**This is the sole license for the current QCeNN-MI release.** It is a custom
source-available license, **not an OSI-approved open-source license**.
Academic and commercial reuse are permitted only subject to the
[attribution and citation requirements](LICENSE). No alternative license is
offered for the current release. Previously distributed copies remain subject
to the terms under which they were originally licensed.
