# QCeNN-MI

[![Quantum feasibility proof](https://github.com/vtavakkoli/QCeNN-MI/actions/workflows/quantum-feasibility.yml/badge.svg)](https://github.com/vtavakkoli/QCeNN-MI/actions/workflows/quantum-feasibility.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

**QCeNN-MI** is a research prototype for **Quantum Cellular Nonlinear Networks for Matrix Inversion**. The project investigates whether matrix-inversion dynamics can be mapped to small, hardware-compatible quantum cell states and then extended toward locally coupled QCeNN dynamics for static and time-varying matrices.

> **Research status:** early feasibility prototype. The current implementation validates the smallest quantum representation assumption on Qiskit. It does **not** claim quantum speed-up or a complete QCeNN implementation.

## Scientific motivation

The project builds on the dynamical matrix-inversion formulation developed in:

> V. Tavakkoli, J. C. Chedjou, and K. Kyamakya, **“A Novel Recurrent Neural Network-Based Ultra-Fast, Robust, and Scalable Solver for Inverting a ‘Time-Varying Matrix’,”** *Sensors*, 19(18), 4002, 2019.  
> DOI: [10.3390/s19184002](https://doi.org/10.3390/s19184002)

That work formulates time-varying matrix inversion as a dynamical residual-reduction problem. QCeNN-MI starts from the same objective,

\[
M X = I,
\qquad
E = M X - I,
\]

and investigates quantum representations and local quantum interactions that can participate in driving the residual toward zero.

## First implemented assumption: A1

The first milestone is intentionally small, falsifiable, and executable on present gate-based quantum systems.

**Assumption A1.** For a real nonsingular \(2\times2\) matrix \(M\), each column of \(M^{-1}\) can be represented by a normalized one-qubit real-amplitude state, up to a classical scalar, and recovered by minimizing the matrix residual.

For column \(j\), the target is \(b=e_j\). A single quantum cell prepares

\[
|q(\theta)\rangle = R_y(\theta)|0\rangle
= \cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle.
\]

A scalar \(s\) reconstructs the magnitude:

\[
x_j = s\,q(\theta).
\]

For a fixed quantum direction, the least-squares optimal scale is

\[
s^*(\theta)=
\frac{(M q(\theta))^T b}
     {(M q(\theta))^T(M q(\theta))}.
\]

The one-dimensional quantum-state parameter is then selected by minimizing

\[
\mathcal{L}_j(\theta)=
\left\|M\big(s^*(\theta)q(\theta)\big)-e_j\right\|_2^2.
\]

Solving both basis-vector targets gives the two columns of the inverse.

### Why this is a useful first quantum test

A1 isolates the most basic representation question before introducing multi-qubit coupling. The implementation:

1. creates a real Qiskit `RY(theta)` quantum circuit;
2. uses Qiskit's statevector machinery during the deterministic feasibility solve;
3. verifies the final states with finite-shot **X** and **Z** measurements on `AerSimulator`;
4. reconstructs the measured state direction without direct statevector access;
5. computes the physically relevant residual \(\|MX-I\|_F\);
6. fails CI if the predefined feasibility thresholds are not met.

The X/Z readout pattern is deliberately hardware-compatible and is the bridge to a future real-QPU backend.

## Reference benchmark

The CI benchmark uses

\[
M=
\begin{bmatrix}
2 & 0.5\\
0.5 & 1.5
\end{bmatrix},
\]

whose classical reference inverse is

\[
M^{-1}=\begin{bmatrix}
0.5454545 & -0.1818182\\
-0.1818182 & 0.7272727
\end{bmatrix}.
\]

The workflow requires:

- ideal Qiskit-statevector residual \(\|MX-I\|_F < 10^{-7}\);
- finite-shot residual \(\|MX-I\|_F < 8\times10^{-2}\) with 8192 shots.

These thresholds validate feasibility only; they are not evidence of quantum advantage.

## Quick start

```bash
git clone https://github.com/vtavakkoli/QCeNN-MI.git
cd QCeNN-MI

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Run the automated tests:

```bash
pytest -q
```

Run the A1 quantum benchmark:

```bash
qcenn-mi \
  --matrix 2.0 0.5 0.5 1.5 \
  --shots 8192 \
  --output-dir results/assumption-a1
```

The command writes:

```text
results/assumption-a1/
├── result.json
└── summary.md
```

## Continuous quantum feasibility test

`.github/workflows/quantum-feasibility.yml` runs on every push and pull request. It performs:

- package installation;
- Python compilation and Ruff checks;
- unit tests;
- Qiskit statevector matrix-inversion test;
- finite-shot Aer verification;
- acceptance-threshold validation;
- publication of the Markdown result in the GitHub Actions job summary;
- upload of `result.json` and `summary.md` as a workflow artifact.

This makes the first scientific assumption continuously reproducible rather than leaving it as an unchecked notebook experiment.

## Repository layout

```text
QCeNN-MI/
├── .github/workflows/
│   └── quantum-feasibility.yml
├── src/qcenn_mi/
│   ├── __init__.py
│   ├── cli.py
│   ├── quantum.py
│   └── solver.py
├── tests/
│   └── test_assumption_a1.py
├── results/
│   └── README.md
├── CITATION.cff
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Research roadmap

| Milestone | Goal | Status |
|---|---|---|
| **A1** | One-qubit inverse-column representation for real 2x2 matrices | Implemented |
| **A2** | Multi-cell local quantum coupling and shared QCeNN interaction templates | Planned |
| **A3** | Stateful tracking of time-varying \(M(t)^{-1}\) | Planned |
| **A4** | Polynomial correction operators inspired by the prior recurrent solver | Planned |
| **A5** | Hardware-aware execution on a real QPU with calibration/noise analysis | Planned |
| **A6** | Structured/sparse matrix scaling study and comparison with classical baselines | Planned |

A later research branch will investigate whether the polynomial term used in the earlier dynamical formulation can be mapped efficiently to quantum polynomial/block-encoding techniques. That is intentionally outside A1.

## Scope and scientific claims

The current prototype demonstrates **representational and execution feasibility** for a small problem. It does not establish:

- asymptotic quantum speed-up;
- superiority over classical matrix inversion;
- efficient dense-matrix data loading;
- scalable extraction of a complete classical inverse from an amplitude-encoded state;
- full QCeNN local dynamics.

Those questions are explicit targets for later milestones and will be evaluated experimentally rather than assumed.

## Citation

If you use the matrix-inversion formulation that motivates this project, please cite:

```bibtex
@article{Tavakkoli2019TimeVaryingMatrix,
  author  = {Tavakkoli, Vahid and Chedjou, Jean Chamberlain and Kyamakya, Kyandoghere},
  title   = {A Novel Recurrent Neural Network-Based Ultra-Fast, Robust, and Scalable Solver for Inverting a {Time-Varying Matrix}},
  journal = {Sensors},
  volume  = {19},
  number  = {18},
  pages   = {4002},
  year    = {2019},
  doi     = {10.3390/s19184002}
}
```

See [`CITATION.cff`](CITATION.cff) for software citation metadata.

## License

QCeNN-MI is released under the [MIT License](LICENSE).
