# QCeNN-MI

[![Quantum feasibility proof](https://github.com/vtavakkoli/QCeNN-MI/actions/workflows/quantum-feasibility.yml/badge.svg)](https://github.com/vtavakkoli/QCeNN-MI/actions/workflows/quantum-feasibility.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

**QCeNN-MI** is a research prototype for **Quantum Cellular Nonlinear Networks for Matrix Inversion**. The project investigates whether matrix-inversion dynamics can be mapped to small, hardware-compatible quantum cell states and then extended toward locally coupled QCeNN dynamics for static and time-varying matrices.

> **Research status:** early feasibility prototype. The implementation covers A1–A4 as a small hybrid simulator prototype on Qiskit. It does **not** claim quantum speed-up or a scalable quantum matrix-inversion algorithm.

## Scientific motivation

The project builds on the dynamical matrix-inversion formulation developed in:

> V. Tavakkoli, J. C. Chedjou, and K. Kyamakya, **“A Novel Recurrent Neural Network-Based Ultra-Fast, Robust, and Scalable Solver for Inverting a ‘Time-Varying Matrix’,”** *Sensors*, 19(18), 4002, 2019.  
> DOI: [10.3390/s19184002](https://doi.org/10.3390/s19184002)

That work formulates time-varying matrix inversion as a dynamical residual-reduction problem. QCeNN-MI starts from the same objective,

$$ MX = I,\qquad E = MX - I. $$

and investigates quantum representations and local quantum interactions that can participate in driving the residual toward zero.

## First implemented assumption: A1

The first milestone is intentionally small, falsifiable, and executable on present gate-based quantum systems.

**Assumption A1.** For a real nonsingular $2\times2$ matrix $M$, each column of $M^{-1}$ can be represented by a normalized one-qubit real-amplitude state, up to a classical scalar, and recovered by minimizing the matrix residual.

For column $j$, the target is $b=e_j$. A single quantum cell prepares

$$ |q(\theta)\rangle = R_y(\theta)|0\rangle = \cos\!\left(\frac{\theta}{2}\right)|0\rangle + \sin\!\left(\frac{\theta}{2}\right)|1\rangle. $$

A scalar $s$ reconstructs the magnitude:

$$ x_j = s\,q(\theta). $$

For a fixed quantum direction, the least-squares optimal scale is

$$ s^*(\theta)=\frac{\bigl(Mq(\theta)\bigr)^T b}{\bigl(Mq(\theta)\bigr)^T\bigl(Mq(\theta)\bigr)}. $$

The one-dimensional quantum-state parameter is then selected by minimizing

$$ \mathcal{L}_j(\theta)=\left\|M\!\left(s^*(\theta)q(\theta)\right)-e_j\right\|_2^2. $$

Solving both basis-vector targets gives the two columns of the inverse.

### Why this is a useful first quantum test

A1 isolates the most basic representation question before introducing multi-qubit coupling. The implementation:

1. creates a real Qiskit `RY(theta)` quantum circuit;
2. uses Qiskit's statevector machinery during the deterministic feasibility solve;
3. verifies the final states with finite-shot **X** and **Z** measurements on `AerSimulator`;
4. reconstructs the measured state direction without direct statevector access;
5. computes the physically relevant residual $\|MX-I\|_F$;
6. fails CI if the predefined feasibility thresholds are not met.

The X/Z readout pattern is deliberately hardware-compatible and is the bridge to a future real-QPU backend.

## Reference benchmark

The CI benchmark uses

$$ M=\begin{bmatrix}2 & 0.5 \\ 0.5 & 1.5\end{bmatrix}. $$

Its classical reference inverse is

$$ M^{-1}=\begin{bmatrix}0.5454545 & -0.1818182 \\ -0.1818182 & 0.7272727\end{bmatrix}. $$

The workflow requires:

- ideal Qiskit-statevector residual $\|MX-I\|_F < 10^{-7}$;
- finite-shot residual $\|MX-I\|_F < 8\times10^{-2}$ with 8192 shots.

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
| **A1** | One-qubit inverse-column representation for real $2\times2$ matrices | Implemented |
| **A2** | Multi-cell local quantum coupling and shared QCeNN interaction templates | Implemented: simulator prototype |
| **A3** | Stateful tracking of time-varying $M(t)^{-1}$ | Implemented: hybrid warm-start tracker |
| **A4** | Polynomial correction operators inspired by the prior recurrent solver | Implemented: hybrid correction + dense quantum dilation |
| **A5** | Hardware-aware execution on a real QPU with calibration/noise analysis | Planned |
| **A6** | Structured/sparse matrix scaling study and comparison with classical baselines | Planned |

A4 includes a classical safeguarded correction and a small dense quantum dilation with ancilla postselection. Efficient quantum polynomial synthesis and scalable block encoding remain future research.

## Scope and scientific claims

The current prototype demonstrates **representational and execution feasibility** for a small problem. It does not establish:

- asymptotic quantum speed-up;
- superiority over classical matrix inversion;
- efficient dense-matrix data loading;
- scalable extraction of a complete classical inverse from an amplitude-encoded state;
- a full physical QCeNN dynamical system beyond the local gate ansatz.

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

## A2–A4: simulator implementation and hardware handoff

```bash
python -m qcenn_mi.experiments --output-dir results/a2-a4
# Optional longer depth sweep; optimization cost grows with depth:
python -m qcenn_mi.experiments --depths 1 2 4 8 16 32 64 150 --output-dir results/depth-sweep
```

The default campaign checks SPD, signed and nonsymmetric matrices (2×2 and
4×4), depths 1/2/4, ideal statevector convergence, finite-shot Pauli tomography,
synthetic depolarizing noise, polynomial correction and time-varying tracking.
It exits nonzero on failed acceptance gates and writes JSON, Markdown and QPY
circuits. A4 also checks a dense polynomial dilation with ideal and finite-shot ancilla
postselection. Noise is synthetic, not calibration-derived. The 150-layer sweep is
available but is not implied to have passed by the default campaign.

- **A2:** inverse columns use one or two qubit cells with local RY layers and
  nearest-neighbor controlled-RY interactions. Interaction angles are optimized
  jointly and shared across columns; local rotations are column-specific.
  Full X/Y/Z Pauli tomography recovers signed entangled directions. Classical
  least-squares scalars recover magnitudes. Measurement costs 3^q settings per
  column, so this readout is intentionally limited to small experiments.
- **A3:** `InverseTracker.update(matrix, time=t)` preserves quantum circuit
  parameters, warm starts each variational solve and tracks a classical inverse
  estimate with the first-order predictor `X - X @ (M_new-M_old) @ X`.
  Times must increase, dimensions stay fixed, and failed solves do not commit
  state. Quantum, predictor and classically corrected residuals are separate.
- **A4:** `polynomial_correct(M, X, order=p)` computes
  `X @ sum((I-MX)^k, k=0..p)`. Before safeguarding, its residual is
  `(I-MX)^(p+1)`. Backtracking accepts only finite, non-increasing residuals.
  This is inspired by recurrent residual reduction, not an implementation of
  the exact 2019 solver. `quantum_polynomial_correct` also builds a dense unitary
  dilation of the left correction polynomial `sum((I-XM)^k)`, applies it to
  amplitude-encoded columns and postselects an ancilla. Ideal statevector output
  is compared with the classical polynomial; shot-based readout uses actual
  ancilla=0 counts and classical least-squares scales. The polynomial and dense
  dilation are constructed classically, so this does not claim efficient quantum
  polynomial synthesis or scalable data access.

See [the simulator and hardware protocol](docs/simulator-to-hardware.md).
The physical experiment and calibration analysis (A5) remain pending.

## 10×10 / 100×100 and dynamic 2×2 checks

```bash
python -m qcenn_mi.scaling --output-dir results/scaling
```

The larger cases use a **block-hybrid** solver: actual Qiskit variational 2×2
quantum solves seed diagonal blocks, followed by classical dense polynomial
corrections using the entire matrix, including off-block entries. This is not
an extension of the amplitude-encoded ansatz to 100 dimensions. The campaign
checks symmetric and nonsymmetric diagonally dominant 10×10 and 100×100
matrices with both ideal and finite-shot quantum seeds, reporting full-matrix
residuals, relative errors and correction history.
No classical inverse is used by the solver; NumPy inversion is reporting-only.
Singular diagonal blocks and nonconvergence are explicit failures. These fixtures
do not establish convergence for arbitrary matrices or quantum speed-up.

A dynamic 2×2 experiment varies all four entries over 21 time points in a full
period, warm starts quantum circuit parameters, and checks ideal, corrected and
finite-shot residuals (4096 shots per measurement setting). Automated tests also
check finite-shot seeds for the block-hybrid solver.

## Standalone coherent inverse action

```bash
python -m qcenn_mi.coherent_benchmark --output-dir results/coherent
```

This additional baseline performs inverse action inside a quantum circuit,
without a variational optimizer, a classically computed inverse, solution-state
fitting, least-squares scale refitting, or classical residual correction.
For real symmetric 2×2/4×4 matrices certified positive definite by Gershgorin
bounds, it implements a truncated Neumann series using coherent selection of
repeated block-encoded factors. It accepts arbitrary real nonzero input vectors.
The same inverse-action operator can be reused for different inputs.

The campaign checks degrees 1/3/7, every inverse column, a superposition input,
ideal residual bounds, finite-shot tomography, measured postselection probability,
and native-basis resource counts. Default readout uses 131072 shots per setting.
The resulting classical inverse is validation readout; the circuit output itself
is an approximate inverse-applied quantum state, conditioned on ancilla success.

This remains a small simulator prototype. Classical dense square-root dilation
constructs the input block encoding; it is not an efficient scalable input oracle.
The method uses extra ancillas, has low postselection probability and substantial
native gate counts. It is a coherent linear-algebra baseline, not local QCeNN
dynamics or evidence of quantum advantage. Hardware and noisy-native-circuit
validation remain pending. See [the coherent protocol](docs/coherent-inverse.md).
