# Larger standalone coherent inverse-action validation

## Verification levels

The requested sizes are 10, 100 and 1000. The corresponding padded quantum
registers have 16, 128 and 1024 basis states. Degree-7 circuits use 14, 17 and
20 qubits, including every selector and flag ancilla.

The study tests four explicitly named input families (dense SPD, tridiagonal
SPD, dense nonsymmetric and signed indefinite real matrices), three right-hand
sides per case (basis, uniform and signed random), and
1/3/7-degree convergence. This does not test all possible matrices. A poorly
conditioned SPD counterexample remains inaccurate at degree 7; unsupported
matrix classes are rejected rather than silently routed to a classical solver.

- **10×10:** exact structured statevector at degree 7; independent native-gate
  Aer check at degree 3; unit test compares the complete state against Qiskit.
- **100×100:** exact structured statevector at degree 7; independent native-gate
  Aer check at degree 1.
- **1000×1000:** exact structured statevector at degree 7; native-gate Aer, noise
  and hardware execution have not been validated at this size.

## Exact structured simulation

Input encoding computes `M = V diag(lambda) V.T` for SPD inputs, or
`M = U diag(sigma) V.T` for general real nonsingular inputs, classically.
This is dense O(N³) preprocessing and is a major limitation, not an efficiency
improvement.
No reciprocal eigenvalues, classical inverse, solution fit or residual correction
are used by the circuit builder or simulator. Classical solves are used only
after simulation to report independent errors.

Set `tau = 2/(s_min+s_max)`, where s denotes the positive eigenvalues for SPD
inputs or singular values otherwise. In the encoding basis, the contraction
has entries `b_j = 1-tau*s_j`. The input block encoding acts on each flag
and data eigenindex as

`[[b_j, sqrt(1-b_j²)], [sqrt(1-b_j²), -b_j]]`.

A Qiskit circuit implements this with Z and uniformly controlled RY gates. A
uniform selector superposes orders 0..degree. Each branch applies a new flag
for every factor; the selector is uncomputed afterward. Every flag and selector
amplitude is retained in the simulation tensor. There is no collapse, reset,
classical feedback or direct polynomial evaluation in the simulator loop.

The structured simulator applies these diagonal-controlled quantum gates directly
to amplitudes instead of allocating gigantic multi-control dense matrices.
The final basis transform V is evaluated on the successful ancilla sector;
other sectors retain a lazy unitary basis transform. Their norms and success
probability are unchanged. A full-state comparison at small size validates this
representation against the exported Qiskit circuit.

For general real inputs, preparation uses U.T and final data conversion uses V;
thus the same polynomial on positive singular values gives inverse action on M.
The output remains the coherent degree-limited approximation to `M^-1 |b>`.
Its unnormalized magnitude follows from known normalization and success amplitude.
This is distinct from the previous hybrid block-initialization/classical
correction path in `scaling.py`.

## What the reported numbers mean

Observed right-hand-side residuals are `||M x-b||/||b||`, calculated from simulated
success amplitudes. Relative solution errors compare them with a classical
reference computed afterward. State norm and padding leakage are checked.

For contraction norm `q=max(abs(1-tau*s))`, the all-input relative residual
bound is `q^(degree+1)`. The full-inverse Frobenius residual is bounded by
`sqrt(N)*q^(degree+1)`. These are **analytical bounds**, not claims that all inverse
columns have been individually simulated or measured.

`sample_z` draws from the exact circuit's joint ancilla-success/data-Z marginal.
It is explicitly labeled exact-distribution sampling rather than Aer execution.
Z probabilities do not reveal amplitude signs and do not reconstruct the full
classical inverse. Reported conditional total variation can be large when a
small accepted-shot budget is spread across 1000 outputs; it is informational.
Full tomography and 1000×1000 signed matrix readout are not performed.

## Resource limits and reproducibility

A preflight estimates state, update and dense-encoding memory before expensive
allocation. The default budget is 512 MiB. At degree 7, the 1000-dimensional
case has a 16 MiB statevector and a conservative 128 MiB working estimate. This
is an estimate, not an operating-system memory cap. Eigensolver and compilation
time still grow with input size.

`LargeCoherentInverse.circuit(rhs)` exports structured Qiskit circuits with lazy
controls. Export above size 128 requires explicit `allow_large=True`; native
dense unitary synthesis at 1000 dimensions can be extremely costly. The study
reports native gate counts only for the smaller independent Aer checks, without
a device coupling map. It does not extrapolate those counts as hardware evidence.

Run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m qcenn_mi.coherent_large_benchmark
```

JSON includes versions, seeds, matrix-generation families, encoding/simulation
times, resource estimates, actual input residuals, independent reference errors,
Z-sampling statistics, native crosschecks and limitations. Detailed sample-count
audits are regenerated in `audit/` and uploaded by CI with the report.
This implementation supports finite real numerically nonsingular inputs;
complex and singular inputs are rejected.
Quantum advantage, arbitrary-matrix support and physical performance remain open.
