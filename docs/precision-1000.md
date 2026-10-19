# 1000×1000 precision and conditioning

The compact coherent inverse action now supports a tighter polynomial target,
higher degrees and an audit of the complete simulated success operator.
Run the reproducible experiment with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m qcenn_mi.precision_benchmark --output-dir results/precision-1000
```

The campaign covers 13 real 1000×1000 matrices: the four earlier dense SPD,
tridiagonal SPD, nonsymmetric and indefinite fixtures, plus nine dense matrices
with prescribed singular-value condition numbers 10, 100 and 1000. Each
controlled group includes SPD, nonsymmetric and symmetric indefinite examples.
Fixed-seed orthogonal matrices mix the prescribed spectra into dense inputs.

## What changed

The encoder retains its original singular values. Previously the compact path
recovered them from `1 - tau*sigma`, which can lose precision through cancellation.
Walk angles now use
`2*atan2(sqrt(sigma_max-sigma), sqrt(sigma-sigma_min))`, avoiding subtraction of
the spectral center near the endpoints. Degree selection uses logarithms without
forming a potentially underflowing ratio of tolerances.

The normal simulator stores the selector preparation vector instead of its dense
unitary. It still evolves every selector, walk and data amplitude through the
controlled rotation gates. The successful sector of selector uncomputation is
the inner product with the preparation vector. Evaluating that projection is
exactly equivalent to applying PREPARE-dagger and selecting selector zero. Other
sectors retain a lazy selector basis change; their total norm is preserved.
This avoids a quadratic selector matrix and dense multiplication on every input.
It does not replace the circuit with reciprocal values or a classical solver.

`return_state=True` explicitly uncomputes every selector sector for small full
statevector comparisons. `return_success=True` exposes just the successful
spectral amplitudes for auditing. Qiskit circuit construction retains the same
PREPARE / controlled walk powers / PREPARE-dagger construction. Tests compare
the lazy and full simulations, native circuits and high-degree precision.

The default degree budget rises from 255 to 1023. An explicit memory budget
(default 512 MiB) checks estimated statevector and encoding workspace before
allocating the large selector state. This is a working-memory estimate, not an
operating-system memory cap; native circuit synthesis has additional costs.

## Error measurements

The analytic polynomial truncation target is **1e-12**. The separate numerical
acceptance limit is **1e-10**, allowing for double-precision encoding, matrix
products and simulated rotations. Passing the analytic bound alone never passes
the experiment. Every tested operator must also meet the numerical residual and
relative inverse-error limits, preserve state norm and have no padding leakage.

For a balanced input in the encoding basis, every spectral coordinate has
nonzero amplitude. The simulator's successful amplitudes therefore expose every
diagonal response of the implemented block. The audit reconstructs that block
as a full matrix X and measures

$$ R = MX-I,\qquad \|R\|_2,\qquad
\frac{\|R\|_F}{\sqrt{N}},\qquad
\frac{\|X-M^{-1}\|_F}{\|M^{-1}\|_F}. $$

The independent classical reference solve runs only after circuit construction
and simulation. It is never an input to the coherent operation. This audit uses
simulator amplitudes; it is not physical tomography or an efficient way to read
out a million classical inverse entries from quantum hardware.

The operator residual is stronger than checking three right-hand sides: for
every nonzero b of each audited matrix,

$$ \frac{\|MXb-b\|_2}{\|b\|_2}\leq\|MX-I\|_2. $$

The campaign additionally tests basis, uniform and random inputs and compares
their residuals with the previous 1e-6 polynomial target on the same matrices.

## Recorded ideal-simulation results

See [the complete report](../results/precision-1000/summary.md) and its JSON
record for individual matrices, environment versions, bounds and references.

| Matrix group | Condition number | Degree | Total qubits | Worst operator residual | Worst-input success |
|---|---:|---:|---:|---:|---:|
| Four earlier families | 1.67–1.97 | 13–16 | 15–16 | 7.3e-13 | ≥0.259 |
| Three controlled families | 10 | 46 | 17 | 1.8e-13 | ≈1e-2 |
| Three controlled families | 100 | 161 | 19 | 1.1e-13 | ≈1e-4 |
| Three controlled families | 1000 | 546 | 21 | 4.7e-13 | ≈1e-6 |

These qubit counts include ten data qubits, one walk ancilla and selectors.
The largest case has a 32 MiB statevector and estimated 256 MiB working memory.
Error-correction overhead is excluded. Worst-input success near 1e-6 means
roughly a million independent attempts per accepted output on that input,
before measurement reconstruction costs. Amplitude amplification is not
implemented. Finite-shot and noisy hardware error do not follow from the small
ideal residual.

## Scope of the invertibility condition

For a real **square** M, `M @ M.T > 0` means positive definite, not elementwise
positive. It is equivalent to nonsingularity, so a mathematical inverse exists.
The implementation applies the reciprocal polynomial to positive singular values
and restores the left/right singular bases; M itself need not be SPD.
It does not solve normal equations or explicitly form M M.T.

For any finite positive singular-value interval, the reciprocal Chebyshev series
converges as degree increases. Its convergence ratio
`r=(sqrt(kappa)-1)/(sqrt(kappa)+1)` approaches one for poor conditioning. Thus
positivity alone provides neither a uniform degree bound nor useful success
probability at fixed resources. The numerical encoder also rejects singular,
numerically singular, complex and non-square inputs.

The report deliberately checks a 1000×1000 diagonal SPD input with condition
number 10000. At target 1e-12 it needs degree 1842 and is explicitly rejected by
the default 1023 budget. Budgets can be raised explicitly; that does not remove
conditioning, floating-point or sampling limits. The 13 successful examples are
not an empirical proof for every invertible matrix.

These are ideal structured statevector simulations. Dense classical
eigendecomposition/SVD still costs O(N³), and dense quantum basis changes still
have substantial gate cost. Native noisy simulation and physical execution at
1000 dimensions remain untested. The earlier hardware-noise limitations in
[the hardware protocol](compact-hardware.md) still apply.


## Run provenance and publication tables

Archived values in `results/` are historical reference runs. The separate clean
`e6769dbe` benchmark matches the manuscript's native depth sequence
66/138/1032/41033 and order-100 noise count 35/64 (echo 10/35); the original
public hardware archive has 72/142/1033/41024 and 34/64 (echo 8/34).
Both are preserved in `publication-evidence/historical-reconciliation.json`
and its linked raw evidence. They are not averaged or silently overwritten.
For new tables and figures use one completed run's `final-report.html` and
`publication-evidence/manifest.json`; see
[publication evidence](publication-evidence.md).
