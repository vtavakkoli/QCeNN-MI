# Polynomial residual-response recurrence

This implementation is a distinct extension of the resident-state collision
channel. It does not compute or load a target inverse/solution during circuit
construction. For `m >= 0`, with `ell=m+1`, it uses

`G_m = I - (I-G)^ell`, `C_m = (I-G)^(ell/2)`.

`E_m(rho) = C_m rho C_m† + Tr(G_m rho) |r><r|`.

The dense compiler constructs `F_m=U sqrt(1-(1-S²)^ell) V†` from the SVD of the
input residual F, then uses its Julia dilation and the existing controlled-swap
environment circuit. Both coherent retention and device reset reuse retain the
same data register. Degree zero reproduces the original compiled dilation.
This is an explicit small dense circuit implementation, not an efficient QSVT
or matrix-oracle implementation.

All eigenvalues of G are in [0,1]. The transformed response stays in [0,1],
vanishes exactly on the original one-dimensional kernel, and has gap at least
`g_m=1-(1-g)^ell`, where `g=1/(kappa²+1)`. Kraus completeness proves CPTP.
The target overlap q with the reference remains at least 1/2. Its infidelity
decreases by at least `q*g_m` times the current infidelity each update, and the
universal contraction uses `theta_m=g_m/2`. The solution extraction probability
is unchanged. The map generally differs from `E_0^ell`.

The improvement is per macro update. `g <= g_m <= min(1,ell*g)`, so charging
ell primitive residual uses removes any claim of asymptotic improvement from
the update count alone. Dense matrix synthesis and the square-root dilation
are charged separately; no free polynomial oracle is assumed.

For a moving target with purified-distance variation v, the bound is
`e_next <= sqrt(1-theta_m)*(e+v)+xi`, capped by the distance range. Logical
global depolarization gives

`e_next² <= (1-p)*(1-theta_m)*min(1,(e+v)²) + p*(1-1/D)`.

The logical-noise studies separate fixed p per macro update from the illustrative
cost-coupled convention `p_m=1-(1-p_base)^ell`. Neither is a physical gate-error
model. Separate small native Aer tests include CX depolarization and an error
after ancilla reset, with other gates/readout ideal.

## Reproduce

`python docker/reproduce.py --profile full` now includes `polynomial-recurrent`.
The aggregate full profile has 25 software stages including the final report.
The reduced smoke configuration is for CI and does not establish large-order
performance. A stand-alone study is

```sh
python -m qcenn_mi.polynomial_benchmark --config configs/full.json --output-dir NEW_DIR
```

The full static grid uses orders 2,4,10,100,1000, conditions 2,10,100,1000,
four matrix families, and degrees 0,1,2,4,8,16. Different initial states are
included at all small orders and selected large nonsymmetric fixtures. First
sustained recorded convergence is not an exact unsampled first hitting time.
Undefined initial coherence readouts are retained rather than assigned a fit.

Large static observables use an exact renewal expansion of the quantum channel,
with diagonal reset-flux evolution and FFT evaluation of reference coherences.
This representation is independently checked against entire density matrices.
It is a classical simulator optimization, not a replacement quantum algorithm.
Dynamic experiments retain full reduced density matrices, vary singular spectra,
conditioning through 1000, RHS and basis orientation, include a jump and end in
a constant well-conditioned hold. Inputs and seeds are archived.

Small explicit circuits provide Choi/Kraus/unitarity/kernel checks and independent
complete-circuit comparisons. Actual converged native circuits are compiled only
within 128 updates and a 50000-CX precheck budget; all resource rejections remain
records. Large native gate costs are never extrapolated from register width.
The companion prior comparison reads the earlier stage's actual matrices and
uses its exact observation times/targets; standalone runs without those records
explicitly report the missing matched comparison.

`polynomial_report.verify(PATH)` checks checksums, actual inputs and CSVs against
raw JSON. Failed accuracy criteria are separate from numerical integrity.
Every experiment records exact source provenance. Manuscript sources remain
outside this repository. Physical QPU performance is unmeasured.

## Composition from the supplied residual unitary

`polynomial_composition.py` implements the same channel by ell controlled calls to the original U_F. A coherent first-residual flag prevents further residual calls on a branch that has already jumped; controlled swaps reset all jump branches only at the end. The no-jump branch is C^ell. Jump effects sum to G_m. No transformed residual factorization or target state is supplied. Extra flags and controls are charged explicitly. The budgeted composition campaign preserves timeouts and synthesis rejections. The dense direct compiler and the controlled composition have different costs, and both remain separate from an efficient general-purpose matrix oracle.
