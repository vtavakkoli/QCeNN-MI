# Dissipative comparison, changing singular spectra, and conditioning

Run the pinned complete validation, including this study:

```sh
python docker/reproduce.py --profile full
```

The host wrapper needs Python 3 and Git, but no scientific Python packages. It
rejects uncommitted source changes, records the exact commit and canonical Git
source hash, and passes both to the container. Windows CRLF source is normalized
by the existing image startup/build policy before the hash is checked. Untracked
generated hardware results are not source changes. An exported validation bundle
uses its `SOURCE_REVISION.json` instead of requiring a `.git` directory.

Plain `docker compose up --build --abort-on-container-exit --exit-code-from
validation` still runs the suite. If no source pin is passed, its commit can be
`unknown`; use the wrapper for publication evidence. Every aggregate run now
freezes its actual source bytes in `source-snapshot.zip` and a per-file SHA-256
inventory **before** experiments start, and checks source stability at the end.
Old archives with unknown original commits are historical records; their source
identity cannot be repaired by assigning a nearby commit retrospectively.

For an installed analysis environment, this study alone is:

```sh
python -m qcenn_mi.dissipative_benchmark \
  --config configs/full.json --output-dir artifacts/dissipative-comparison
```

The full configuration uses orders 10, 100 and 1000. Smoke CI uses order 3,
shorter budgets and fewer noise/dynamic settings. Study success means numerical
integrity and consistency, not that every deliberately difficult case converged.

## Matched comparison

Primary baseline: Zhong-Xia Shang, *A Simple Quantum Linear-System Solver via
Dissipation*, arXiv:2609.12284v1 (10 September 2026), Eqs. (10), (20):

`F_S=(A,-beta)`, `L_S(rho)=Tr(F_S†F_S rho)R-{F_S†F_S,rho}/2`.

We integrate **L_S/2** with DOP853, not an Euler approximation or our discrete
channel relabeled as Shang's solver. This divides its maximum jump-rate bound
of 2 by 2; every tested generator then has `||G|| <= 1`. The padded extension
resets padding toward the reference, but ideal reference-initial trajectories
never populate it. The continuous baseline is independently checked against
physical-basis Liouvillian exponentials for small systems. Tolerance refinement
is additionally recorded for comparison cases through order 100.

Three methods share exactly the archived matrix, RHS, physical initial state
`|r><r|`, observation times and two targets: relative coherence residual <=1e-6
**and** conditional solution-state infidelity <=1e-6. These are the same solution
metrics despite different augmented targets/reference weights. The first
sustained **recorded** time is reported only if all subsequent recorded points
meet both thresholds. A null value is a right-censored failure within budget,
not an infinite convergence time. Observations do not prove between-sample
monotonicity. ODE tails below numerical precision are not exact error estimates.

Methods are the finite-step recurrence, its continuous generator with the same
F, and Shang's differently scaled encoding. The middle control isolates finite
step versus continuous evolution from the reference-weight change. A model time
unit is **not** a Lindbladian-simulation query, channel-synthesis cost or hardware
duration; no runtime speedup or optimal complexity claim follows.

## Changing singular values and condition number

The additional nonsymmetric family is `M_j=U diag(sigma_j) V^T`, with fixed,
independent dense orthogonal U,V and fixed random RHS. Unlike column rotation,
every singular value changes: norm, spectral shape and condition vary. Full
condition sequence is 2,4,10,30,100,100,30,10,4,2, at dwells 32 and 128 updates,
followed by 1024 updates on the final condition-2 matrix. Per-update logical
noise is 0,1e-4,1e-3. Every snapshot's actual dense matrix and RHS are archived.

The physical density is never reinitialized. Changing SVD coordinates applies
`W rho W†`, `W=V_new† V_old`; this is an exact simulator coordinate change,
not a predictor or a quantum preparation step. Gate acquisition/synthesis is
preprogrammed and not free hardware work. An independent reference is used only
for diagnostics and verification of the moving-target bound. Both failures
during ill conditioning and recovery after the final hold remain in trajectories.

## Condition-100 diagnosis

The earlier condition-100 stress failures were **ideal**, at 4096 updates. New
experiments report the same budget with p=1e-6,1e-5,1e-4,1e-3 and the exact
stationary state of `D_p o E`. The latter solves the channel fixed-point equation,
without using a solution oracle. Its trace, positivity and fixed-point residual
are checked. A longer ideal order-10 run uses an independently powered small
superoperator to test finite-budget recovery. That numerical shortcut is not a
claim of cost-free channel powering on hardware.

The static bound has contraction parameter `theta=1/[2(kappa^2+1)]`; noise can
accumulate over its slow relaxation. Absolute coherence readout also contains
`1/gamma` and a reference-probability denominator. Consequently, an accurate
augmented state need not give a uniformly accurate absolute solution at large
condition number. Exact measured stationary residuals, rather than a sufficient
upper bound alone, identify whether extra iterations can meet the chosen target.

Global p is depolarization of the **entire padded logical register after one
update**. It is not an error probability per gate, per CX, per qubit, or per
microsecond. Continuous comparison cases here are ideal. Earlier native small
CX-only and larger Chebyshev circuit noise studies remain separate. None uses
physical device calibrations or establishes a QPU noise threshold.

## Evidence integrity

`result.json`, `matched_comparison.csv`, `condition_100_noise.csv`,
`spectrum_tracking.csv`, `figures/`, `inputs/` and `MANIFEST.json` are generated
together. The verifier checks raw input hashes and regenerates CSV text from raw
results, so editing a table and rehashing it does not change the scientific data.
The aggregate final report incorporates the comparison and all new figures.
The manuscript remains outside the repository.
