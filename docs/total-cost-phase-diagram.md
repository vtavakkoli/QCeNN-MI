# When is persistent Chebyshev tracking cheaper?

This is an **accuracy-constrained, assumption-weighted total-cost** experiment.
It measures *conditional regions*, not a quantum speedup or device performance.

## Comparable workload and information boundaries

At snapshot t, three candidate policies can create a solution-encoding state:

- **resident** persists the density state through all changing inputs since
  t=0, applying K filtered macro updates per snapshot;
- **fresh** prepares a reference density at t and applies the same family of
  Chebyshev CPTP map K times at that snapshot;
- **direct** applies a Chebyshev polynomial filter once to a freshly prepared
  reference and postselects. Geometric retries and preparations are charged.

Methods may independently choose Chebyshev degree d and macro updates K,
but *only from the same predeclared, query-budgeted candidate grid*.
A fixed (d,K) is selected on separate validation seeds to reach the joint
conditional solution-infidelity and relative-residual pass coverage requirement.
The fixed policy is then evaluated on untouched test seeds. **Only when both
resident and fresh pass the heldout coverage criterion can lower cost be called
a conditional winner**. Infeasibility and ties are not erased; direct filtering
appears as a stronger additional comparator.

The density-matrix simulator computes ideal/phenomenologically noisy state
properties exactly for scoring. It does not estimate residuals from actual
finite-shot experiments. The output therefore provides **cost-per-assumed
readout copies and exact-state validation**, not shot-certified accuracy.
Classical solves are used for scoring only, never for quantum-state updates.

## Readout and preparation cost

Let \\(\\mathcal T\\) be the set of zero-based readout checkpoints, \\(S=(N+1)H\\)
the declared independent copies per checkpoint (H is shots per coherence
setting), \\(w_q\\) the equivalent-unit primitive oracle cost, \\(w_r\\)
the macro reset cost, \\(w_p\\) the reference preparation cost,
\\(w_m\\) the measurement cost and \\(w_c\\) the one-time input compile cost.

With a constant policy (d,K) and *destructive* measurement of each copy,

\\[
Q_{\\mathrm{resident}}=2dK S\\sum_{t\\in\\mathcal T}(t+1),\\qquad
Q_{\\mathrm{fresh}}=2dK S|\\mathcal T|.
\\]

For a method with its own independently selected degree and K,

\\[
C_{\\mathrm{resident}} =
S (2d_rK_r w_q+K_r w_r)\\sum_{t\\in\\mathcal T}(t+1)
+S|\\mathcal T|(w_p+w_m) +(t_{\\max}+1)w_c,
\\]

\\[
C_{\\mathrm{fresh}} =
S (2d_fK_f w_q+K_f w_r)|\\mathcal T|
+S|\\mathcal T|(w_p+w_m) +|\\mathcal T|w_c.
\\]

The compilation term assumes each input circuit is created once and can
be used across independent measurement copies. This is **optimistic** for
changing input families, and is displayed as its own component. Resident
readout requires repeated full path preparations (or independent copies
propagated live in parallel), never cloning an unknown state.

A sufficient *cost* condition for resident to beat fresh, after both
independently meet the same heldout accuracy requirement, is the exact
inequality

\\[
S (2d_rK_r w_q+K_r w_r)\\sum_{t\\in\\mathcal T}(t+1)
+(t_{\\max}+1)w_c
<
S (2d_fK_f w_q+K_f w_r)|\\mathcal T|+|\\mathcal T|w_c.
\\]

If both methods have the **same degree, updates and nonnegative costs**,
the resident total-cost model is never strictly better for ordinary
destructive readout. Any reported resident benefit must come from a
*lower required (d,K)* or from additional explicit workload assumptions.
This is an important potential negative outcome.

For **direct** postselection, if the accepted probability at t is p_t,
the mean per-accepted-copy calls are \\(2d/p_t\\) and the mean reference
preparations are \\(1/p_t\\); costs must not be compared before charging
retries. Hardware probability estimation and correlated shot failures are
not implemented.

## Sweeps and scope

The experiment sweeps drift in **radians per input snapshot** (rotating
singular-vector angle, RHS modulation or synthetic spectral changes, depending
on family), query-coupled logical depolarization, and destructive readout
interval; it records independently the *actual moving-target projector
norm* for each trajectory. Matrices are small, real and square; no native
large-N QSP hardware tests are claimed.

- **Validation and test:** disjoint, fixed seeds (defaults 23, 29 and 41, 43).
- **Accuracy:** defaults to residual <= 0.05, conditional solution infidelity
  <= 0.01, joint coverage >= 90% of measured checkpoints. Tighten to 1e-3
  to reproduce the paper's stricter failure behavior. The relaxed thresholds
  are an *exploratory frontier*, not a change to the original manuscript's
  acceptance rules.
- **Costs:** default normalized equivalent units: per oracle 1,
  preparation 5, per macro reset 2, measurement execution 0.1,
  per unique compiled matrix 10. **These are illustrative, not calibrated
  physical latencies.** They are CLI configurable; report sensitivity before
  claiming a total-cost regime.
- **Native proxy:** optional `--native-evidence /path/to/native-limitations/result.json`
  matches only explicitly compiled (N, condition, degree) single-macro
  circuits. Its gate-count-weighted extrapolation is marked non-calibrated
  for changing inputs, and direct filtering is omitted because there is no
  comparable compiled direct circuit in that evidence.

### Quick smoke

```bash
.venv-qcenn/bin/python -m qcenn_mi.total_cost_phase_diagram \\
  --smoke --output-dir artifacts/total-cost-smoke
```

### More informative full frontier

```bash
nohup .venv-qcenn/bin/python -u -m qcenn_mi.total_cost_phase_diagram \\
  --size 2 --condition 10 --horizon 12 --query-budget 256 \\
  --target-residual 0.01 --target-infidelity 0.001 \\
  --required-coverage 0.9 --shots-per-setting 16 \\
  --drifts 0 0.01 0.05 0.2 0.6 \\
  --noises 0 0.000001 0.00001 0.00002 0.00005 0.0001 \\
  --intervals 0 1 3 6 \\
  --output-dir artifacts/total-cost-kappa10 \\
  > total-cost.log 2>&1 < /dev/null &
```

A new output directory is required. Review `summary.csv` for each
drift/noise/readout cell and its selected (d,K), validation coverage,
**heldout** coverage, total cost and winner status. `result.json`
retains per-test-checkpoint residual, infidelity, target motion,
selected policy and cost components, and `MANIFEST.json` hashes the files.
Because a hardware-independent fixed-shot proxy cannot certify output
accuracy, the manuscript must distinguish such phase diagrams from real
measured time-to-solution.
