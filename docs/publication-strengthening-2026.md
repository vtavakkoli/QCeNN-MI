# Recurrent Chebyshev publication strengthening (2026-10-26)

## Reproducible experiments

Use a clean checkout and pinned dependencies (or the repository's locked
Docker image), with single-threaded BLAS for reproducible timings:

```bash
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q tests/test_temporal_chebyshev_publication.py
python -m qcenn_mi.temporal_chebyshev_publication --smoke --output-dir artifacts/temporal-smoke
python -m qcenn_mi.temporal_chebyshev_publication --output-dir artifacts/temporal-full
python -m qcenn_mi.phase_bias_audit --conditions 2 10 --output-dir artifacts/phase-bias
```

Each run refuses to overwrite an existing evidence directory. Preserve all
failures and equal outcomes. A GitHub Actions artifact contains the data for
the run's exact source revision. Inspect the workflow result before citing
numbers in a manuscript.

## Matched design / fair comparison boundaries

Three methods are compared at identical (M(t), b(t)), snapshot, seed, matrix
condition number, residual and conditional-infidelity criteria, and a common
primitive residual-unitary query **ceiling per snapshot**.

* **resident:** the signed Chebyshev reset channel updates the same physical
  density operator through successive changing-matrix snapshots.
* **fresh:** the very same signed-filter reset channel starts from the known
  reference density at every snapshot, with exactly the same number of macro
  updates and primitive calls as resident.
* **one_shot:** a freshly prepared, postselected direct Chebyshev filtered
  reference state uses a larger degree consuming the available single-attempt
  query ceiling. The success probability and **expected retries** are recorded.
  Repeating this one-shot preparation to acquire accepted states costs more
  queries: per-accepted-state expected calls equal 2d/p_success. Do not compare
  postselected fidelity against unconditional CPTP output without describing
  this difference.

The resident/fresh comparison is the strict paired comparison. The one-shot
comparator provides a *stronger static filter within one circuit* but is not
an equal-expected-cost comparison unless retries and the required number of
measurements are accounted for. Neither preparation overhead nor full-vector
readout are included in this benchmark; no end-to-end speedup follows.

Inputs are deterministic rotating right-singular-vector matrices with exact
controlled conditioning. Both RHS variation and gradual or abrupt matrix
changes are included. The exact augmented target is evaluated **only after**
updates; no classical solve influences a quantum-state update. The independent
Chebyshev cross-check uses numpy.polynomial.chebyshev rather than the stable
response implementation. The reference reduced-channel implementation in the
repository remains an additional independent cross-check; complete native
circuits are tested separately for small sizes.

## Fidelity, noise, and failure accounting

The simulator records signed coherence relative residual, augmented purified
distance, conditional solution infidelity, trace error, minimum density
eigenvalue, per-method pass/fail, and per-snapshot query charges. Conditional
solution fidelity is not substituted for augmented-state fidelity. Each
logical noise probability p_oracle is converted into
p_macro=1-(1-p_oracle)^(2d) for a macro update. One-shot results receive
phenomenological depolarization **after postselection**. This is **not** a
native-gate error model or a calibrated noise treatment.

The separate phase-bias audit compiles a complete small real-Julia QSP signal
circuit; it checks the actual extracted success block against the attenuated
Chebyshev polynomial, the floating-point coefficient residual, kernel
leakage, state trace/positivity, and incremental bias from finite phase
synthesis and logical noise. The coefficient residual is *not* an
interval-certified bound. A failure of numerical phase synthesis must stay
a failure, not become a skipped passing data point.

Do not reuse the ideal spectral-channel results as evidence of large-scale
compiled QSP: 1000-dimensional examples remain reduced-channel simulations.

## Revision of the journal paper

**Suggested narrower contribution claim:** The work specifies recurrent,
resident-data Chebyshev quantum channels and their exact/finite-synthesis
realizations; it studies whether *carrying a quantum state through changing
inputs* reduces transient re-preparation costs against restarting the same
channel, subject to explicit matrix drift, noise, oracle, postselection and
measurement assumptions. The sign-changing polynomial, the original
residual-reset channel, QSP and Chebyshev filtering are known ingredients.
No general advantage over direct one-shot QSVT or modern quantum linear
system solvers has been established.

**Proposed revised abstract (replace the existing abstract only after the
matched CI evidence is finalized):**

> We study persistent quantum-channel dynamics for matrix inverse actions
> under static and time-dependent inputs. We introduce a deferred-reset
> power-polynomial map and a signed Chebyshev recurrent channel, establish
> their complete positivity and stationary-state convergence, and give
> real-Julia circuit realizations with explicit residual-oracle charges.
> A conditional Chebyshev query separation holds against the specified
> residual-reset baseline, but not against leading one-shot linear-system
> algorithms. To isolate the value of recurrence, we compare persistent
> evolution with restarting the same channel and with direct postselected
> Chebyshev preparation on matched changing-matrix schedules. We separately
> report finite phase-synthesis error, logical-noise bias, conditioning
> failures, postselection overhead, measurement cost, and implementation
> limits. Large-scale results are reduced-channel simulations; complete
> circuits and synthetic native noise are restricted to small examples.
> We do not claim an end-to-end quantum speedup.

### Editing / length reduction plan

1. Keep the title, access model, three central theorems, circuit figures, and
   one **paired time-varying comparison** near the main results.
2. Move auxiliary snapshot Chebyshev-LCU derivations and extended classical
   recurrent inversion history to supplemental material **without deleting
   references or proof dependencies**. Compress duplicated novelty disclaimers
   into one explicit access-model table.
3. Combine power-polynomial failure tables with an explicit Chebyshev/static
   comparison table; keep *every* negative and resource-rejected result in a
   supplement with archived raw CSV/JSON.
4. Distinguish the reference reset channel from Shang's continuous Lindbladian,
   full optimized QLSP systems, and direct one-shot postselected filtering.
5. Rerun and rebuild the paper from editable LaTeX, review figure captions,
   revise counts and cross-references, then visually inspect **every** page.

**Blocking source issue:** The current GitHub tree contains research source
and results, but not the editable 21-page IEEE Access manuscript
(`.tex`, figures, bibliography). Do not assert the PDF has been shortened,
regenerated, or source-verified until those editable files are added and a
source-identical PDF is compiled.

## Required submission gates

Run `python -m qcenn_mi.publication_preflight --repository . --evidence
artifacts/temporal-full --manuscript-source paper/main.tex --manuscript-pdf
paper/main.pdf --output artifacts/submission-preflight.json --strict`
from a checkout with the manuscript assets and compiled PDF. This only checks
machine-verifiable integrity, not correctness of claims, reference DOIs, visual
typesetting, journal limits, or author approval; manual sign-off is always
required. Consult IEEE Access's *current* author instructions and policies,
including AI-assistance and data availability declarations, at submission.
