# Large recurrent-channel simulation, noise, and dynamical x1 tracking

This study extends the recurrent collision channel to configurable matrix orders
10, 100, and 1000. The large runs are classical simulations of the exact reduced
quantum channel. They are not native-gate simulations of the large circuits or
physical quantum-device experiments. The smaller complete circuits provide
independent implementation checks.

## Exact reduced representation

For the residual operator F in [the circuit derivation](recurrent-quantum.md),
write F=U S V† and represent the data density matrix in the basis V. Define
g_i=S_i², c_i=sqrt(1-g_i), and r_i=(V†|reference>)_i. One complete quantum update is

\[
 \rho'_{ij}=c_i c_j\rho_{ij}+
 \left(\sum_l g_l\rho_{ll}\right)r_i r_j^*.
\]

This updates every entry, including coherences and mixed-state components. It
does not substitute a classical vector iteration for quantum evolution. One
update costs O(P²) simulator arithmetic and memory for a padded register of
dimension P. Dense input SVDs remain O(N³) classical preprocessing.

The implementation never renormalizes the density matrix to hide trace drift,
fits an output scale to a reference solution, or applies classical residual
correction. Complete-circuit comparisons include ideal and noisy cases at
orders 2, 3 and 10. Time-varying basis changes are also checked against complete
circuits in the unit tests.

## Noise model

Each specified logical update probability p adds

\[
 \mathcal D_p(\rho)=(1-p)\rho+
 p\operatorname{Tr}(\rho)I/P.
\]

The default probabilities are 0, 10^-5, 10^-4, 10^-3 and 10^-2. This is global
depolarization once per complete update on the **padded data register**, including
the reference coordinate. It is not a CX-gate error rate. It does not include
gate synthesis, routing, imperfect resets, readout noise, or a backend's measured
calibration. The model is cross-checked using Qiskit Aer's depolarizing channel
in the complete small circuits.

The earlier native CX-noise experiment and this logical-noise experiment remain
separate. Passing either configured profile does not establish general physical
noise resilience. Noise floors and failed accuracy targets are retained.

## Inputs and accuracy

The default main study uses three real nonsingular families: dense SPD,
nonsymmetric, and symmetric indefinite, at condition numbers 2 and 10. Ideal
tests use basis-first, uniform and signed random right-hand sides. The four
nonzero noise profiles use the signed random RHS. A condition-100 nonsymmetric
stress probe is included at each requested size.

Main update budgets are 512 at condition number 2 and 4096 at condition number
10. Stress cases use 4096. Ideal relative-residual target is 10^-6; the noisy
target is 10^-2. These are experimental criteria, not a claim that all inputs or
noise strengths meet them. The software/numerical-integrity verdict and noisy
accuracy verdict are stored separately.

Actual matrices, RHS vectors, and independent reference solutions are saved in
NPZ files with SHA-256 digests. The solver does not receive a reference vector.
Different right-hand sides have different residual encodings. Selected inverse
actions are tested; all columns of a large inverse are not reconstructed.

## Meaning of x1

The plotted first component is obtained from the augmented state's coherence:

\[
 \widehat x_1(k)=\frac{\|b\|_2}{a\|M\|_2}
 \frac{\operatorname{Re}\rho_{0,r}(k)}{\rho_{r,r}(k)}.
\]

Here r is the reference coordinate and a=sigma_min(M)/||M||. The initial
reference state gives a zero initial estimate. This formula preserves absolute
scale and sign; it does not refit the output to a classical solution. Exact
simulator coherences are used for plots. On hardware, estimating them would
require final measurements, not a free continuous observation of the quantum
state. Recorded values never determine a subsequent update or stopping time.

## Dynamical matrices

The dynamical family is M(k)=M(0)G(theta(k)), where G rotates the first two
solution coordinates. Every matrix remains nonsingular with condition number 2.
The angle initially stays zero, then drifts continuously, jumps at update 960,
and eventually stays fixed. The data state is not reinitialized at transitions.

The residual basis transforms covariantly. If V(theta)=T(theta)†V(0), then the
change of simulation coordinates is

\[
 W=I+B\,[G(\Delta\theta)-I]B^\dagger,
 \qquad B=V(0)^\dagger[e_0,e_1].
\]

Applying W rho W† costs O(P²) using its rank-two structure. It leaves the
physical state unchanged before the new residual channel is applied. This is
not a predictor that moves the estimate toward a known answer. The construction
is independently compared with complete circuits for changing matrices.

A manufactured base solution defines a reproducible benchmark RHS and gives a
visible, analytically checkable first component. It is not loaded as an initial
or intermediate quantum state. A separate dense solve checks the base reference;
the moving reference is G(theta)† times that vector. Matrix values are
preprogrammed; this is not autonomous live acquisition or a general dynamic
noise-resilience theorem.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m qcenn_mi.recurrent_scaling_benchmark --config configs/full.json \
  --output-dir results/recurrent-scaling
```

Use a fresh directory. The full configuration uses three worker processes; the
smoke configuration uses one worker and order 10. Full Docker validation now
includes this study and has 22 stages in total, counting the final report.

Outputs include `result.json`, three CSVs, actual input NPZ files, `REPORT.md`,
PNG/SVG scientific figures, and a verified `MANIFEST.json`. Check them with
`qcenn_mi.recurrent_scaling_report.verify(PATH)` from the matching source.
The source commit, source fingerprint, dependency versions, input archive audit,
and simulation scope are recorded. The uploaded historical run is never merged
or averaged into the new results.
