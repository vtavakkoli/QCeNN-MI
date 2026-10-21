# Chebyshev-filtered recurrence and implementation cost

This is distinct from the power response `I-(I-G)^(m+1)`. It does not replace
the retained negative equal-call results for that method. Polynomial filtering
and QSP are established techniques (Lin and Tong, Quantum 4, 361, 2020;
Gilyen et al., STOC 2019). The extension supplies a resident-state reset
channel and an explicit real-Julia circuit in this access model.

For a certified gap `g`, use
`r_d(lambda)=T_d((1+g-2lambda)/(1-g))/T_d((1+g)/(1-g))`.
The no-event operator is signed: `C=r_d(G)`. Reset every other outcome to
the known reference. `I-C*C` is positive and Kraus completeness proves CPTP.
For `d>=1`, the original one-dimensional kernel and fixed state are preserved.
With `eta=sech(2d*atanh(sqrt(g)))`, infidelity contracts by at most
`1-q*(1-eta**2)`. The moving-target and logical-noise overlap bounds carry
over with this rate; negative filter values do not invalidate them.

## Explicit circuit

`chebyshev_recurrent.qsp_filter_circuit` synthesizes phases for the even
polynomial `r_d(1-x*x)` of degree `2d`. The real Julia residual unitary has
signal blocks unitarily equivalent to `[[x,-s],[s,x]]`. Flag phase rotations
convert these to the standard Wx QSP convention. A selector controls a phase
sequence and its complex conjugate. Both share the same real input unitary,
so extraction of the imaginary signal block uses exactly `2d` input calls.
This construction assumes a **real Julia dilation**, not arbitrary black-box
block-encoding access. pyqsp 0.2.0 supplies symmetric phase synthesis.

Every non-success pair of signal flags is ORed into a reset flag and swaps
the data with a fresh known reference register. Data persist across updates.
Width is `2*n_r+3` with reset reuse, or `n_r+K*(n_r+3)` with retained
environments. Reset count is `(K-1)*(n_r+3)`. The success filter is verified
independently against a dense functional calculus; complete circuits are
compared with density evolution for both workspaces and complex initial data.

Numerical phase synthesis deliberately attenuates the ideal polynomial by
`1-1e-12`. The resulting implemented channel remains CPTP but has small kernel
leakage; it must not be called an exact fixed-state implementation. Phase
coefficient errors are floating-point estimates, not interval certificates.
Degrees above 64 and large reset-density simulations are explicitly rejected.
Large matrix simulations use spectral channels, not full QSP circuits.

## Query comparison and its limits

The power certificate obeys
`1-q*(1-(1-g)^ell) >= (1-q*g)^ell`; it cannot establish improved certified
decay per charged primitive call. In contrast, the Chebyshev response needs
`d=O(1/sqrt(g))` for constant suppression and `O(log(1/eps))` macros.
For augmented-state purified distance, ideal real-Julia query cost is
`O(kappa*log(1/eps))`. The diagonal `diag(1,1/kappa)`, RHS `(1,0)` and
initial second solution basis state give the original channel a lower bound
`Omega(kappa**2*log(1/eps))`. This is a separation between specified recurrent
channels, not an improvement over the best QLSP or classical algorithms.
Readout, extraction probability, loading and dense preprocessing remain costs.

## Matched experiments and complete cost

```bash
python -m qcenn_mi.chebyshev_benchmark --config configs/full.json --output-dir NEW
docker compose up --build --abort-on-container-exit --exit-code-from validation
```

The full profile adds this as a new stage. It retains exact source, inputs,
phase arrays, QPY circuits, CSV tables, failures and resource rejections.
Static inputs, initial states and accuracy targets match across methods;
macro caps are 64 for Chebyshev and the original conditioning-dependent
budgets for power methods. These unequal caps are not an equal-budget speedup
benchmark. A separate analytic slow-mode family is cross-checked by direct
small-density evolution and reports its ideal query counts explicitly.

Small cost cases actually compile and simulate the complete converged native
circuit. They require the joint 1e-6 residual/infidelity criterion plus a state
error reserve `P <= .01*gamma/200` for a common 1e-2-confidence residual budget.
Both the first joint-accuracy time and the stricter cost time are retained.
Actual coherence measurement basis rotations are synthesized and counted.
Hoeffding shots include all N coherences and the reference population. A
rejected tighter readout certificate remains rejected rather than assigned
zero shots. Input preprocessing, phase synthesis, transformation, native
compilation, repeated executions and readout enter the reported model.
Simulator-only spectral replay and diagnostic matrix extraction are timed
separately and excluded from implementation latency. Circuit assembly is
included with native compilation. Accuracy crossings are selected offline
from benchmark reference records; they are not a free hardware stopping test.

The duration model uses declared CX/one-qubit/reset durations and dependency
scheduling, not calibrated device measurements. Reuse refers to repetitions
of the same compiled M,b instance; changing b generally changes this encoding.
Synthetic noise applies actual CX and reset errors to native circuits.
Logical fixed-macro and query-coupled depolarization are separate models.
No physical hardware job or unmeasured hardware speedup is claimed.
