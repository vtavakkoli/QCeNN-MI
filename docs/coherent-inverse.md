# Standalone coherent inverse-action prototype

## What is quantum, and what remains classical

The quantum circuit applies an approximate inverse to an input state. It does
not receive a classical inverse, target solution, optimized solution angles,
or classically computed inverse polynomial. It contains no mid-circuit reset,
measurement or classical feedback. All ancilla success conditions are measured
at the end. This is a postselected coherent operation, not a deterministic
unitary implementation of a nonunitary inverse.

Classical work remains: input validation, Gershgorin bounds, a dense square root
to construct the block encoding, state preparation/compilation and final readout.
Dense input encoding limits scalability; no efficiency or speed-up is claimed.
This is a separate linear-algebra baseline, not the original local QCeNN ansatz.

## Construction

For a real symmetric matrix with certified positive spectral bounds `a,b`, set

- `tau = 2/(a+b)`;
- `B = I-tau*M`, with `||B|| <= q = (b-a)/(b+a) < 1`;
- `U_B = [[B, sqrt(I-B²)], [sqrt(I-B²), -B]]`.

The upper block of `U_B` is `B`. A selector prepares a uniform superposition of
orders `k=0..d`. For branch `k`, the circuit applies `U_B` exactly k times with
fresh flag qubits. Distinct flags ensure that projection onto all-zero flags
gives `B^k`, rather than an incorrect power of a reused unitary dilation.
Uncomputing the selector and projecting every non-data qubit onto zero gives

`(1/(d+1)) * sum(B^k, k=0..d) |b_normalized>`.

Thus the unnormalized inverse approximation is recovered with known multiplier
`tau*(d+1)*||b||`. Its ideal linear-system residual obeys
`||M x_d-b|| <= q^(d+1)||b||`. For a reconstructed full inverse, the Frobenius
residual is bounded by `sqrt(n)*q^(d+1)`.

No classical matrix powers or polynomial summation enter this circuit builder.
Only individual controlled input-encoding blocks are constructed classically.
This deliberately simple SELECT uses `d*(d+1)/2` controlled block calls and
`log2(n)+d+ceil(log2(d+1))` total qubits. Degree is capped at 7 in this prototype.

## Measurement and verification

`build_inverse_action(M, degree=7)` returns a reusable operator. Use
`operator.circuit(rhs)` for the structured circuit suitable for native-gate
compilation or QPY export. `operator.ideal(rhs)` obtains simulator amplitudes
solely for validation. `operator.measure(rhs, shots=131072)` uses actual Aer
measurements, postselects all ancillas and performs full data-qubit tomography.

Measured magnitude comes from success frequency:
`||x|| = tau*(d+1)*||b||*sqrt(p_success)`.
There is no least-squares magnitude refit. For the real SPD domain, positive
`b.T @ x` fixes the tomography global sign. Classical readout is an experiment
on the quantum state, not part of coherent inverse action. Shot residuals are
reported separately from ideal polynomial truncation errors.

The benchmark solves classically only after constructing/executing its circuits,
to calculate independent reference errors. Tests disable `numpy.linalg.inv` and
`numpy.linalg.solve` during circuit construction to guard against answer leakage.
Other tests check linearity, the analytic truncation bound, degree convergence,
probability normalization, finite-shot readout and multi-ancilla postselection.

## Simulation representation and hardware costs

For ideal/finite-shot noiseless simulation, each individual controlled SELECT
block is fused into its exact small unitary matrix. The full inverse/polynomial
is never fused or precomputed. A test compares this representation with the
structured circuit. Explicit Aer basis selection avoids unsupported multiplexer
assembly. Measurement audits identify the fused simulator representation.

QPY exports and native resource counts use the structured controlled-block
circuit, not the fused simulator representation. Counts are compiled to
`rz,sx,x,cx` without a hardware coupling map; routing and calibration can increase
cost further. Noise on fused simulator gates would not represent native-gate
noise, so this campaign makes no such noise claim. Future noisy/hardware tests
must use the structured circuit compiled for the actual target.

Default acceptance: full-inverse ideal residual <1e-3, finite-shot residual <0.15,
superposition relative residual <1e-3, and decreasing ideal residual across
1/3/7 degrees. Raw counts and QPY files are regenerated and uploaded as CI
artifacts. JSON and Markdown reference reports are committed.

The supported matrix domain is real symmetric 2×2/4×4 with a positive Gershgorin
lower bound. This sufficient SPD test rejects some valid SPD matrices. Signed,
indefinite, nonsymmetric, complex and larger matrices are not supported here.
