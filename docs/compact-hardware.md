# Compact coherent inversion and hardware simulation

## Qubits: matrix dimension is not a qubit count

For inverse action, the vector is amplitude encoded in `ceil(log2(N))` data
qubits. The inverse is applied as an operator; the circuit does not allocate
one qubit per row or per matrix entry. Non-power-of-two inputs are padded.

| Matrix size | Data qubits | Degree-7 old total | Compact degree-7 total |
|---|---:|---:|---:|
| 2×2 | 1 | 11 | 5 |
| 4×4 | 2 | 12 | 6 |
| 10×10 | 4 | 14 | 8 |
| 100×100 | 7 | 17 | 11 |
| 1000×1000 | 10 | 20 | 14 |

Compact total = data + one walk ancilla + `ceil(log2(degree+1))` selector qubits.
Adaptive degree can require more selectors; some tridiagonal fixtures use one
more qubit. These counts exclude error-correction overhead and any provider
compiler ancillas. A 42-qubit raw capacity is sufficient for the named circuits,
but does not establish usable depth, connectivity, noise or sampling performance.
Full classical inverse readout remains much more expensive than inverse action.

## Compact approximation

The method retains explicit classical dense eigendecomposition/SVD input encoding.
For positive encoded singular values in `[a,b]`, set `c=(a+b)/2`, `h=(b-a)/2`,
`r=h/(c+sqrt(a*b))`, and scaled value `t=(sigma-c)/h`. The reciprocal expansion is

`1/(c+h*t) = (1/sqrt(a*b)) * [1 + 2*sum((-r)^k T_k(t), k>=1)]`.

A single walk flag implements a rotation with top block `T_k(t)` when raised
to power k. LCU weights are the analytic Chebyshev coefficients. Binary powers
of the known multiplexed input walk use scaled rotation angles, reducing the
SELECT construction to one multiplexed gadget per selector bit. The circuit
does not receive inverse eigenvalues, a fitted solution state or a classical
residual correction. This is a dense compiled spectral baseline, not an efficient
black-box/QSVT data-access construction or a new quantum advantage claim.

The ideal relative residual is bounded by
`2*sqrt(b/a)*r^(degree+1)/(1-r)`. Degree adapts to the requested tolerance, subject
to an explicit degree budget. Condition number 100 now reaches residual below
1e-8 at degree 92, compared with about 0.85 for the prior degree-7 Neumann test.
The higher degree uses more selectors, not one fresh ancilla per factor.

`operator_circuit()` exposes the input-independent operation for already prepared
data. `circuit(rhs)` defaults to fusing the known input basis change into input
state preparation, reducing native gate count. This computes U.T @ b for input
encoding, not an inverse solution. Set `fuse_input=False` to retain the separate
input-basis unitary. Tests compare both modes and reuse the same operator on
different prepared states. Dense O(N³) encoding cost remains a major limitation.

## Hardware-simulation workflow

`python -m qcenn_mi.hardware --qubit-budget 42` prepares 2×2, 4×4, 10×10 and
100×100 cases. It performs:

1. Compact inverse construction with target ideal tolerance 1e-6.
2. Compilation to rz/sx/x/cx on a bidirectional line with a connected active
   register. The 42-qubit parameter checks capacity; the topology is a stand-in,
   not a statement about any particular 42-qubit device.
3. Native Aer statevector execution. Final routing permutations are restored
   before computing the inverse-action residual.
4. An independent reference projection appended **after** inverse action. Its
   reference is solved classically only after construction/native execution.
   It tests the output; it does not fit or correct the inverse circuit.
5. Ideal finite-shot execution and synthetic gate/readout-noise execution of
   the routed verifier. No noise is attached to fused ideal polynomial gates.
6. QPY exports, raw-count audits, resource reports, success rates and confidence
   intervals. QPY order is `[inverse_action, reference_verifier_with_measurements]`.

Noise uses depolarizing errors on SX/X and CX, plus symmetric readout error.
RZ is modeled as a noiseless virtual rotation. Profiles are synthetic:

| Profile | One-qubit error | CX error | Readout flip |
|---|---:|---:|---:|
| optimistic_synthetic | 1e-5 | 1e-4 | 0.005 |
| stress_synthetic | 1e-4 | 1e-3 | 0.02 |

2×2/4×4/10×10 use exact density-matrix channel simulation with 32768 measurement
shots. The 100×100 run uses statevector noise trajectories with 64 shots to bound
runtime; its confidence intervals are wide. Increase `--large-noise-shots` for
higher precision. The benchmark does not treat a small sample as certification.

The observed echo-return fraction is all-zero reference returns divided by
ancilla-success shots. It includes verifier gate errors and readout errors, and
is not a tomography-derived fidelity or full signed inverse reconstruction.
The provisional noisy gate requires at least 20 accepted shots and a 95% Wilson
lower confidence bound of at least 0.9. It is a project acceptance criterion,
not a universal physical-device standard.

Native ideal correctness and noisy quality are separate. The CLI normally exits
on native-correctness failure so that CI can retain expected noisy failures as
research evidence. `--require-noisy-quality` also fails on noisy quality. A green
CI job never means that all hardware-quality gates passed; the summary prints
that decision explicitly.

## Current evidence and limits

The compact numerical study passes all four existing matrix families through
1000 dimensions, with worst ideal input residual below 3e-7. Hardware-oriented
native checks cover the dense SPD fixture through 100 dimensions.

For the assumed line layout, dense 100×100 still requires about 14625 CX gates
and native depth 41024, despite using only 11 qubits. Its optimistic synthetic
run returns the reference in 8 of 34 accepted shots (64 total): about 24%, with
95% interval roughly 12–40%. It fails the provisional quality gate. 10×10 passes
the optimistic model but fails the stress model. No result supports perfect
hardware accuracy or hardware qualification of the dense 100×100 circuit.

These measurements justify prioritizing small hardware tests and reducing dense
encoding depth, using structured matrix input, or applying an appropriate
preconditioner before claiming practical large-instance performance. They do not
establish speed-up over classical inversion.

## An actual device

The current workflow submits no physical QPU jobs. To evaluate a real backend,
provide its name, native basis, coupling map, calibration and provider interface.
Regenerate the original circuit and use `compile_native(..., backend=backend)`;
do not assume the exported generic line layout matches the actual device.
Use provider calibration-derived Aer noise for device-faithful simulation.

`run_verification(..., backend=backend)` executes only when explicitly called
and requires Backend.run support. Providers exposing runtime primitives instead
need a provider-specific adapter. Record calibration, routing, job IDs, shot
counts and compilation settings for every physical run. The 42-qubit count alone
cannot substitute for those checks. Hardware calibration and physical execution
remain pending, including for 1000×1000.
