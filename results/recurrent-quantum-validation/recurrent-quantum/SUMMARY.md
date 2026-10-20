# Recurrent quantum circuit experiment

Software/numerical verdict: PASS.
Physical hardware: not run. Synthetic CX-only noise is reported separately below.

Every update executes in one quantum circuit. The coherent circuit retains its
environment; the reset circuit reuses it on device. Neither reads or reconstructs
the data register between iterations. Classical solves are independent validation only.

| Size | Family | Updates | Augmented fidelity | Conditional solution fidelity |
|---|---|---|---|---|
| 2 | controlled_spd | 64 | 0.9999998891 | 0.9999998204 |
| 2 | controlled_nonsymmetric | 64 | 0.9999998891 | 0.9999998204 |
| 2 | controlled_indefinite | 64 | 0.9999998891 | 0.9999998204 |
| 3 | controlled_spd | 64 | 0.9999998307 | 0.9999998405 |
| 3 | controlled_nonsymmetric | 64 | 0.9999998307 | 0.9999998405 |
| 3 | controlled_indefinite | 64 | 0.9999998307 | 0.9999998405 |
| 4 | controlled_spd | 64 | 0.9999999733 | 0.9999999627 |
| 4 | controlled_nonsymmetric | 64 | 0.9999999733 | 0.9999999627 |
| 4 | controlled_indefinite | 64 | 0.9999999733 | 0.9999999627 |
| 10 | controlled_spd | 64 | 0.9999998671 | 0.9999998513 |
| 10 | controlled_nonsymmetric | 64 | 0.9999998671 | 0.9999998513 |
| 10 | controlled_indefinite | 64 | 0.9999998671 | 0.9999998513 |

## Native circuit and synthetic noise

Order 2, 32 updates: 5 qubits, 1088 CX gates, depth 3456.
Final ideal Z readout: 1487/4096 outcomes in the solution subspace. Signed inverse entries are not reconstructed.

| CX depolarizing probability | Conditional solution fidelity | 0.99 quality gate |
|---|---|---|
| 1e-05 | 0.99927485 | True |
| 0.0001 | 0.99447178 | True |

These are CX-only synthetic channels with ideal resets, 1-qubit gates and readout.
Their quality verdict does not change the software implementation verdict.

## Conditioning stress probes

| Condition number | Updates | Augmented fidelity | Meets accuracy target |
|---|---|---|---|
| 10 | 64 | 0.83410416 | False |
| 100 | 64 | 0.68723814 | False |

Slow convergence at higher conditioning is retained as an accuracy limitation.

## Limits

- Dense classical gate compilation, not an efficient quantum input oracle or speedup.
- RHS-dependent solver state; not the existing input-independent inverse operator.
- Separate residual-channel prototype; no Chebyshev or QSVT acceleration implemented.
- Coherent workspace grows with iteration count; reset mode discards device ancillas.
- Convergence certificates concern ideal static gates, not arbitrary hardware noise.
- Order-100/1000 entries are width estimates, not simulation or hardware execution.
- Final Z counts do not recover signed amplitudes or a complete classical inverse.
