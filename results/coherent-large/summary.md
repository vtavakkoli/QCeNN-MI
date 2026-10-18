# Large standalone coherent inverse-action study

Status: PASS

Exact structured statevector simulation, including all selector and flag ancillas.
Classical spectral input encoding is required. Results do not establish universal convergence.

| Size | Family | Padded | Qubits | RHS residual | Min success | Inverse bound |
|---:|---|---:|---:|---:|---:|---:|
| 10 | dense_spd | 16 | 14 | 1.617e-05 | 1.68% | 5.118e-05 |
| 10 | tridiagonal_spd | 16 | 14 | 2.687e-05 | 1.46% | 1.003e-04 |
| 10 | dense_nonsymmetric | 16 | 14 | 1.485e-05 | 1.71% | 4.707e-05 |
| 10 | signed_indefinite | 16 | 14 | 1.453e-05 | 1.68% | 4.597e-05 |
| 100 | dense_spd | 128 | 17 | 1.553e-05 | 1.67% | 1.573e-04 |
| 100 | tridiagonal_spd | 128 | 17 | 3.892e-05 | 1.42% | 8.920e-04 |
| 100 | dense_nonsymmetric | 128 | 17 | 1.575e-05 | 1.67% | 1.595e-04 |
| 100 | signed_indefinite | 128 | 17 | 1.501e-05 | 1.68% | 1.506e-04 |
| 1000 | dense_spd | 1024 | 20 | 1.506e-05 | 1.62% | 4.944e-04 |
| 1000 | tridiagonal_spd | 1024 | 20 | 4.076e-05 | 1.42% | 3.982e-03 |
| 1000 | dense_nonsymmetric | 1024 | 20 | 1.570e-05 | 1.62% | 5.121e-04 |
| 1000 | signed_indefinite | 1024 | 20 | 1.546e-05 | 1.62% | 4.928e-04 |

The full-inverse column is an analytical bound, not full-matrix tomography.
Z sampling does not reconstruct signed inverse entries.

## Native Aer crosschecks

- 10×10, degree 3: agreement error 5.173e-15; 2187 CX gates.
- 100×100, degree 1: agreement error 4.514e-14; 15735 CX gates.

Counterexample: condition number 100, degree 7 gives residual 0.852.
Higher degree, better encoding/preconditioning, or another method is required.
1000×1000 native-gate Aer, native-gate noise and physical hardware are not validated.
The classical eigendecomposition/SVD and dense encoding prevent any claim of quantum advantage.
