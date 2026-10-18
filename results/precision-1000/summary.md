# 1000×1000 coherent inverse precision study

Numerical validation: PASS

Polynomial truncation target 1e-12; numerical operator-residual gate 1e-10. These are separate: floating-point encoding and gate simulation errors are not covered by the truncation bound.

| Family | Condition | Degree | Qubits | Operator residual 2-norm | Relative inverse error F | Min success (all inputs) |
|---|---:|---:|---:|---:|---:|---:|
| dense_spd | 1.67 | 13 | 15 | 6.822e-13 | 4.215e-13 | 3.588e-01 |
| tridiagonal_spd | 1.97 | 16 | 16 | 1.522e-13 | 9.063e-14 | 2.590e-01 |
| dense_nonsymmetric | 1.67 | 13 | 15 | 7.285e-13 | 4.503e-13 | 3.571e-01 |
| signed_indefinite | 1.67 | 13 | 15 | 6.786e-13 | 4.197e-13 | 3.590e-01 |
| controlled_spd | 10 | 46 | 17 | 1.789e-13 | 5.411e-14 | 1.000e-02 |
| controlled_nonsymmetric | 10 | 46 | 17 | 1.783e-13 | 6.258e-14 | 1.000e-02 |
| controlled_indefinite | 10 | 46 | 17 | 1.787e-13 | 6.232e-14 | 1.000e-02 |
| controlled_spd | 100 | 161 | 19 | 1.027e-13 | 1.489e-14 | 1.000e-04 |
| controlled_nonsymmetric | 100 | 161 | 19 | 9.957e-14 | 4.347e-14 | 1.000e-04 |
| controlled_indefinite | 100 | 161 | 19 | 9.845e-14 | 4.357e-14 | 1.000e-04 |
| controlled_spd | 1e+03 | 546 | 21 | 4.662e-13 | 7.008e-14 | 1.000e-06 |
| controlled_nonsymmetric | 1e+03 | 546 | 21 | 4.561e-13 | 1.503e-13 | 1.000e-06 |
| controlled_indefinite | 1e+03 | 546 | 21 | 4.447e-13 | 1.502e-13 | 1.000e-06 |

On the four earlier fixture families, the same-input worst residual improves by 2.31e+05–1.62e+06 relative to the 1e-6 polynomial target.

The full-operator audit reconstructs the simulated success operator and checks ||M X - I||_2, which bounds ||M X b - b||_2 / ||b||_2 for every input b of each tested matrix. It uses exact simulator amplitudes, not measured inverse readout.

M M.T > 0 guarantees invertibility for a real square M, but neither a uniform degree budget nor uniform success probability. The 1000×1000 condition-10000 input is explicitly resource-limited: Requested accuracy needs degree 1842; budget is 1023

Classical dense spectral/SVD encoding remains required. This is not native noisy 1000-dimensional execution, a hardware result, or evidence of quantum speed-up.
