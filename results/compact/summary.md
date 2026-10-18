# Compact coherent inverse improvements

Numerical validation: PASS

| Size | Family | Degree | Total qubits | Old qubits | Worst residual | Min success |
|---:|---|---:|---:|---:|---:|---:|
| 10 | dense_spd | 7 | 8 | 14 | 1.276e-07 | 60.2% |
| 10 | tridiagonal_spd | 7 | 8 | 14 | 2.531e-07 | 49.4% |
| 10 | dense_nonsymmetric | 7 | 8 | 14 | 1.171e-07 | 61.6% |
| 10 | signed_indefinite | 7 | 8 | 14 | 1.144e-07 | 60.7% |
| 100 | dense_spd | 7 | 11 | 17 | 1.215e-07 | 60.0% |
| 100 | tridiagonal_spd | 8 | 12 | 17 | 1.088e-07 | 43.0% |
| 100 | dense_nonsymmetric | 7 | 11 | 17 | 1.242e-07 | 59.8% |
| 100 | signed_indefinite | 7 | 11 | 17 | 1.182e-07 | 60.4% |
| 1000 | dense_spd | 7 | 14 | 20 | 1.028e-07 | 58.3% |
| 1000 | tridiagonal_spd | 8 | 15 | 20 | 1.586e-07 | 41.2% |
| 1000 | dense_nonsymmetric | 7 | 14 | 20 | 1.101e-07 | 58.1% |
| 1000 | signed_indefinite | 7 | 14 | 20 | 1.179e-07 | 58.3% |

Condition-number-100 case: degree 92, 12 qubits, residual 8.638e-09 (previously about 0.852).

These are ideal structured-statevector results; native noise is reported separately.
Classical spectral/SVD encoding is still required; quantum advantage is not established.
