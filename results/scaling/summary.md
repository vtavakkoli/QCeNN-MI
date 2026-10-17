# Larger matrix and dynamic simulator checks

Status: PASS

Large cases use quantum 2x2 blocks and classical full-matrix correction.

| Size | Case | Seed shots | Initial residual | Final residual | Relative error |
|---:|---|---:|---:|---:|---:|
| 10 | symmetric | ideal | 6.643e-02 | 1.513e-13 | 4.885e-14 |
| 10 | symmetric | 4096 | 7.362e-02 | 1.269e-13 | 4.145e-14 |
| 10 | nonsymmetric | ideal | 9.858e-02 | 1.673e-13 | 5.529e-14 |
| 10 | nonsymmetric | 4096 | 1.038e-01 | 4.832e-13 | 1.626e-13 |
| 100 | symmetric | ideal | 2.922e-01 | 9.537e-12 | 9.943e-13 |
| 100 | symmetric | 4096 | 3.013e-01 | 1.067e-11 | 1.107e-12 |
| 100 | nonsymmetric | ideal | 4.113e-01 | 2.751e-12 | 2.700e-13 |
| 100 | nonsymmetric | 4096 | 4.177e-01 | 3.111e-12 | 3.074e-13 |

Dynamic 2x2: 21 time points; maximum quantum residual 8.236e-09; maximum shot residual 2.581e-02 (4096 shots per setting).

No physical hardware test or large amplitude-encoded quantum solve is claimed.
