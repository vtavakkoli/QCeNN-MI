# Complete QCeNN-MI validation

Run ID: `qcenn-b1c07881e9ed4223ba824b4d2b969965`; profile: `full`.

Detailed evidence: [final-report.html](final-report.html), [final-report.md](final-report.md), and [manifest](publication-evidence/manifest.json).

Software validation: PASS

Source: `6a745ae83ae17163bf4d86d4d622c6c63305c19e`; tree SHA-256: `36404cf7a25b54a7d6bc3b3009a174754bea9a00d73f49c6083c274fbc397ff0`.

| Stage | Result | Wall seconds | CPU seconds | Peak child MiB |
|---|---|---:|---:|---:|
| runtime-smoke | PASS | 0.38 | 0.38 | 88.9 |
| compile | PASS | 0.12 | 0.12 | 14.8 |
| lint | PASS | 0.04 | 0.08 | 22.0 |
| unit-tests | PASS | 93.09 | 93.33 | 346.7 |
| assumption-a1 | PASS | 0.99 | 0.99 | 124.6 |
| a2-a4 | PASS | 19.12 | 19.24 | 139.9 |
| scaling | PASS | 31.23 | 31.22 | 130.7 |
| coherent | PASS | 20.63 | 21.01 | 473.3 |
| coherent-large | PASS | 6.71 | 6.82 | 344.7 |
| compact | PASS | 1.91 | 1.91 | 207.6 |
| precision-1000 | PASS | 21.17 | 21.15 | 248.8 |
| theory | PASS | 1.07 | 1.07 | 107.5 |
| time-varying | PASS | 23.13 | 23.13 | 327.3 |
| hardware | PASS | 40.82 | 42.19 | 510.1 |
| temporal-refresh | PASS | 4.11 | 4.12 | 164.1 |
| noise-verification | PASS | 15.40 | 15.41 | 162.8 |
| baselines | PASS | 22.37 | 22.35 | 227.3 |
| resources | PASS | 3.13 | 3.22 | 255.2 |
| operator-audit | PASS | 7.01 | 7.15 | 161.8 |
| recurrent-quantum | PASS | 18.34 | 18.34 | 201.9 |
| recurrent-scaling | PASS | 153.17 | 370.62 | 244.2 |
| dissipative-comparison | PASS | 346.24 | 346.17 | 2453.5 |
| polynomial-recurrent | PASS | 289.61 | 290.19 | 485.1 |
| polynomial-composition | PASS | 99.11 | 99.61 | 321.2 |
| final-report | PASS | 7.30 | 7.29 | 353.0 |

All synthetic noisy quality gates: False

A software PASS does not certify noisy quality or physical execution.
No EuroHPC allocated-node run or physical QPU run is claimed by this artifact.
