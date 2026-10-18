# Standalone coherent inverse action

Status: PASS

Coherent Neumann-LCU circuit; no classical fitting or residual correction.
Classical dense block-encoding construction remains required. Ideal and finite-shot noiseless simulation only.

| Case | Qubits | Ideal residual | Shot residual | Min success | Max CX |
|---|---:|---:|---:|---:|---:|
| spd2 | 11 | 8.695e-06 | 2.281e-02 | 1.32% | 6608 |
| spd4 | 12 | 4.411e-07 | 3.645e-02 | 1.30% | 28113 |

Degree convergence:

| Case | Degree | Residual | Bound |
|---|---:|---:|---:|
| spd2 | 1 | 7.042e-02 | 1.397e-01 |
| spd2 | 3 | 3.507e-03 | 1.380e-02 |
| spd2 | 7 | 8.695e-06 | 1.346e-04 |
| spd4 | 1 | 3.347e-02 | 8.864e-02 |
| spd4 | 3 | 7.898e-04 | 3.929e-03 |
| spd4 | 7 | 4.411e-07 | 7.718e-06 |

Measurement scales come from ancilla success frequency, not least-squares refitting.
Raw counts and QPY files are generated alongside this report and uploaded by CI.
High ancilla/gate counts and low postselection probability limit hardware practicality.
This is a coherent inverse-action baseline; quantum advantage and local QCeNN dynamics are not established.
