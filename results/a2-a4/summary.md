# A2–A4 simulator campaign

Status: PASS

Synthetic noise; no physical hardware execution. Safeguarded corrections are classical.

| Case | Layers | Ideal residual | Shot residual | Noisy residual | Corrected |
|---|---:|---:|---:|---:|---:|
| spd2 | 1 | 1.606e-15 | 7.094e-03 | 7.094e-03 | 1.487e-07 |
| spd2 | 2 | 8.512e-13 | 7.094e-03 | 7.094e-03 | 1.487e-07 |
| spd2 | 4 | 1.698e-12 | 7.094e-03 | 7.094e-03 | 1.487e-07 |
| nonsymmetric4 | 1 | 5.349e-16 | 1.636e-02 | 1.586e-02 | 1.274e-06 |
| nonsymmetric4 | 2 | 5.569e-13 | 1.636e-02 | 1.649e-02 | 1.389e-06 |
| nonsymmetric4 | 4 | 5.718e-13 | 1.636e-02 | 1.718e-02 | 1.645e-06 |
| signed2 | 1 | 1.465e-16 | 1.159e-02 | 1.152e-02 | 5.300e-07 |
| signed2 | 2 | 1.013e-12 | 1.159e-02 | 1.152e-02 | 5.300e-07 |
| signed2 | 4 | 2.159e-16 | 1.159e-02 | 1.152e-02 | 5.300e-07 |

## A4 dense quantum polynomial dilation

| Case | Initial residual | Ideal operator residual | Postselected shot residual |
|---|---:|---:|---:|
| spd2 | 7.869e-01 | 3.571e-01 | 2.762e-01 |
| nonsymmetric4 | 8.586e-01 | 1.597e-01 | 8.936e-02 |
| signed2 | 5.802e-01 | 9.000e-02 | 3.178e-02 |

Tracking samples: 15; maximum corrected residual: 3.359e-16.

QPY files contain state preparation circuits. Add the X/Y/Z measurement settings
using readout_direction(), transpile for the selected backend and record its calibration.
Full tomography uses 3**qubits settings per column and is not a scalable readout.
