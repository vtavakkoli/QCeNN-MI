# Polynomial recurrent-channel validation

These are actual records from the full 25-stage profile, run
`qcenn-b1c07881e9ed4223ba824b4d2b969965`. All stages passed numerical-integrity
validation; 267 unit tests passed. Accuracy failures remain failures.

The executed source is `6a745ae83ae17163bf4d86d4d622c6c63305c19e`.
The public counterpart `4e22cc07f7bb4fa6a4be1f91cfd4d73423832c2f` has the
independently verified identical Git tree
`35f668eb38284ec9ad5cd9e1de0043e48b07c1d2`. Commit metadata differs. The portable
source fingerprint is
`36404cf7a25b54a7d6bc3b3009a174754bea9a00d73f49c6083c274fbc397ff0`.

The compressed JSON records preserve all static/dynamic trajectories and their
actual input checksums. The complete validation archive additionally includes
raw NPZ input bytes, QPY circuits, every stage's raw results, logs and checksums;
generate it from the pinned source with `python docker/reproduce.py --profile full`.
`summary.md` reports stage timings and resource observations. The JSON gzip
files are ordinary gzip streams, not Git LFS pointers.

| Study | Outcome |
| --- | --- |
| Static grid, N=2,4,10,100,1000; conditions 2,10,100,1000 | 770/1416 meet the joint sampled 1e-6 criterion |
| Condition 1000 | No accuracy pass within 16384 macro updates |
| Changing spectra/RHS/jumps/final hold | 9/9 ideal polynomial trajectories recover after the hold; noisy floors remain |
| Logical high-conditioning noise | None of 72 transient or stationary residuals reaches 1e-2 |
| Dense transformed dilation | 72 small native compilations; 48 large resource rejections |
| Independent two-update complete circuits | 72 comparisons, maximum density discrepancy below 1e-14 |
| Controlled baseline-unitary composition | 22 completed, 38 resource rejections, maximum discrepancy below 5.7e-13 |
| Separate equal-call audit | 9/18 pass; no residual improvement at condition 100 under the same primitive-call budget |

The fixed-call audit executes the pinned implementation and separately hashed
script. Its selected input bytes were independently checked equal to the final
main campaign's inputs. It is a distinct audit, not an extra main pipeline stage.

Large cases simulate a reduced quantum channel using classical spectral/renewal
methods. Native counts and gate-noise tests cover only circuits actually built.
The dense compiler uses classical spectral preprocessing; the composed circuit
charges its extra controls and residual calls. No physical hardware execution,
calibrated noise threshold or quantum runtime speedup is claimed. A Docker
executable was unavailable in the local workspace; the full process-based
profile and startup checks were executed, and the public implementation CI
passed.

The manuscript remains outside this repository. Its historical unknown-source
run and seven previously claimed recovered payloads are not certified by these
new records. A new pinned run does not establish an older run's source identity.
