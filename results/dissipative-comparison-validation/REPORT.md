# Matched dissipative comparison and variable-spectrum tracking

All results are reduced-density simulations. No physical gate noise was tested.
Shang's generator is divided by two to bound the total jump rate by one.
Model time and update counts are not hardware/query costs.

| N | condition | method | residual | first sustained sampled time |
|---:|---:|---|---:|---:|
| 10 | 2 | discrete | 4.96962e-16 | 108 |
| 10 | 2 | matched_continuous | 1.15387e-12 | 128 |
| 10 | 2 | shang_rate_normalized | 2.17074e-12 | 176 |
| 10 | 10 | discrete | 2.63712e-12 | 2176 |
| 10 | 10 | matched_continuous | 6.3967e-12 | 2176 |
| 10 | 10 | shang_rate_normalized | 2.31107e-08 | 3328 |
| 10 | 100 | discrete | 0.482455 | None |
| 10 | 100 | matched_continuous | 0.482465 | None |
| 10 | 100 | shang_rate_normalized | 0.863624 | None |
| 100 | 2 | discrete | 2.27371e-15 | 108 |
| 100 | 2 | matched_continuous | 1.06238e-10 | 128 |
| 100 | 2 | shang_rate_normalized | 6.05222e-11 | 192 |
| 100 | 10 | discrete | 5.09675e-11 | 2304 |
| 100 | 10 | matched_continuous | 1.61369e-10 | 2304 |
| 100 | 10 | shang_rate_normalized | 9.57398e-07 | 4096 |
| 100 | 100 | discrete | 0.249842 | None |
| 100 | 100 | matched_continuous | 0.249861 | None |
| 100 | 100 | shang_rate_normalized | 0.862527 | None |
| 1000 | 2 | discrete | 3.26497e-15 | 108 |
| 1000 | 2 | matched_continuous | 4.83032e-10 | 128 |
| 1000 | 2 | shang_rate_normalized | 1.15753e-10 | 176 |
| 1000 | 10 | discrete | 5.48828e-11 | 2304 |
| 1000 | 10 | matched_continuous | 1.81628e-09 | 2304 |
| 1000 | 10 | shang_rate_normalized | 1.09072e-06 | None |
| 1000 | 100 | discrete | 0.267682 | None |
| 1000 | 100 | matched_continuous | 0.267707 | None |
| 1000 | 100 | shang_rate_normalized | 0.896071 | None |

Condition-100 transient error and exact logical-noise stationary error are
reported separately in condition_100_noise.csv. Longer ideal N=10 evolution
is a small-superoperator computation, not a native quantum execution.

Dynamic matrices change every singular value, norm and condition number.
The same density state survives every snapshot; only its simulator basis changes.
