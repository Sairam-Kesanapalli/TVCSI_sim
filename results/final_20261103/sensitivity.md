# Sensitivity of the constraint metrics (default point)

The constraint parameters of P1 (Eq. 39) have no single correct value. Defaults are taken from 3GPP where a standard exists (ASSUMPTIONS.md items 39-41); this file re-checks every conclusion over a range of values. Mean ± 95% t-interval over the test seeds.

## eps (Eq. 39j): share of seeds where every critical user meets the tolerance

Evaluated over each critical user's feasible slots. Default eps = 0.1 (3GPP TS 38.214 5.2.2.1 CQI target BLER). The URLLC target 1e-5 (TR 38.913 7.9) is not resolvable here: each critical user has about 1900 slots per seed, of which only the feasible ones count.

| scheme | eps = 0.01 | eps = 0.05 | eps = 0.1 |
|---|---|---|---|
| Exact | 0.1 | 1.0 | 1.0 |
| Proposed | 0.0 | 0.1 | 1.0 |
| Optimised | 0.0 | 0.1 | 1.0 |
| Baseline | 0.0 | 0.0 | 0.1 |
| importance_only | 0.0 | 0.0 | 0.0 |
| snr_threshold | 0.0 | 0.1 | 0.1 |
| prediction_only | 0.0 | 0.0 | 0.0 |
| BestBeam_oracle | 1.0 | 1.0 | 1.0 |

## lambda (Eq. 39a): P1 objective, one weight varied at a time

Defaults: lambda_F = 0.001, lambda_B = 0.05, lambda_R = 0.1. No standard value exists for these weights.

**lambda_F**

| scheme | 0 | 0.0005 | 0.001 | 0.002 |
|---|---|---|---|---|
| Exact | 2.702 ± 0.030 | 2.692 ± 0.031 | 2.682 ± 0.031 | 2.662 ± 0.032 |
| Proposed | 2.576 ± 0.038 | 2.448 ± 0.038 | 2.320 ± 0.038 | 2.064 ± 0.038 |
| Optimised | 2.573 ± 0.038 | 2.445 ± 0.038 | 2.317 ± 0.038 | 2.061 ± 0.038 |
| Baseline | 2.510 ± 0.042 | 2.382 ± 0.042 | 2.254 ± 0.042 | 1.998 ± 0.042 |
| importance_only | 2.504 ± 0.040 | 2.376 ± 0.040 | 2.248 ± 0.040 | 1.992 ± 0.040 |
| snr_threshold | 1.822 ± 0.150 | 1.694 ± 0.150 | 1.566 ± 0.150 | 1.310 ± 0.150 |
| prediction_only | 0.908 ± 0.216 | 0.908 ± 0.216 | 0.908 ± 0.216 | 0.908 ± 0.216 |

**lambda_B**

| scheme | 0 | 0.05 | 0.1 |
|---|---|---|---|
| Exact | 2.708 ± 0.030 | 2.682 ± 0.031 | 2.656 ± 0.031 |
| Proposed | 2.399 ± 0.035 | 2.320 ± 0.038 | 2.240 ± 0.040 |
| Optimised | 2.397 ± 0.035 | 2.317 ± 0.038 | 2.238 ± 0.040 |
| Baseline | 2.344 ± 0.040 | 2.254 ± 0.042 | 2.165 ± 0.045 |
| importance_only | 2.293 ± 0.039 | 2.248 ± 0.040 | 2.203 ± 0.042 |
| snr_threshold | 1.640 ± 0.148 | 1.566 ± 0.150 | 1.491 ± 0.152 |
| prediction_only | 0.908 ± 0.216 | 0.908 ± 0.216 | 0.908 ± 0.216 |

**lambda_R**

| scheme | 0 | 0.1 | 0.2 |
|---|---|---|---|
| Exact | 2.787 ± 0.030 | 2.682 ± 0.031 | 2.578 ± 0.032 |
| Proposed | 2.439 ± 0.035 | 2.320 ± 0.038 | 2.201 ± 0.041 |
| Optimised | 2.437 ± 0.035 | 2.317 ± 0.038 | 2.198 ± 0.041 |
| Baseline | 2.383 ± 0.038 | 2.254 ± 0.042 | 2.125 ± 0.046 |
| importance_only | 2.382 ± 0.037 | 2.248 ± 0.040 | 2.113 ± 0.043 |
| snr_threshold | 1.798 ± 0.134 | 1.566 ± 0.150 | 1.333 ± 0.166 |
| prediction_only | 1.250 ± 0.191 | 0.908 ± 0.216 | 0.566 ± 0.241 |

### Full grid

Over all 36 combinations of the three weights: Proposed beats Baseline in mean in **36 of 36**; it is significantly worse (95% interval below zero) in **0 of 36**.

Best implementable scheme (highest mean objective) per combination: Proposed 36.

## Payload size (Eq. 39i): every payload multiplied by a scale factor

Base sizes (TS 22.185): critical 1200 B, normal 300 B, background 50 B; 270 PRBs shared by K = 8 users. The scale applies to the Shannon check of Eq. 39i only; the measured Q curves keep the base sizes (a different payload would be a different measured curve).

| scale | infeasible share (oracle) | Exact miss (feasible) | Optimised miss (feasible) | Proposed miss (feasible) | Baseline miss (feasible) |
|---|---|---|---|---|---|
| 0.25 | 0.000 ± 0.000 | 0.002 ± 0.000 | 0.005 ± 0.001 | 0.005 ± 0.001 | 0.007 ± 0.001 |
| 0.5 | 0.000 ± 0.000 | 0.004 ± 0.001 | 0.007 ± 0.001 | 0.007 ± 0.001 | 0.011 ± 0.001 |
| 1 | 0.003 ± 0.000 | 0.009 ± 0.001 | 0.013 ± 0.002 | 0.012 ± 0.002 | 0.020 ± 0.002 |
| 2 | 0.066 ± 0.005 | 0.010 ± 0.001 | 0.026 ± 0.003 | 0.026 ± 0.003 | 0.035 ± 0.004 |
