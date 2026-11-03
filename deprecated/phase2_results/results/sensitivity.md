# Sensitivity of the constraint metrics (default point)

The constraint parameters of P1 (Eq. 39) have no single correct value. Defaults are taken from 3GPP where a standard exists (ASSUMPTIONS.md items 39-41); this file re-checks every conclusion over a range of values. Mean ± 95% t-interval over the test seeds.

## eps (Eq. 39j): share of seeds where every critical user meets the tolerance

Evaluated over each critical user's feasible slots. Default eps = 0.1 (3GPP TS 38.214 5.2.2.1 CQI target BLER). The URLLC target 1e-5 (TR 38.913 7.9) is not resolvable here: each critical user has about 1900 slots per seed, of which only the feasible ones count.

| scheme | eps = 0.01 | eps = 0.05 | eps = 0.1 |
|---|---|---|---|
| Exact | 0.1 | 1.0 | 1.0 |
| Proposed | 0.0 | 0.0 | 0.0 |
| Optimised | 0.0 | 0.0 | 0.0 |
| Baseline | 0.0 | 0.0 | 0.0 |
| importance_only | 0.0 | 0.0 | 0.0 |
| snr_threshold | 0.0 | 0.0 | 0.0 |
| prediction_only | 0.0 | 0.0 | 0.0 |
| BestBeam_oracle | 1.0 | 1.0 | 1.0 |

## lambda (Eq. 39a): P1 objective, one weight varied at a time

Defaults: lambda_F = 0.001, lambda_B = 0.05, lambda_R = 0.1. No standard value exists for these weights.

**lambda_F**

| scheme | 0 | 0.0005 | 0.001 | 0.002 |
|---|---|---|---|---|
| Exact | 2.702 ± 0.030 | 2.691 ± 0.030 | 2.680 ± 0.030 | 2.658 ± 0.031 |
| Proposed | 0.908 ± 0.216 | 0.908 ± 0.216 | 0.908 ± 0.216 | 0.908 ± 0.216 |
| Optimised | 0.908 ± 0.216 | 0.908 ± 0.216 | 0.908 ± 0.216 | 0.908 ± 0.216 |
| Baseline | 2.501 ± 0.032 | 2.373 ± 0.032 | 2.245 ± 0.032 | 1.989 ± 0.032 |
| importance_only | 2.521 ± 0.037 | 2.393 ± 0.037 | 2.265 ± 0.037 | 2.009 ± 0.037 |
| snr_threshold | 1.939 ± 0.103 | 1.811 ± 0.103 | 1.683 ± 0.103 | 1.427 ± 0.103 |
| prediction_only | 0.908 ± 0.216 | 0.908 ± 0.216 | 0.908 ± 0.216 | 0.908 ± 0.216 |

**lambda_B**

| scheme | 0 | 0.05 | 0.1 |
|---|---|---|---|
| Exact | 2.706 ± 0.030 | 2.680 ± 0.030 | 2.654 ± 0.031 |
| Proposed | 0.908 ± 0.216 | 0.908 ± 0.216 | 0.908 ± 0.216 |
| Optimised | 0.908 ± 0.216 | 0.908 ± 0.216 | 0.908 ± 0.216 |
| Baseline | 2.341 ± 0.029 | 2.245 ± 0.032 | 2.150 ± 0.034 |
| importance_only | 2.309 ± 0.036 | 2.265 ± 0.037 | 2.221 ± 0.039 |
| snr_threshold | 1.769 ± 0.101 | 1.683 ± 0.103 | 1.597 ± 0.106 |
| prediction_only | 0.908 ± 0.216 | 0.908 ± 0.216 | 0.908 ± 0.216 |

**lambda_R**

| scheme | 0 | 0.1 | 0.2 |
|---|---|---|---|
| Exact | 2.785 ± 0.030 | 2.680 ± 0.030 | 2.575 ± 0.031 |
| Proposed | 1.250 ± 0.191 | 0.908 ± 0.216 | 0.566 ± 0.241 |
| Optimised | 1.250 ± 0.191 | 0.908 ± 0.216 | 0.566 ± 0.241 |
| Baseline | 2.374 ± 0.030 | 2.245 ± 0.032 | 2.116 ± 0.034 |
| importance_only | 2.396 ± 0.035 | 2.265 ± 0.037 | 2.133 ± 0.040 |
| snr_threshold | 1.898 ± 0.090 | 1.683 ± 0.103 | 1.468 ± 0.117 |
| prediction_only | 1.250 ± 0.191 | 0.908 ± 0.216 | 0.566 ± 0.241 |

### Full grid

Over all 36 combinations of the three weights: Proposed beats Baseline in mean in **0 of 36**; it is significantly worse (95% interval below zero) in **36 of 36**.

Best implementable scheme (highest mean objective) per combination: importance_only 24, Baseline 12.

## Payload size (Eq. 39i): every payload multiplied by a scale factor

Base sizes (TS 22.185): critical 1200 B, normal 300 B, background 50 B; 270 PRBs shared by K = 8 users. The scale applies to the Shannon check of Eq. 39i only; the measured Q curves keep the base sizes (a different payload would be a different measured curve).

| scale | infeasible share (oracle) | Exact miss (feasible) | Optimised miss (feasible) | Proposed miss (feasible) | Baseline miss (feasible) |
|---|---|---|---|---|---|
| 0.25 | 0.000 ± 0.000 | 0.002 ± 0.000 | 0.281 ± 0.072 | 0.281 ± 0.072 | 0.009 ± 0.001 |
| 0.5 | 0.000 ± 0.000 | 0.004 ± 0.001 | 0.360 ± 0.076 | 0.360 ± 0.076 | 0.012 ± 0.001 |
| 1 | 0.003 ± 0.000 | 0.009 ± 0.001 | 0.442 ± 0.075 | 0.442 ± 0.075 | 0.020 ± 0.001 |
| 2 | 0.066 ± 0.005 | 0.010 ± 0.001 | 0.515 ± 0.073 | 0.515 ± 0.073 | 0.037 ± 0.002 |
