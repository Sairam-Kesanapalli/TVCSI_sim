# TVCSI simulator — results summary

> **DUMMY MODELS.** Every number below comes from the placeholder encoder/decoder (log-normal decoding noise) and the persistence beam predictor. They test the pipeline and the schedulers; they are **not research results** about learned semantic CSI feedback.

Default configuration: K=8, M=B=32, speed=30 km/h, SNR=15 dB, crit_share=0.25, budget_frac=0.25 (C_FB_max=256 bits/slot), T=2000 slots (first 100 excluded), 10 test seeds. Values are mean ± 95% t-interval half-width over seeds.

## Default configuration, all schemes

| scheme | utility | regret | CVaR0.95 | crit. violation | bits/slot | switch rate |
|---|---|---|---|---|---|---|
| Exact | 1.751 ± 0.035 | 0.017 ± 0.001 | 2.551 ± 0.054 | 0.957 ± 0.013 | 98.8 ± 3.5 | 0.243 ± 0.008 |
| Optimised | 1.315 ± 0.081 | 0.339 ± 0.024 | 3.231 ± 0.128 | 0.983 ± 0.004 | 256.0 ± 0.0 | 0.159 ± 0.006 |
| Proposed | 1.367 ± 0.050 | 0.311 ± 0.018 | 3.184 ± 0.099 | 0.978 ± 0.006 | 256.0 ± 0.0 | 0.164 ± 0.007 |
| Baseline | 1.601 ± 0.039 | 0.129 ± 0.005 | 2.825 ± 0.066 | 0.960 ± 0.012 | 256.0 ± 0.0 | 0.238 ± 0.008 |
| prediction_only | 0.506 ± 0.132 | 0.721 ± 0.060 | 4.298 ± 0.198 | 0.987 ± 0.016 | 0.0 ± 0.0 | 0.000 ± 0.000 |
| importance_only | 1.506 ± 0.046 | 0.246 ± 0.013 | 2.997 ± 0.071 | 0.962 ± 0.012 | 256.0 ± 0.0 | 0.111 ± 0.004 |
| snr_threshold | 1.176 ± 0.111 | 0.348 ± 0.034 | 3.425 ± 0.121 | 0.971 ± 0.013 | 256.0 ± 0.0 | 0.214 ± 0.007 |
| full_feedback | 1.723 ± 0.037 | 0.044 ± 0.002 | 2.606 ± 0.060 | 0.957 ± 0.013 | 1024.0 ± 0.0 | 0.351 ± 0.012 |
| Proposed_no_confidence | 1.046 ± 0.113 | 0.425 ± 0.039 | 3.638 ± 0.191 | 0.982 ± 0.005 | 256.0 ± 0.0 | 0.160 ± 0.010 |
| Proposed_no_importance | 1.167 ± 0.071 | 0.393 ± 0.028 | 3.456 ± 0.115 | 0.984 ± 0.006 | 256.0 ± 0.0 | 0.150 ± 0.007 |

`full_feedback` ignores the budget (reference only). Utility = mean over slots of sum_k U_k. Violation = fraction of (critical user, slot) pairs with Q < Q_min.

## Gap decomposition at the default point (paired by seed)

| pair | meaning | utility difference |
|---|---|---|
| Exact - Optimised | information gap | 0.436 ± 0.064 |
| Optimised - Proposed | greedy gap | -0.052 ± 0.063 |
| Proposed - Baseline | contribution | -0.234 ± 0.036 |

## Greedy vs DP

* Random instances (2000, K<=8, random budgets): mean greedy/DP value ratio **0.9829**.
* On the TVCSI estimates along the Proposed trajectory at the default point: mean ratio of estimated utility **0.99997** (± 0.00001 over seeds), worst slot 0.9942, greedy exactly optimal in 93.2% of slots.

## Expected ordering Exact >= Optimised >= Proposed >= Baseline

Checked at the default point and at every point of the five 1-D sweeps using the paired per-seed utility difference. A violation is listed when the mean difference is negative; 'significant' means the 95% interval excludes zero.

**The expected ordering is violated at 52 of 90 (point, pair) checks.**

| sweep | x | pair | mean difference | 95% CI excludes 0 |
|---|---|---|---|---|
| default | default | Optimised - Proposed | -0.052 ± 0.063 | no |
| default | default | Proposed - Baseline | -0.234 ± 0.036 | yes |
| budget | 0.05 | Optimised - Proposed | -0.077 ± 0.057 | yes |
| budget | 0.1 | Optimised - Proposed | -0.128 ± 0.040 | yes |
| budget | 0.2 | Optimised - Proposed | -0.030 ± 0.090 | no |
| budget | 0.2 | Proposed - Baseline | -0.273 ± 0.038 | yes |
| budget | 0.3 | Optimised - Proposed | -0.090 ± 0.031 | yes |
| budget | 0.3 | Proposed - Baseline | -0.013 ± 0.005 | yes |
| budget | 0.5 | Optimised - Proposed | -0.020 ± 0.030 | no |
| budget | 0.5 | Proposed - Baseline | -0.099 ± 0.017 | yes |
| budget | 0.75 | Optimised - Proposed | -0.007 ± 0.003 | yes |
| critical_share | 0 | Optimised - Proposed | -0.068 ± 0.059 | yes |
| critical_share | 0 | Proposed - Baseline | -0.271 ± 0.050 | yes |
| critical_share | 0.1 | Optimised - Proposed | -0.046 ± 0.074 | no |
| critical_share | 0.1 | Proposed - Baseline | -0.281 ± 0.073 | yes |
| critical_share | 0.25 | Optimised - Proposed | -0.052 ± 0.063 | no |
| critical_share | 0.25 | Proposed - Baseline | -0.234 ± 0.036 | yes |
| critical_share | 0.5 | Optimised - Proposed | -0.024 ± 0.038 | no |
| critical_share | 0.5 | Proposed - Baseline | -0.239 ± 0.038 | yes |
| critical_share | 0.75 | Optimised - Proposed | -0.007 ± 0.061 | no |
| critical_share | 0.75 | Proposed - Baseline | -0.272 ± 0.048 | yes |
| snr | 5 | Optimised - Proposed | -0.000 ± 0.001 | no |
| snr | 5 | Proposed - Baseline | -0.006 ± 0.001 | yes |
| snr | 10 | Proposed - Baseline | -0.089 ± 0.014 | yes |
| snr | 15 | Optimised - Proposed | -0.052 ± 0.063 | no |
| snr | 15 | Proposed - Baseline | -0.234 ± 0.036 | yes |
| snr | 20 | Optimised - Proposed | -0.075 ± 0.119 | no |
| snr | 20 | Proposed - Baseline | -0.400 ± 0.093 | yes |
| snr | 25 | Optimised - Proposed | -0.045 ± 0.081 | no |
| snr | 25 | Proposed - Baseline | -0.282 ± 0.081 | yes |
| speed | 3 | Optimised - Proposed | -0.022 ± 0.023 | no |
| speed | 10 | Optimised - Proposed | -0.036 ± 0.020 | yes |
| speed | 10 | Proposed - Baseline | -0.048 ± 0.027 | yes |
| speed | 20 | Optimised - Proposed | -0.036 ± 0.060 | no |
| speed | 20 | Proposed - Baseline | -0.177 ± 0.033 | yes |
| speed | 30 | Optimised - Proposed | -0.052 ± 0.063 | no |
| speed | 30 | Proposed - Baseline | -0.234 ± 0.036 | yes |
| speed | 40 | Optimised - Proposed | -0.032 ± 0.057 | no |
| speed | 40 | Proposed - Baseline | -0.324 ± 0.046 | yes |
| speed | 60 | Optimised - Proposed | -0.047 ± 0.094 | no |
| speed | 60 | Proposed - Baseline | -0.298 ± 0.088 | yes |
| users | 2 | Optimised - Proposed | -0.001 ± 0.004 | no |
| users | 2 | Proposed - Baseline | -0.023 ± 0.007 | yes |
| users | 4 | Proposed - Baseline | -0.094 ± 0.037 | yes |
| users | 6 | Optimised - Proposed | -0.044 ± 0.023 | yes |
| users | 6 | Proposed - Baseline | -0.136 ± 0.024 | yes |
| users | 8 | Optimised - Proposed | -0.052 ± 0.063 | no |
| users | 8 | Proposed - Baseline | -0.234 ± 0.036 | yes |
| users | 12 | Optimised - Proposed | -0.062 ± 0.108 | no |
| users | 12 | Proposed - Baseline | -0.438 ± 0.061 | yes |
| users | 16 | Optimised - Proposed | -0.042 ± 0.082 | no |
| users | 16 | Proposed - Baseline | -0.676 ± 0.092 | yes |

Violations per pair: Optimised - Proposed: 27, Proposed - Baseline: 25.

### Where Proposed significantly beats Baseline

| sweep | x | Proposed - Baseline |
|---|---|---|
| budget | 0.05 | 0.569 ± 0.082 |
| budget | 0.1 | 0.153 ± 0.053 |
| budget | 0.75 | 0.024 ± 0.004 |
| budget | 1 | 0.055 ± 0.003 |
| speed | 3 | 0.045 ± 0.006 |

## Other observations (measured, stated plainly)

* Proposed utility is **not monotone in the budget**: 1.359 at budget_frac 0.1 vs 1.269 at 0.2. Bits used grow 2.00x but the share of users fed back per slot only 1.20x (0.25 -> 0.30): the extra bits go mostly into higher modes for the same users, and mean regret moves 0.295 -> 0.356. This is consistent with the estimator valuing mode 1 at almost nothing (calibration table below).
* Proposed utility is **not monotone in the budget**: 1.588 at budget_frac 0.3 vs 1.570 at 0.5. Bits used grow 1.80x but the share of users fed back per slot only 1.25x (0.50 -> 0.62): the extra bits go mostly into higher modes for the same users, and mean regret moves 0.177 -> 0.182. This is consistent with the estimator valuing mode 1 at almost nothing (calibration table below).
* Exact spends only 99 of 256 bits per slot on average and still reaches 1.751, above full feedback (1.723). The genie skips feedback whenever the predicted beam is already right and picks the cheapest mode whose decoded beam is right, so it is an upper bound on scheduling, not on CSI quality.
* The critical violation rate is 0.957-0.983 (scheme means) for all four schemes at the default point, the genie included: Q >= 0.8 for a critical user needs about 16.1 dB, above the median best-beam SINR of about 11.3 dB (ASSUMPTIONS.md item 10). Fig 12 therefore has little dynamic range.
* Optimised (exact DP on Uhat) is below Proposed (greedy on the same Uhat) at most points, although greedy is within 0.01% of the DP optimum on Uhat itself (above). Optimising an inaccurate estimate more exactly does not translate into higher true utility here, and each scheme's predictor history follows its own decisions.

## TVCSI estimator calibration (default point, Proposed trajectory)

Mean estimated gain of feedback, Uhat[m] - Uhat[0], against the true gain U[m] - U[0] realised on the same user-slots, and their per-sample correlation.

| mode | estimated gain | true gain | correlation |
|---|---|---|---|
| 1 | 0.0007 | 0.0309 | 0.032 |
| 2 | 0.0087 | 0.0511 | -0.035 |
| 3 | 0.0114 | 0.0580 | -0.008 |

Uhat plugs the *mean* regret of a confidence bin into Q (spec 6): Uhat = v Q(gamma_ref (1 - r_hat)). With the persistence predictor the mode-0 regret in a bin is bimodal (the beam either stayed or moved), so the plug-in value ignores the chance of a large loss, and the confidence (age since feedback) carries little information about which user's beam actually moved. Both show up above as an underestimated and nearly uncorrelated feedback gain. This is a property of the estimator specified for this stage combined with the dummy predictor, not a tuning choice; it was left unchanged.

## Feedback needed to reach 0.9 x full-feedback utility (fig 10)

| speed [km/h] | Exact | Optimised | Proposed | Baseline |
|---|---|---|---|---|
| 3 | 0.050 ± 0.000 | 0.078 ± 0.005 | 0.077 ± 0.005 | 0.167 ± 0.017 |
| 10 | 0.050 ± 0.000 | 0.085 ± 0.003 | 0.092 ± 0.003 | 0.168 ± 0.015 |
| 20 | 0.050 ± 0.000 | 0.270 ± 0.005 | 0.270 ± 0.007 | 0.183 ± 0.005 |
| 30 | 0.050 ± 0.000 | 0.502 ± 0.073 | 0.288 ± 0.002 | 0.214 ± 0.012 |
| 40 | 0.050 ± 0.001 | 0.575 ± 0.036 | 0.521 ± 0.045 | 0.292 ± 0.011 |
| 60 | 0.073 ± 0.003 | 0.614 ± 0.032 | 0.566 ± 0.022 | 0.384 ± 0.008 |

Entries are budget fractions of K*C_3.

## Solver runtime per slot decision [ms] (fig 14)

| K | Baseline | Brute force | DP | Greedy |
|---|---|---|---|---|
| 2 | 0.019 | 0.018 | 0.045 | 0.036 |
| 4 | 0.020 | 0.038 | 0.083 | 0.062 |
| 6 | 0.020 | 0.381 | 0.130 | 0.093 |
| 8 | 0.018 | 7.856 | 0.153 | 0.101 |
| 10 | 0.019 | 187.488 | 0.189 | 0.126 |
| 12 | 0.023 | — | 0.289 | 0.169 |
| 16 | 0.020 | — | 0.378 | 0.242 |
