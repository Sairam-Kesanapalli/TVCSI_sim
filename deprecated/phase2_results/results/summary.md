# TVCSI simulator — results summary

> **DUMMY MODELS.** Every number below comes from the placeholder encoder/decoder (log-normal decoding noise) and the persistence beam predictor. They test the pipeline and the schedulers; they are **not research results** about learned semantic CSI feedback.

Default configuration: K=8, M=B=32, speed=30 km/h, SNR=15 dB, crit_share=0.25, budget_frac=0.25 (C_FB_max=256 bits/slot), T=2000 slots (first 100 excluded), 10 test seeds. Values are mean ± 95% t-interval half-width over seeds.

Task quality: **measured**, Q = 1 - BLER of each message type's payload over 33 PRBs per user of a 270-PRB carrier (5G NR PDSCH link-level simulation, data/q_curves.csv; ASSUMPTIONS.md Phase 2).

## Default configuration, all schemes

| scheme | utility | regret | CVaR0.95 | crit. violation | bits/slot | switch rate |
|---|---|---|---|---|---|---|
| Exact | 2.833 ± 0.029 | 0.258 ± 0.010 | 1.047 ± 0.025 | 0.170 ± 0.014 | 22.2 ± 1.1 | 0.065 ± 0.003 |
| Optimised | 1.250 ± 0.191 | 0.721 ± 0.060 | 3.420 ± 0.266 | 0.763 ± 0.151 | 0.0 ± 0.0 | 0.000 ± 0.000 |
| Proposed | 1.250 ± 0.191 | 0.721 ± 0.060 | 3.420 ± 0.266 | 0.763 ± 0.151 | 0.0 ± 0.0 | 0.000 ± 0.000 |
| Baseline | 2.726 ± 0.028 | 0.129 ± 0.005 | 1.290 ± 0.055 | 0.252 ± 0.018 | 256.0 ± 0.0 | 0.238 ± 0.008 |
| prediction_only | 1.250 ± 0.191 | 0.721 ± 0.060 | 3.420 ± 0.266 | 0.763 ± 0.151 | 0.0 ± 0.0 | 0.000 ± 0.000 |
| importance_only | 2.697 ± 0.034 | 0.246 ± 0.013 | 1.316 ± 0.045 | 0.314 ± 0.024 | 256.0 ± 0.0 | 0.111 ± 0.004 |
| snr_threshold | 2.239 ± 0.088 | 0.348 ± 0.034 | 2.147 ± 0.135 | 0.441 ± 0.086 | 256.0 ± 0.0 | 0.214 ± 0.007 |
| full_feedback | 2.809 ± 0.029 | 0.044 ± 0.002 | 1.094 ± 0.027 | 0.186 ± 0.016 | 1024.0 ± 0.0 | 0.351 ± 0.012 |
| Proposed_no_confidence | 1.250 ± 0.191 | 0.721 ± 0.060 | 3.420 ± 0.266 | 0.763 ± 0.151 | 0.0 ± 0.0 | 0.000 ± 0.000 |
| Proposed_no_importance | 1.250 ± 0.191 | 0.721 ± 0.060 | 3.420 ± 0.266 | 0.763 ± 0.151 | 0.0 ± 0.0 | 0.000 ± 0.000 |

`full_feedback` ignores the budget (reference only). Utility = mean over slots of sum_k U_k. Violation = fraction of (critical user, slot) pairs with Q < Q_min.

## Feasibility oracle and the constraints of P1 (default point)

`BestBeam_oracle` gives every user the beam with the largest **true** gain in every slot, for free: perfect CSI everywhere, no budget. No scheme can beat it in any slot. A critical slot is **feasible** if this oracle reaches Q >= Q_min there; otherwise every scheme violates. Violations among feasible slots are the part a scheduler can fix. All of this is computed after the decisions, from the true channel, and no scheme sees it.

* Critical slots that are infeasible even with the true best beam: **16.2%**. Payload (39i) infeasible slots, all users: 0.3%.
* Constraint parameters (3GPP-cited where a standard exists; ASSUMPTIONS.md items 39-41): eps = 0.1 (TS 38.214); payload {'critical': 1200, 'normal': 300, 'background': 50} bytes (TS 22.185) over 5.94 MHz per user (270 PRBs / K, TS 38.101-1); lambda_F = 0.001, lambda_B = 0.05, lambda_R = 0.1 (no standard value). Sensitivity to all of them: results/sensitivity.md.

| scheme | crit. violation (all) | crit. violation (feasible) | worst crit. user (feasible) | 39j met (share of seeds) | payload miss 39i (all) | payload miss (feasible) | P1 objective 39a |
|---|---|---|---|---|---|---|---|
| Exact | 0.170 ± 0.014 | 0.010 ± 0.001 | 0.013 ± 0.002 | 1.00 ± 0.00 | 0.012 ± 0.002 | 0.009 ± 0.001 | 2.680 ± 0.030 |
| Optimised | 0.763 ± 0.151 | 0.721 ± 0.176 | 0.831 ± 0.194 | 0.00 ± 0.00 | 0.444 ± 0.075 | 0.442 ± 0.075 | 0.908 ± 0.216 |
| Proposed | 0.763 ± 0.151 | 0.721 ± 0.176 | 0.831 ± 0.194 | 0.00 ± 0.00 | 0.444 ± 0.075 | 0.442 ± 0.075 | 0.908 ± 0.216 |
| Baseline | 0.252 ± 0.018 | 0.108 ± 0.007 | 0.119 ± 0.007 | 0.00 ± 0.00 | 0.024 ± 0.001 | 0.020 ± 0.001 | 2.245 ± 0.032 |
| prediction_only | 0.763 ± 0.151 | 0.721 ± 0.176 | 0.831 ± 0.194 | 0.00 ± 0.00 | 0.444 ± 0.075 | 0.442 ± 0.075 | 0.908 ± 0.216 |
| importance_only | 0.314 ± 0.024 | 0.182 ± 0.017 | 0.199 ± 0.015 | 0.00 ± 0.00 | 0.044 ± 0.005 | 0.041 ± 0.005 | 2.265 ± 0.037 |
| snr_threshold | 0.441 ± 0.086 | 0.334 ± 0.101 | 0.461 ± 0.123 | 0.00 ± 0.00 | 0.141 ± 0.021 | 0.138 ± 0.021 | 1.683 ± 0.103 |
| full_feedback | 0.186 ± 0.016 | 0.029 ± 0.003 | 0.035 ± 0.005 | 1.00 ± 0.00 | 0.010 ± 0.001 | 0.007 ± 0.001 | 1.535 ± 0.033 |
| Proposed_no_confidence | 0.763 ± 0.151 | 0.721 ± 0.176 | 0.831 ± 0.194 | 0.00 ± 0.00 | 0.444 ± 0.075 | 0.442 ± 0.075 | 0.908 ± 0.216 |
| Proposed_no_importance | 0.763 ± 0.151 | 0.721 ± 0.176 | 0.831 ± 0.194 | 0.00 ± 0.00 | 0.444 ± 0.075 | 0.442 ± 0.075 | 0.908 ± 0.216 |
| BestBeam_oracle | 0.162 ± 0.015 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.00 ± 0.00 | 0.003 ± 0.000 | 0.000 ± 0.000 | n/a |

'39j met' is 1 for a seed when every critical user's violation rate over its feasible slots is <= eps. The P1 objective is utility - lambda_F bits - lambda_B sum_k s_k - lambda_R CVaR; it is undefined (n/a) for the oracle, which uses no feedback channel.

## Gap decomposition at the default point (paired by seed)

| pair | meaning | utility difference |
|---|---|---|
| Exact - Optimised | information gap | 1.583 ± 0.179 |
| Optimised - Proposed | greedy gap | 0.000 ± 0.000 |
| Proposed - Baseline | contribution | -1.476 ± 0.182 |

## Greedy vs DP

* Random instances (2000, K<=8, random budgets): mean greedy/DP value ratio **0.9829**.
* On the TVCSI estimates along the Proposed trajectory at the default point: mean ratio of estimated utility **1.00000** (± 0.00000 over seeds), worst slot 1.0000, greedy exactly optimal in 100.0% of slots.

## Expected ordering Exact >= Optimised >= Proposed >= Baseline

Checked at the default point and at every point of the five 1-D sweeps using the paired per-seed utility difference. A violation is listed when the mean difference is negative; 'significant' means the 95% interval excludes zero.

**The expected ordering is violated at 29 of 90 (point, pair) checks.**

| sweep | x | pair | mean difference | 95% CI excludes 0 |
|---|---|---|---|---|
| default | default | Proposed - Baseline | -1.476 ± 0.182 | yes |
| budget | 0.1 | Proposed - Baseline | -0.751 ± 0.156 | yes |
| budget | 0.2 | Proposed - Baseline | -1.435 ± 0.180 | yes |
| budget | 0.3 | Proposed - Baseline | -1.476 ± 0.182 | yes |
| budget | 0.5 | Proposed - Baseline | -1.520 ± 0.179 | yes |
| budget | 0.75 | Proposed - Baseline | -1.520 ± 0.179 | yes |
| budget | 1 | Proposed - Baseline | -1.520 ± 0.179 | yes |
| critical_share | 0 | Proposed - Baseline | -1.297 ± 0.189 | yes |
| critical_share | 0.1 | Proposed - Baseline | -1.404 ± 0.171 | yes |
| critical_share | 0.25 | Proposed - Baseline | -1.476 ± 0.182 | yes |
| critical_share | 0.5 | Proposed - Baseline | -1.634 ± 0.164 | yes |
| critical_share | 0.75 | Proposed - Baseline | -1.799 ± 0.175 | yes |
| snr | 5 | Proposed - Baseline | -0.510 ± 0.164 | yes |
| snr | 10 | Proposed - Baseline | -1.156 ± 0.213 | yes |
| snr | 15 | Proposed - Baseline | -1.476 ± 0.182 | yes |
| snr | 20 | Proposed - Baseline | -1.330 ± 0.202 | yes |
| snr | 25 | Proposed - Baseline | -0.985 ± 0.216 | yes |
| speed | 3 | Proposed - Baseline | -1.625 ± 0.254 | yes |
| speed | 10 | Proposed - Baseline | -1.584 ± 0.197 | yes |
| speed | 20 | Proposed - Baseline | -1.533 ± 0.179 | yes |
| speed | 30 | Proposed - Baseline | -1.476 ± 0.182 | yes |
| speed | 40 | Proposed - Baseline | -1.443 ± 0.190 | yes |
| speed | 60 | Proposed - Baseline | -1.335 ± 0.182 | yes |
| users | 2 | Proposed - Baseline | -0.301 ± 0.120 | yes |
| users | 4 | Proposed - Baseline | -0.586 ± 0.239 | yes |
| users | 6 | Proposed - Baseline | -1.136 ± 0.220 | yes |
| users | 8 | Proposed - Baseline | -1.476 ± 0.182 | yes |
| users | 12 | Proposed - Baseline | -1.880 ± 0.282 | yes |
| users | 16 | Proposed - Baseline | -1.946 ± 0.308 | yes |

Violations per pair: Proposed - Baseline: 29.

### Where Proposed significantly beats Baseline

Nowhere: no sweep point has a paired 95% interval above zero.

## Other observations (measured, stated plainly)

* Exact spends 22 of 256 bits per slot on average and reaches 2.833, above full feedback (2.809). The genie skips feedback whenever the predicted beam is already right and picks the cheapest mode whose decoded beam is right, so it is an upper bound on scheduling, not on CSI quality.
* The critical violation rate is 0.170-0.763 (scheme means) for the four main schemes at the default point. 16.2% of critical slots are infeasible even with the true best beam, so every scheme violates there; over the feasible slots the rates are Exact 0.010, Optimised 0.721, Proposed 0.721, Baseline 0.108 (fig 12b).
* Optimised (exact DP on Uhat) is below Proposed (greedy on the same Uhat) at 0 of 29 sweep points, although greedy is within 0.00% of the DP optimum on Uhat itself (above). Where Optimised loses, optimising an inaccurate estimate more exactly does not translate into higher true utility, and each scheme's predictor history follows its own decisions.

## TVCSI estimator calibration (default point, Proposed trajectory)

Mean estimated gain of feedback, Uhat[m] - Uhat[0], against the true gain U[m] - U[0] realised on the same user-slots, and their per-sample correlation.

| mode | estimated gain | true gain | correlation |
|---|---|---|---|
| 1 | 0.0000 | 0.1710 | nan |
| 2 | 0.0000 | 0.1900 | nan |
| 3 | 0.0000 | 0.1949 | nan |

Uhat plugs the *mean* regret of a confidence bin into Q (spec 6): Uhat = v Q(gamma_ref (1 - r_hat)). With the persistence predictor the mode-0 regret in a bin is bimodal (the beam either stayed or moved), so the plug-in value ignores the chance of a large loss, and the confidence (age since feedback) carries little information about which user's beam actually moved. Both show up above as an underestimated and nearly uncorrelated feedback gain. This is a property of the estimator specified for this stage combined with the dummy predictor, not a tuning choice; it was left unchanged.

## Feedback needed to reach 0.9 x full-feedback utility (fig 10)

| speed [km/h] | Exact | Optimised | Proposed | Baseline |
|---|---|---|---|---|
| 3 | 0.050 ± 0.000 | not reached | not reached | 0.168 ± 0.019 |
| 10 | 0.050 ± 0.000 | not reached | not reached | 0.168 ± 0.017 |
| 20 | 0.050 ± 0.000 | not reached | not reached | 0.172 ± 0.014 |
| 30 | 0.050 ± 0.000 | not reached | not reached | 0.177 ± 0.010 |
| 40 | 0.050 ± 0.000 | not reached | not reached | 0.184 ± 0.008 |
| 60 | 0.050 ± 0.000 | not reached | not reached | 0.197 ± 0.004 |

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
