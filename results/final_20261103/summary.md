# TVCSI simulator — results summary

Models: **trained** (`tvcsi.models.trained:TrainedEncoder`, `tvcsi.models.trained:TrainedDecoder`, `tvcsi.models.trained:Persistence`; weights `data/models/s6_models.npz`, trained by scripts/train_models.py, research/checkpoints/S6_checkpoint.md).

Default configuration: K=8, M=B=32, speed=30 km/h, SNR=15 dB, crit_share=0.25, budget_frac=0.25 (C_FB_max=256 bits/slot), T=2000 slots (first 100 excluded), 10 test seeds. Values are mean ± 95% t-interval half-width over seeds.

Task quality: **measured**, Q = 1 - BLER of each message type's payload over 33 PRBs per user of a 270-PRB carrier (5G NR PDSCH link-level simulation, data/q_curves.csv; ASSUMPTIONS.md Phase 2).

## Default configuration, all schemes

| scheme | utility | regret | CVaR0.95 | crit. violation | bits/slot | switch rate |
|---|---|---|---|---|---|---|
| Exact | 2.833 ± 0.029 | 0.257 ± 0.010 | 1.045 ± 0.024 | 0.171 ± 0.015 | 20.3 ± 1.3 | 0.065 ± 0.003 |
| Optimised | 2.773 ± 0.033 | 0.162 ± 0.011 | 1.196 ± 0.058 | 0.207 ± 0.020 | 256.0 ± 0.0 | 0.199 ± 0.008 |
| Proposed | 2.774 ± 0.033 | 0.161 ± 0.011 | 1.189 ± 0.055 | 0.207 ± 0.020 | 256.0 ± 0.0 | 0.198 ± 0.008 |
| Baseline | 2.729 ± 0.036 | 0.124 ± 0.008 | 1.291 ± 0.056 | 0.249 ± 0.023 | 256.0 ± 0.0 | 0.223 ± 0.008 |
| prediction_only | 1.250 ± 0.191 | 0.721 ± 0.060 | 3.420 ± 0.266 | 0.763 ± 0.151 | 0.0 ± 0.0 | 0.000 ± 0.000 |
| importance_only | 2.684 ± 0.036 | 0.255 ± 0.014 | 1.348 ± 0.047 | 0.324 ± 0.026 | 256.0 ± 0.0 | 0.113 ± 0.005 |
| snr_threshold | 2.129 ± 0.132 | 0.385 ± 0.030 | 2.329 ± 0.167 | 0.466 ± 0.111 | 256.0 ± 0.0 | 0.186 ± 0.010 |
| full_feedback | 2.798 ± 0.030 | 0.057 ± 0.004 | 1.126 ± 0.037 | 0.195 ± 0.018 | 1024.0 ± 0.0 | 0.357 ± 0.013 |
| Proposed_no_importance | 2.767 ± 0.034 | 0.152 ± 0.010 | 1.210 ± 0.053 | 0.206 ± 0.020 | 256.0 ± 0.0 | 0.213 ± 0.009 |
| Proposed_plugin | 1.250 ± 0.191 | 0.721 ± 0.060 | 3.420 ± 0.266 | 0.763 ± 0.151 | 0.0 ± 0.0 | 0.000 ± 0.000 |
| Optimised_plugin | 1.250 ± 0.191 | 0.721 ± 0.060 | 3.420 ± 0.266 | 0.763 ± 0.151 | 0.0 ± 0.0 | 0.000 ± 0.000 |
| Proposed_risk | 2.773 ± 0.034 | 0.165 ± 0.011 | 1.188 ± 0.052 | 0.205 ± 0.020 | 256.0 ± 0.0 | 0.197 ± 0.008 |
| fill_periodic | 2.622 ± 0.034 | 0.215 ± 0.008 | 1.563 ± 0.078 | 0.335 ± 0.025 | 256.0 ± 0.0 | 0.119 ± 0.004 |
| fill_age | 2.674 ± 0.034 | 0.173 ± 0.007 | 1.421 ± 0.058 | 0.294 ± 0.021 | 256.0 ± 0.0 | 0.131 ± 0.005 |
| fill_uncertainty | 1.971 ± 0.111 | 0.488 ± 0.038 | 2.594 ± 0.203 | 0.530 ± 0.078 | 256.0 ± 0.0 | 0.093 ± 0.004 |
| fill_importance | 2.684 ± 0.036 | 0.255 ± 0.014 | 1.348 ± 0.047 | 0.324 ± 0.026 | 256.0 ± 0.0 | 0.113 ± 0.005 |
| fill_channel | 1.824 ± 0.145 | 0.527 ± 0.036 | 2.838 ± 0.180 | 0.663 ± 0.108 | 256.0 ± 0.0 | 0.091 ± 0.005 |

`full_feedback` ignores the budget (reference only). Utility = mean over slots of sum_k U_k. Violation = fraction of (critical user, slot) pairs with Q < Q_min.

## Feasibility oracle and the constraints of P1 (default point)

`BestBeam_oracle` gives every user the beam with the largest **true** gain in every slot, for free: perfect CSI everywhere, no budget. No scheme can beat it in any slot. A critical slot is **feasible** if this oracle reaches Q >= Q_min there; otherwise every scheme violates. Violations among feasible slots are the part a scheduler can fix. All of this is computed after the decisions, from the true channel, and no scheme sees it.

* Critical slots that are infeasible even with the true best beam: **16.2%**. Payload (39i) infeasible slots, all users: 0.3%.
* Constraint parameters (3GPP-cited where a standard exists; ASSUMPTIONS.md items 39-41): eps = 0.1 (TS 38.214); payload {'critical': 1200, 'normal': 300, 'background': 50} bytes (TS 22.185) over 5.94 MHz per user (270 PRBs / K, TS 38.101-1); lambda_F = 0.001, lambda_B = 0.05, lambda_R = 0.1 (no standard value). Sensitivity to all of them: results/sensitivity.md.

| scheme | crit. violation (all) | crit. violation (feasible) | worst crit. user (feasible) | 39j met (share of seeds) | payload miss 39i (all) | payload miss (feasible) | P1 objective 39a |
|---|---|---|---|---|---|---|---|
| Exact | 0.171 ± 0.015 | 0.011 ± 0.002 | 0.014 ± 0.003 | 1.00 ± 0.00 | 0.012 ± 0.002 | 0.009 ± 0.001 | 2.682 ± 0.031 |
| Optimised | 0.207 ± 0.020 | 0.054 ± 0.008 | 0.064 ± 0.012 | 1.00 ± 0.00 | 0.016 ± 0.002 | 0.013 ± 0.002 | 2.317 ± 0.038 |
| Proposed | 0.207 ± 0.020 | 0.054 ± 0.008 | 0.063 ± 0.011 | 1.00 ± 0.00 | 0.016 ± 0.002 | 0.012 ± 0.002 | 2.320 ± 0.038 |
| Baseline | 0.249 ± 0.023 | 0.105 ± 0.013 | 0.120 ± 0.017 | 0.10 ± 0.23 | 0.023 ± 0.002 | 0.020 ± 0.002 | 2.254 ± 0.042 |
| prediction_only | 0.763 ± 0.151 | 0.721 ± 0.176 | 0.831 ± 0.194 | 0.00 ± 0.00 | 0.444 ± 0.075 | 0.442 ± 0.075 | 0.908 ± 0.216 |
| importance_only | 0.324 ± 0.026 | 0.194 ± 0.020 | 0.213 ± 0.021 | 0.00 ± 0.00 | 0.047 ± 0.006 | 0.044 ± 0.005 | 2.248 ± 0.040 |
| snr_threshold | 0.466 ± 0.111 | 0.365 ± 0.129 | 0.496 ± 0.178 | 0.10 ± 0.23 | 0.178 ± 0.036 | 0.175 ± 0.036 | 1.566 ± 0.150 |
| full_feedback | 0.195 ± 0.018 | 0.040 ± 0.005 | 0.048 ± 0.007 | 1.00 ± 0.00 | 0.012 ± 0.001 | 0.009 ± 0.001 | 1.519 ± 0.035 |
| Proposed_no_importance | 0.206 ± 0.020 | 0.053 ± 0.009 | 0.064 ± 0.013 | 1.00 ± 0.00 | 0.015 ± 0.002 | 0.012 ± 0.002 | 2.305 ± 0.041 |
| Proposed_plugin | 0.763 ± 0.151 | 0.721 ± 0.176 | 0.831 ± 0.194 | 0.00 ± 0.00 | 0.444 ± 0.075 | 0.442 ± 0.075 | 0.908 ± 0.216 |
| Optimised_plugin | 0.763 ± 0.151 | 0.721 ± 0.176 | 0.831 ± 0.194 | 0.00 ± 0.00 | 0.444 ± 0.075 | 0.442 ± 0.075 | 0.908 ± 0.216 |
| Proposed_risk | 0.205 ± 0.020 | 0.052 ± 0.008 | 0.061 ± 0.011 | 1.00 ± 0.00 | 0.016 ± 0.002 | 0.013 ± 0.002 | 2.320 ± 0.038 |
| fill_periodic | 0.335 ± 0.025 | 0.207 ± 0.019 | 0.221 ± 0.022 | 0.00 ± 0.00 | 0.040 ± 0.004 | 0.037 ± 0.003 | 2.162 ± 0.041 |
| fill_age | 0.294 ± 0.021 | 0.158 ± 0.013 | 0.172 ± 0.016 | 0.00 ± 0.00 | 0.031 ± 0.002 | 0.027 ± 0.002 | 2.223 ± 0.038 |
| fill_uncertainty | 0.530 ± 0.078 | 0.440 ± 0.086 | 0.573 ± 0.099 | 0.00 ± 0.00 | 0.214 ± 0.037 | 0.211 ± 0.037 | 1.418 ± 0.132 |
| fill_importance | 0.324 ± 0.026 | 0.194 ± 0.020 | 0.213 ± 0.021 | 0.00 ± 0.00 | 0.047 ± 0.006 | 0.044 ± 0.005 | 2.248 ± 0.040 |
| fill_channel | 0.663 ± 0.108 | 0.600 ± 0.126 | 0.680 ± 0.111 | 0.00 ± 0.00 | 0.264 ± 0.043 | 0.261 ± 0.043 | 1.248 ± 0.163 |
| BestBeam_oracle | 0.162 ± 0.015 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.00 ± 0.00 | 0.003 ± 0.000 | 0.000 ± 0.000 | n/a |

'39j met' is 1 for a seed when every critical user's violation rate over its feasible slots is <= eps. The P1 objective is utility - lambda_F bits - lambda_B sum_k s_k - lambda_R CVaR; it is undefined (n/a) for the oracle, which uses no feedback channel.

## Proposed vs every simple baseline (default point, seed-paired, Holm-corrected)

| other | Proposed - other | wins | Holm p | Proposed_risk - other | wins | Holm p |
|---|---|---|---|---|---|---|
| Baseline | 0.045 ± 0.007 | 10/10 | 4.7e-07 | 0.044 ± 0.007 | 10/10 | 3.8e-07 |
| fill_periodic | 0.152 ± 0.012 | 10/10 | 3e-09 | 0.151 ± 0.012 | 10/10 | 4e-09 |
| fill_age | 0.100 ± 0.008 | 10/10 | 3e-09 | 0.099 ± 0.008 | 10/10 | 4e-09 |
| fill_uncertainty | 0.803 ± 0.107 | 10/10 | 1.5e-07 | 0.802 ± 0.106 | 10/10 | 1.5e-07 |
| fill_importance | 0.090 ± 0.011 | 10/10 | 1e-07 | 0.089 ± 0.011 | 10/10 | 1e-07 |
| fill_channel | 0.950 ± 0.128 | 10/10 | 1.5e-07 | 0.949 ± 0.127 | 10/10 | 1.5e-07 |
| importance_only | 0.090 ± 0.011 | 10/10 | 1e-07 | 0.089 ± 0.011 | 10/10 | 1e-07 |
| snr_threshold | 0.645 ± 0.118 | 10/10 | 6e-07 | 0.644 ± 0.118 | 10/10 | 6.1e-07 |

fill_* are the five faculty baselines (largest mode that fits, ASSUMPTIONS item 64); Proposed_risk adds the critical-user virtual queue (items 65-66).

## Gap decomposition at the default point (paired by seed)

| pair | meaning | utility difference |
|---|---|---|
| Exact - Optimised | information gap | 0.060 ± 0.008 |
| Optimised - Proposed | greedy gap | -0.001 ± 0.002 |
| Proposed - Baseline | contribution | 0.045 ± 0.007 |

## Greedy vs DP

* Random instances (2000, K<=8, random budgets): mean greedy/DP value ratio **0.9829**.
* On the TVCSI estimates along the Proposed trajectory at the default point: mean ratio of estimated utility **0.99997** (± 0.00000 over seeds), worst slot 0.9956, greedy exactly optimal in 93.6% of slots.

## Expected ordering Exact >= Optimised >= Proposed >= Baseline

Checked at the default point and at every point of the five 1-D sweeps using the paired per-seed utility difference. A violation is listed when the mean difference is negative; 'significant' means the 95% interval excludes zero.

**The expected ordering is violated at 20 of 90 (point, pair) checks.**

| sweep | x | pair | mean difference | 95% CI excludes 0 |
|---|---|---|---|---|
| default | default | Optimised - Proposed | -0.001 ± 0.002 | no |
| budget | 0.75 | Optimised - Proposed | -0.000 ± 0.000 | no |
| critical_share | 0 | Optimised - Proposed | -0.000 ± 0.001 | no |
| critical_share | 0.1 | Optimised - Proposed | -0.001 ± 0.001 | no |
| critical_share | 0.25 | Optimised - Proposed | -0.001 ± 0.002 | no |
| critical_share | 0.5 | Optimised - Proposed | -0.001 ± 0.002 | no |
| critical_share | 0.75 | Optimised - Proposed | -0.000 ± 0.002 | no |
| snr | 10 | Optimised - Proposed | -0.001 ± 0.005 | no |
| snr | 15 | Optimised - Proposed | -0.001 ± 0.002 | no |
| snr | 25 | Optimised - Proposed | -0.000 ± 0.000 | no |
| speed | 3 | Optimised - Proposed | -0.001 ± 0.002 | no |
| speed | 10 | Optimised - Proposed | -0.000 ± 0.002 | no |
| speed | 20 | Optimised - Proposed | -0.001 ± 0.002 | no |
| speed | 30 | Optimised - Proposed | -0.001 ± 0.002 | no |
| speed | 40 | Optimised - Proposed | -0.001 ± 0.001 | no |
| speed | 60 | Optimised - Proposed | -0.000 ± 0.003 | no |
| users | 2 | Optimised - Proposed | -0.000 ± 0.001 | no |
| users | 4 | Optimised - Proposed | -0.000 ± 0.001 | no |
| users | 8 | Optimised - Proposed | -0.001 ± 0.002 | no |
| users | 16 | Optimised - Proposed | -0.000 ± 0.001 | no |

Violations per pair: Optimised - Proposed: 20.

### Where Proposed significantly beats Baseline

| sweep | x | Proposed - Baseline |
|---|---|---|
| budget | 0.05 | 1.301 ± 0.167 |
| budget | 0.1 | 0.690 ± 0.172 |
| budget | 0.2 | 0.071 ± 0.011 |
| budget | 0.3 | 0.045 ± 0.007 |
| budget | 0.5 | 0.018 ± 0.006 |
| budget | 0.75 | 0.023 ± 0.006 |
| budget | 1 | 0.024 ± 0.006 |
| critical_share | 0 | 0.017 ± 0.004 |
| critical_share | 0.1 | 0.033 ± 0.006 |
| critical_share | 0.25 | 0.045 ± 0.007 |
| critical_share | 0.5 | 0.061 ± 0.007 |
| critical_share | 0.75 | 0.078 ± 0.005 |
| snr | 5 | 0.137 ± 0.058 |
| snr | 10 | 0.083 ± 0.013 |
| snr | 15 | 0.045 ± 0.007 |
| snr | 20 | 0.023 ± 0.004 |
| snr | 25 | 0.007 ± 0.002 |
| speed | 3 | 0.038 ± 0.015 |
| speed | 10 | 0.033 ± 0.008 |
| speed | 20 | 0.035 ± 0.006 |
| speed | 30 | 0.045 ± 0.007 |
| speed | 40 | 0.070 ± 0.010 |
| speed | 60 | 0.107 ± 0.011 |
| users | 2 | 0.005 ± 0.001 |
| users | 4 | 0.013 ± 0.004 |
| users | 6 | 0.033 ± 0.009 |
| users | 8 | 0.045 ± 0.007 |
| users | 12 | 0.050 ± 0.003 |
| users | 16 | 0.048 ± 0.006 |

## Other observations (measured, stated plainly)

* Exact spends 20 of 256 bits per slot on average and reaches 2.833, above full feedback (2.798). The genie skips feedback whenever the predicted beam is already right and picks the cheapest mode whose decoded beam is right, so it is an upper bound on scheduling, not on CSI quality.
* The critical violation rate is 0.171-0.249 (scheme means) for the four main schemes at the default point. 16.2% of critical slots are infeasible even with the true best beam, so every scheme violates there; over the feasible slots the rates are Exact 0.011, Optimised 0.054, Proposed 0.054, Baseline 0.105 (fig 12b).
* Optimised (exact DP on Uhat) is below Proposed (greedy on the same Uhat) at 19 of 29 sweep points, although greedy is within 0.00% of the DP optimum on Uhat itself (above). Where Optimised loses, optimising an inaccurate estimate more exactly does not translate into higher true utility, and each scheme's predictor history follows its own decisions.

## TVCSI estimator calibration (default point, Proposed trajectory)

Mean estimated gain of feedback, Uhat[m] - Uhat[0], against the true gain U[m] - U[0] realised on the same user-slots, and their per-sample correlation.

| mode | estimated gain | true gain | correlation |
|---|---|---|---|
| 1 | 0.0016 | -0.0030 | 0.132 |
| 2 | 0.0151 | 0.0126 | 0.237 |
| 3 | 0.0175 | 0.0156 | 0.278 |

Uhat = v E[Q | gNB-side features] from the frozen learned estimator (S4, ASSUMPTIONS items 56-63), fitted on train seeds and chosen on validation seeds; these numbers are on the test seeds along the Proposed trajectory. The pre-S4 mean-regret plug-in (kept as Proposed_plugin) had a constant estimated gain of 0 here (correlation 0).

## Feedback needed to reach 0.9 x full-feedback utility (fig 10)

| speed [km/h] | Exact | Optimised | Proposed | Baseline |
|---|---|---|---|---|
| 3 | 0.050 ± 0.000 | 0.050 ± 0.000 | 0.050 ± 0.000 | 0.167 ± 0.019 |
| 10 | 0.050 ± 0.000 | 0.050 ± 0.000 | 0.050 ± 0.000 | 0.166 ± 0.017 |
| 20 | 0.050 ± 0.000 | 0.050 ± 0.000 | 0.050 ± 0.001 | 0.170 ± 0.016 |
| 30 | 0.050 ± 0.000 | 0.050 ± 0.004 | 0.050 ± 0.005 | 0.176 ± 0.011 |
| 40 | 0.050 ± 0.000 | 0.059 ± 0.008 | 0.059 ± 0.008 | 0.183 ± 0.009 |
| 60 | 0.050 ± 0.000 | 0.082 ± 0.007 | 0.082 ± 0.007 | 0.196 ± 0.005 |

Entries are budget fractions of K*C_3.

## Solver runtime per slot decision [ms] (fig 14)

| K | Baseline | Brute force | DP | Greedy |
|---|---|---|---|---|
| 2 | 0.020 | 0.018 | 0.041 | 0.031 |
| 4 | 0.019 | 0.030 | 0.091 | 0.060 |
| 6 | 0.020 | 0.237 | 0.116 | 0.078 |
| 8 | 0.021 | 6.572 | 0.168 | 0.101 |
| 10 | 0.016 | 133.136 | 0.171 | 0.108 |
| 12 | 0.018 | — | 0.196 | 0.125 |
| 16 | 0.018 | — | 0.284 | 0.164 |
