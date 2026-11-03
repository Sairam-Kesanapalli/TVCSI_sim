# Assumptions, deviations and tuned constants

Every choice the spec left open, and every place where the code differs from the spec or from
the formulation document (`docs/5G_group_5.pdf`, Eqs. 1-39). Where the task prompt and the PDF
differ, the prompt was followed.

## Environment and inputs

1. **Python 3.12** was used (3.11 is not installed on the build machine). The code needs >= 3.11
   and uses nothing specific to 3.12.
2. **Only one PDF was available**: `docs/5G_group_5.pdf`, the system-model / problem-formulation
   document. The Team 5 proposal PDF mentioned in the task was not in the workspace, so nothing
   was taken from it.

## Scope relative to problem P1 (Eq. 39)

3. The simulator implements the feedback-mode sub-problem only. Single semantic representation
   (L = 1, so y is fixed), one user per resource block with no interference (Eq. 26 reduces to
   gamma_k = snr * g_{k,bhat} / M, spec 3.5), no power allocation, and no payload constraint (39i).
4. The schedulers maximise the (estimated) utility sum under the hard feedback budget (39c).
   The penalty terms lambda_F, lambda_B, lambda_R of (39a) are not used and the reliability
   constraint (39j) is not enforced. CVaR (Eq. 37), beam switching (Eq. 38) and the critical
   violation rate (Eq. 34) are reported as evaluation metrics only.
5. **Notation.** Eq. 25 uses v_k(t) for the beamformer, which clashes with the importance v_k(t)
   of Eq. 16. In the code `v` always means importance. The beamformer is never materialised: the
   served beam is the codebook column `W[:, bhat_k]`. If a beamformer variable is ever added, it
   must be named `f_k`.

## Channel (spec 3.1)

6. Angles evolve as a Gaussian random walk and are then reflected into [-pi/2, pi/2] with
   theta <- arcsin(sin(theta)). sin(theta), and so the channel, is unchanged by this; it only
   keeps the stored angle physical. This is how "keep sin(theta) wrapped into [-1, 1]" was read.
7. **c_theta = 2e-4 rad per (km/h) per slot (the default, unchanged).** Measured rate of change
   of the true best beam (seed 0, 32 users x 2000 slots):

   | speed [km/h] | rho = J0(2 pi f_d Ts) | best-beam changes / slot, c_theta = 0 | c_theta = 2e-4 | c_theta = 1e-3 |
   |---|---|---|---|---|
   | 3  | 0.9991 | 0.031 | 0.030 | 0.055 |
   | 10 | 0.9896 | 0.093 | 0.100 | 0.169 |
   | 30 | 0.9087 | 0.261 | 0.288 | 0.442 |
   | 60 | 0.6598 | 0.470 | 0.506 | 0.692 |

   With the default, the best beam changes about every 3.5 slots at 30 km/h, so beam changes
   clearly occur there. They come mostly from path-gain fading, which swaps the dominant path.
   Angle drift adds about 10 % at 30 km/h. No tuning was needed.
8. Pilot estimate: hhat = h + CN(0, sigma_e^2 I) per antenna with sigma_e^2 = M / snr_pilot and
   snr_pilot = snr_dB + 5 dB, as specified. The pilot SNR moves with snr_dB in the SNR sweep.
9. `load_deepmimo()` is a stub raising NotImplementedError. Its docstring states the
   [users, time, M] interface.

## Link and task quality (spec 3.5-3.6)

10. **Calibration (scripts/calibrate_q.py): defaults kept** (snr_dB = 15, gamma0 = 14/10/6 dB,
    s = 1.5 dB). Mean Q over channel realisations at the default SNR:

    | type | gamma0 | mean Q, best beam | mean Q, 3 dB beam loss | change |
    |---|---|---|---|---|
    | critical   | 14 dB | 0.236 | 0.069 | 0.168 |
    | normal     | 10 dB | 0.613 | 0.321 | 0.293 |
    | background |  6 dB | 0.889 | 0.701 | 0.188 |

    None of the types is saturated (check: mean Q at zero regret within [0.05, 0.95]). A 3 dB
    loss visibly changes the critical users' Q (check: change >= 0.1). So, by the rule in 3.6,
    nothing was adjusted. **Consequence:** with P = 3 paths the median best-beam SINR is 11.3 dB.
    Q >= Q_min = 0.8 for a critical user needs gamma >= 16.1 dB, which is rarely reached. The
    critical violation rate is therefore high (around 0.95) for every scheme, including the
    genie. Fig 12 has little dynamic range at the default settings. This was not tuned away.
11. Q is evaluated as 0.5 (1 + tanh(z / 2)), a numerically stable form of the logistic. SINR is
    floored at 1e-12 before conversion to dB.

## Importance (spec 3.7)

12. The number of critical users is round-half-up(crit_share * K). With K = 2 and crit_share =
    0.25 this gives 1. The first |K_c| user indices are critical. The rest alternate normal,
    background, normal, ... starting with normal.
13. The initial Markov state is drawn from the chain's stationary distribution
    (P(high) = p_lh / (p_lh + p_hl)), so there is no start-up transient.

## Models (spec 5)

14. **Dummy encoder noise is shared across modes.** ghat^m = g(hhat) * exp(sigma_m * Z), where Z
    is one N(0, 1) draw per (slot, user, beam) used by all three modes. The spec formula does not
    say whether N(0, 1) is common to the modes. Independent draws were used first. That let the
    genie `Exact` pick the luckiest of three independent decodes, and it beat full feedback by
    a noticeable margin. A shared Z matches progressive refinement (Eq. 10: a higher mode refines
    the same description). The dummy code has B entries rather than d_m; feedback size enters
    only through C_m, as the spec allows.
15. **Predictor interface.** The predictor owns its history buffer: `predict()` takes no argument
    and `update(ghat, fresh)` adds a `fresh` mask (True where the user fed back). The mask is
    needed to compute the age, and a recurrent model needs its own state anyway. This differs
    from the spec's `predict(ghat_hist)` signature.
16. **Persistence predictor details.** Before any feedback it holds the uninformative prior
    g = M for every beam (nominal SNR attainable) with confidence 0, so the first slots request
    feedback instead of deadlocking at gamma_ref ~ 0. age = slots since the user's last mode > 0
    feedback (1 right after feedback). confidence = exp(-age / tau) with **tau = 1 / (1 - rho)**
    at the configured speed (config `dummy.tau: auto`): 10.95 slots at 30 km/h, about 1075 at
    3 km/h. A fixed number can be set instead.
17. `models: trained` loads `package.module:ClassName` for encoder, decoder and predictor. The
    constructors receive `cfg=`, plus `W=` and `rng=` (encoder) or `K=` (predictor), plus
    `trained.kwargs`.

## TVCSI estimator (spec 6)

18. **Samples.** Every (user, slot) of a fitting run records the confidence at decision time and
    the regret each of the four modes would have realised in that slot. Mode 0 comes from the
    policy's own predictor; modes 1-3 come from the shared decoded feedback. This is the realised
    regret of each mode, observed for all modes at once instead of only the chosen one, so every
    cell of the table gets data. Warm-up slots are included.
19. **Fitting policies**: `prediction_only` plus three random-mode policies. Each user
    independently uses mode 0 with probability p0 in {0.5, 0.9, 0.98}, otherwise a uniform mode
    in 1..3. This spreads the predictor age (and so the confidence) over a wide range. These
    policies are budget-exempt; they only collect data.
20. **Bins.** 10 confidence quantiles are requested. Duplicate edges (many samples at exactly
    confidence 0, from `prediction_only`) are merged, and an edge equal to the minimum is
    dropped. At the default point this leaves **8 bins**. Empty cells fall back to the pooled
    mean of their mode.
21. **One table per sweep point** (`tvcsi.refit_per_point: true`). Mode regrets depend strongly
    on the pilot SNR, and prediction regret on speed. A single 30 km/h / 15 dB table applied at
    5 dB or 60 km/h would be miscalibrated, and that would confound the speed and SNR figures
    with estimator mismatch. Each point therefore gets its own table, fitted on the validation
    seeds at that configuration. This is equivalent to the gNB calibrating its estimator for its
    deployment. It affects Optimised, Proposed and both ablations alike. Set the flag to false to
    use one table fitted at the default configuration. The default-point table is
    `results/tvcsi_table.json`; per-point tables are cached in `results/tables/<config hash>.json`.
22. gamma_ref, Uhat and TVCSI follow spec 6 exactly, including the plug-in of the *mean* bin
    regret inside Q. See results/summary.md for a measured consequence. The estimator was not
    modified.

## Solvers and schemes (spec 7)

23. DP: the budget is discretised in gcd(C_m) = 32-bit units, using floor(C_FB_max / 32) units.
    This is exact because every C_m is a multiple of 32. Ties go to the cheaper mode.
24. Greedy: upper concave hull per user, built from (0, Uhat_0) and keeping only strictly
    positive slopes. A mode pruned from the hull is never revisited. That, plus stopping when the
    next hull step does not fit, is the greedy gap the DP closes.
25. `Exact` runs the same DP on the true utilities U[k][m] = v_k Q(gamma^m_k). Its mode-0 entry
    uses Exact's own predictor history.
26. Periodic baseline: P = ceil(K C_mfixed / C_FB_max); user k feeds back iff (t + k) mod P == 0;
    on overflow, users are dropped in ascending index order. If C_FB_max < C_mfixed, it never
    feeds back.
27. snr_threshold gives mode m_fixed by ascending gamma_ref; importance_only gives mode 3 by
    descending v. Both stop at the first user that does not fit. All their requests cost the
    same, so this is the same as skipping.

## Metrics, figures and runs (spec 3.9, 8)

28. Switch rate uses the served beam; the first slot counts as no switch (it is inside the
    warm-up anyway). Utility per bit = utility / mean bits per slot (NaN for zero bits).
    The violation rate is NaN when there are no critical users (crit_share = 0); that point is
    then missing from the curve.
29. CVaR uses the exact sorted-tail form of the empirical distribution, tested against the
    min-over-eta form.
30. Seeds: test 1000-1009, validation 2000-2004 (table fitting only), train 3000-3019 (dataset
    export only). Each random source (channel, importance, pilot noise, encoder noise, policy)
    has its own stream from `SeedSequence(seed).spawn`, so adding a scheme never changes a
    realisation.
31. Identical configurations that appear in several sweeps (the default point is in the K,
    speed, crit_share and SNR sweeps) are simulated once and reported in each.
32. Fig 10 uses an extra speed x budget grid: 6 speeds x 7 budget fractions, schemes main +
    full_feedback. The point estimate comes from the seed-mean utility curve with linear
    interpolation between grid points. Error bars come from per-seed crossings and are drawn only
    when every seed reaches the target. The point is omitted when the mean curve never does.
33. Fig 14 times solvers on random uniform utility instances at budget_frac 0.25 (50 instances
    x 10 seeds per K). The K grid adds K = 10 to the sweep grid so that brute force is timed up
    to its stated limit. The brute-force enumeration of mode vectors is cached per K and
    excluded from the timing. The "Baseline" curve is the periodic rule.
34. Supplementary figures come from the same runs as the main figures: all ten schemes are
    simulated together on each realisation. Panel (a) adds the supplementary baselines, panel
    (b) the two ablations. Splitting into two panels keeps each panel within the 8-hue
    categorical palette.
35. Figure colours: Exact (a genie bound) is neutral black and dashed; the implementable schemes
    take the validated categorical palette slots in fixed order. The palette validator script
    could not be run because node.js is not installed, so the palette's documented validated
    order is used unchanged.
36. The dataset export (`results/dataset/{train,val,test}.npz`) is generated at the default
    configuration (K = 8, 30 km/h, T = 2000), with seeds stacked along the user axis.

## Phase 1: feasibility oracle and the missing constraints of P1 (faculty review, Nov 2026)

37. **True best-beam oracle (`BestBeam_oracle`).** In every slot each user gets the codebook beam
    with the largest *true* gain, for free (perfect, instantaneous CSI for everyone, no budget).
    Q is increasing in the beam gain, so no scheme can beat it in any slot: it bounds the utility
    from above and the violation rates from below. It is computed in `sim.py` after the slot loop,
    from the true channel, and is never passed to a scheme (test
    `test_constraints_do_not_change_decisions`, and spec test 10 still passes). It appears as an
    extra row in every results CSV; its `bits` is NaN because it uses no feedback channel.
38. **Feasible slot.** A critical (user, slot) pair is *feasible* if the oracle reaches Q >= Q_min
    there. In an infeasible pair every scheme violates, so
    `violation_rate = infeasible_share + (1 - infeasible_share) * violation_rate_feasible`
    holds exactly (tested). `violation_rate_feasible` is the part a scheduler can change.
    **Measured at the default point: 95.7% of critical slots are infeasible** (seed-mean, 10 test
    seeds), which is the whole of Exact's 0.957 violation rate. At 20 dB SNR it is 48%, at 25 dB 6.5%.
39. **Constraint 39j (reliability) as a metric.** Evaluated per critical user over that user's
    feasible slots; `crit_worst_violation_feasible` is the worst user and `reliability_met` is 1 when
    every critical user is <= eps. Values: see "Parameter values" below (P1).
40. **Constraint 39i (payload) as a metric.** Delta R_k(t) = W_k Ts log2(1 + gamma_k) against a
    per-type payload S_k. A stale beam can land in a near-null of the true channel (SINR ~ -36 dB was
    observed), so a scheme can miss even when bandwidth is huge. Values: P2-P8 below.
41. **Objective 39a as a metric.** `objective_p1 = utility - lambda_F bits - lambda_B sum_k s_k -
    lambda_R CVaR_alpha(L_sem)` from per-slot means over a run. No scheme optimises it yet.
    Values: P9-P12 below.
42. All Phase 1 parameters live in the `constraints` section of the config, which is excluded from
    the TVCSI table cache key (`sweep._table_cfg`): no decision depends on them, so the cached
    tables stay valid. Phase 1 changes no decision and no previously reported number (checked
    against the committed `default_runs.csv`: largest difference 2.2e-16).

## Phase 2: measured task quality (faculty review, Nov 2026)

43. **Q is measured, not assumed.** The logistic Q of spec 3.6 (items 10-11) had made-up
    thresholds. It is replaced by Q_type(gamma) = 1 - BLER_type(gamma): the probability that the
    user's semantic message (its payload, P6-P8) is delivered intact in the slot over its PRBs (P4),
    measured with a standard-compliant 5G NR link-level simulation (`scripts/measure_q.py`, values
    P13-P22). This is the PHY abstraction (link-to-system mapping) of system-level simulators.
    The old model stays available as `quality.model: logistic`, bit-identical to the old code
    (test `test_logistic_model_is_the_legacy_formula`).
44. **AWGN at the slot SINR is exact here.** The simulated channel has one coefficient vector per
    slot and is flat across the band, so every resource element of a slot sees the same SINR and no
    effective-SINR mapping (EESM/MIESM) is needed: BLER over AWGN at that SINR is the right curve.
45. **Q now depends on K.** The carrier is shared, so each user's PRBs (P4) shrink as K grows and a
    higher MCS is needed for the same message. A curve is measured for every (type, PRBs) the
    default point and the K sweep use (18 curves, `data/q_curves.csv`); fig 2 (utility vs K)
    therefore includes the effect of sharing the carrier.
46. **Carrier re-chosen** (P2, P22): no 15 kHz carrier met the 90% target; the largest, 50 MHz,
    reaches it in 84.4% of critical slots (validation seeds) and was kept without tuning any other
    parameter. The remaining infeasible slots are reported by the Phase 1 oracle in every result.
47. **The TVCSI table does not depend on Q.** It stores beam regret, and its fitting policies never
    read Q, so `quality`, `radio` and `link_level` are excluded from its cache key
    (`sweep._table_cfg`; test `test_table_does_not_depend_on_quality_model`).
48. **`scripts/calibrate_q.py` no longer tunes anything.** Its saturation / sensitivity checks were
    written to tune the logistic thresholds; a measured curve has no thresholds to tune. It now
    plots the measured curves against the legacy ones and the true best-beam SINR spread, and
    prints where each curve crosses Q_min.
49. **The Eq. 39i check keeps the Shannon bound** (P5). Q already includes delivery with a real
    MCS, so the check is now a weaker, MCS-free bound; `constraints.payload_check_scale` scales the
    payload in this check only (sensitivity), never the Q curves.
51. **Measured curves and their cross-check** (`data/q_curves.csv`, `data/q_validation.csv`,
    `figures/q_curves_validation.png`). All 18 curves measured on an RTX 4060 GPU (63 min).
    SNR at BLER 0.1, K = 2/4/6/8/12/16: critical -1.26/2.46/5.23/8.14/11.92/16.74 dB; normal
    -5.17/-4.79/-2.55/-1.63/0.48/2.39 dB; background -5.16/-5.95/-5.84/-5.83/-5.60/-5.62 dB.
    NVIDIA's pre-computed tables (Sionna SYS) cover 10 of them (MCS >= 3): 6 agree within 0.16 dB,
    2 within 0.5 dB, and the two MCS 4 curves differ by 0.73 and 0.99 dB, our curve needing more
    SNR. To decide which is plausible, both were compared with the finite-blocklength limit for the
    same block (normal approximation, Polyanskiy-Poor-Verdu 2010): our curves sit a consistent
    1.0-2.1 dB above it (1.0-1.2 dB at QPSK, 1.5-2.1 dB at 16/64QAM), the usual gap of practical LDPC
    decoding; the tables' MCS 4 entries sit only 0.27-0.48 dB above it, implausibly close, and those
    tables are coarse in SNR and simulated only up to 2000-bit code blocks (ours: 2616 and 5392).
    The measured curves are kept.
52. **The link-level tools are optional.** The simulator reads the committed `data/q_curves.csv`;
    only re-measuring needs PyTorch and Sionna (`pip install -e .[linklevel]`).

## Parameter values: what, why, how, where

Every number added in Phase 1, in the same format. "Source" is the document the value comes from;
3GPP specifications were read through the itecspec.com mirrors listed in each entry. Where no
standard value exists, this is said explicitly and the value is stress-tested instead
(`results/sensitivity.md`, produced by `scripts/sensitivity.py`).

**P1. Reliability tolerance eps (`constraints.eps_violation`)**
* What: epsilon_k in Eqs. 34 / 39j, the largest allowed probability that a critical user's task
  quality falls below Q_min.
* Value: **0.10** for every critical user.
* Why: the PDF defines eps_k only as a symbol. 0.1 is the error-rate target 5G itself uses for normal
  (eMBB) traffic, so it is the natural "reliable enough" level for a standard 5G link.
* How: taken directly. The stricter URLLC value 1e-5 was considered and rejected: a critical user
  has about 1900 evaluated slots per seed, so failure rates below about 1e-3 cannot be measured.
* Source: 3GPP TS 38.214, clause 5.2.2.1 (CQI is the highest index whose PDSCH transport block error
  probability does not exceed 0.1 for CQI tables 1, 2, 4; 0.00001 for table 3) -
  https://www.itecspec.com/3gpp/38.214/s/5.2.2.1 ; URLLC 1e-5 for 32 bytes in 1 ms: TR 38.913
  clause 7.9 - https://itecspec.com/3gpp/38.913/s/7.9
* Stress test: eps = 0.01, 0.05, 0.10. Conclusion unchanged (Proposed meets 39j in 0% of seeds at
  every eps).

**P2. Carrier size (`radio.carrier_prbs`; was `constraints.payload.carrier_prbs`)**
* What: number of physical resource blocks (PRBs) in the carrier, shared by all users.
* Value: **270 PRBs = 50 MHz at 15 kHz SCS** (Phase 2). Phase 1 used 106 PRBs (20 MHz).
* Why: 15 kHz is the only subcarrier spacing whose slot is 1 ms, the slot length the simulator
  already uses (`channel.Ts_s = 1e-3`). Phase 1 took the 20 MHz row. The Phase 2 link-level check
  showed that on 20 MHz a critical message (1200 bytes over 13 PRBs) does not fit in one slot at
  ANY MCS of the PDSCH table, so the carrier was re-chosen by the rule in P22: the result is 50 MHz,
  the largest 15 kHz carrier.
* How: `scripts/measure_q.py select` (P22), result in `data/carrier_selection.csv`.
* Source: 3GPP TS 38.101-1, Table 5.3.2-1 (15 kHz row: 20 MHz 106, 25 MHz 133, 30 MHz 160,
  40 MHz 216, 50 MHz 270 PRBs) - https://itecspec.com/spec/3gpp-38-101-1-5-3-ue-channel-bandwidth .
  Band n78 (3.5 GHz, our carrier frequency) lists 10-100 MHz channel bandwidths in 3GPP band
  combination tables; the per-SCS row of TS 38.101-1 Table 5.3.5-1 for n78 was not read directly.

**P3. PRB width (`radio.prb_hz`)**
* What: bandwidth of one PRB.
* Value: **180 kHz**.
* Why / how: an NR PRB is 12 consecutive subcarriers; 12 x 15 kHz = 180 kHz.
* Source: 3GPP TS 38.211 (definition of a resource block, 12 subcarriers).

**P4. Bandwidth per user W_k (derived, `config.user_bandwidth_hz`)**
* What: the bandwidth one user's semantic payload is sent over (Eq. 39i and the Q curves).
* Value: **floor(270 / K) PRBs**, at least 1. At K = 8: 33 PRBs = **5.94 MHz**
  (K = 2, 4, 6, 12, 16: 135, 67, 45, 22, 16 PRBs).
* Why: the simulator already assumes one user per resource block with no interference
  (item 3), i.e. orthogonal, equal sharing. Equal sharing is the simplest allocation that adds
  no new decision variable.
* How: derived from P2 and P3. Leftover PRBs (270 - 8 x 33 = 6 at K = 8) are left unused.
* Source: derived; no further assumption.

**P5. Deliverable bits Delta R_k(t) (formula, `link.deliverable_bits`)**
* What: right-hand side of Eq. 39i, the bits user k can receive in one slot.
* Value: **W_k Ts log2(1 + gamma_k)**.
* Why: the Shannon capacity of the interference-free link is the largest rate any modulation and
  coding scheme could carry, so a payload that fails this test fails for every real MCS. This keeps
  the Phase 1 check free of any MCS assumption.
* How: standard capacity formula; W_k from P4, Ts = 1 ms, gamma_k = the SINR the scheme realised.
* Source: Shannon (1948). Since Phase 2, Q itself measures delivery with a real 5G NR MCS, so this
  check is an MCS-free upper bound kept for Eq. 39i; Q is always the stricter of the two.

**P6. Critical payload (`radio.payload_bytes.critical`)**
* What: semantic payload S_k of a critical user per slot.
* Value: **1200 bytes = 9600 bits**.
* Why: critical messages report events (e.g. a hazard). The V2X requirement for event-triggered
  messages allows payloads of up to 1200 bytes; the upper end is used because the constraint
  must hold for the largest message.
* How: taken directly. Shannon SINR for Eq. 39i at K = 8: 9600 / (5.94 MHz x 1 ms) = 1.62 bit/s/Hz,
  gamma >= 2^1.62 - 1 = **3.2 dB** (Phase 1, 20 MHz: 12.1 dB). The real 5G link needs more: see P13.
* Source: 3GPP TS 22.185, requirement R-5.2.2-002 - https://www.itecspec.com/3gpp/22.185/s/5.2.2

**P7. Normal payload (`radio.payload_bytes.normal`)**
* Value: **300 bytes = 2400 bits**.
* Why: normal messages are regular status updates, i.e. periodic broadcast messages, whose
  payload is 50-300 bytes; the upper end is used.
* How: taken directly. Shannon SINR at K = 8: 0.40 bit/s/Hz, **-4.9 dB** (Phase 1: 0.2 dB).
* Source: 3GPP TS 22.185, requirement R-5.2.2-001 (same link as P6).

**P8. Background payload (`radio.payload_bytes.background`)**
* Value: **50 bytes = 400 bits**.
* Why: background messages are the least informative periodic updates, so the lower end of the
  same 50-300 byte range is used.
* How: taken directly. Shannon SINR at K = 8: 0.07 bit/s/Hz, **-13.2 dB** (Phase 1: -9.0 dB).
* Source: 3GPP TS 22.185, requirement R-5.2.2-001.
* Stress test for P6-P8: every payload x 0.25, 0.5, 1, 2 in the Eq. 39i check
  (`constraints.payload_check_scale`); the Q curves keep the base sizes. Results:
  `results/sensitivity.md`.

**P9. Feedback weight lambda_F (`constraints.lambda_F`)**
* What: cost of one feedback bit per slot in the P1 objective (Eq. 39a).
* Value: **1e-3 per bit**.
* Why: **no standard or paper value exists**; the weight is a design choice that converts bits into
  utility units. It is chosen so that the penalty is noticeable but does not dominate.
* How: the default budget is 256 bits/slot, so a scheme that spends it all pays 0.256, about 15% of
  the genie's utility (1.75).
* Source: none (design choice). Stress test: 0, 5e-4, 1e-3, 2e-3.

**P10. Switching weight lambda_B (`constraints.lambda_B`)**
* What: cost of one beam switch (Eq. 38) per slot in Eq. 39a.
* Value: **0.05 per switch**.
* Why: no standard value exists (design choice).
* How: at the default point a scheme makes about 8 x 0.24 = 1.9 switches per slot, so the penalty is
  about 0.1, about 6% of the genie's utility.
* Source: none (design choice). Stress test: 0, 0.05, 0.10.

**P11. Risk weight lambda_R (`constraints.lambda_R`)**
* What: weight of CVaR_0.95 of the semantic loss (Eq. 37) in Eq. 39a.
* Value: **0.10**.
* Why: no standard value exists (design choice).
* How: CVaR is about 2.5-3.2 at the default point, so the penalty is 0.25-0.32, about 15-18% of the
  genie's utility, of the same order as the feedback penalty.
* Source: none (design choice). Stress test: 0, 0.10, 0.20.
* Stress test for P9-P11, all 36 combinations: Proposed beats Baseline in 0 of 36.

**P12. Sensitivity grids (`constraints.sensitivity`)**
* What: the values at which `scripts/sensitivity.py` re-scores the runs.
* Why: each grid brackets its default (stricter and looser, or half and double), and includes 0 for
  the lambdas so the plain-utility case is always covered.
* How: eps and lambda are re-scored from `results/default_runs.csv` (no new simulation); payload
  scales re-simulate the default point, because the payload check needs the per-slot SINR.

### Phase 2 values (measured task quality)

**P13. Task-quality model (`quality.model: measured`)**
* What: Q_k(gamma), the downstream task quality of Eq. 20.
* Value: **Q_type(gamma) = 1 - BLER_type(gamma)**, measured (`data/q_curves.csv`).
* Why: the faculty asked to replace the arbitrary logistic with something practically measured.
  A semantic message helps the task only if it arrives; the probability that it arrives in the
  slot is the transport-block success rate of the 5G data channel that carries it. Mapping SINR to
  block error rate with curves from a link-level simulation is how 5G system-level simulators
  model the physical layer ("PHY abstraction" / link-to-system mapping).
* How: `scripts/measure_q.py curves`, settings P14-P21; cross-checked against NVIDIA's
  pre-computed tables (`scripts/measure_q.py validate`, `data/q_validation.csv`).
* Source: method - S. Lagen et al., "New Radio Physical Layer Abstraction for System-Level
  Simulations of 5G Networks", IEEE ICC 2020, https://arxiv.org/abs/2001.10309 ; Sionna PHY
  abstraction, https://nvlabs.github.io/sionna/sys/tutorials/PHY_Abstraction.html .
  Tool - Sionna 2.2.0 (NVIDIA, Apache-2.0) on PyTorch 2.15 (CPU).

**P14. Physical channel: PDSCH (downlink)**
* What: which 5G channel the semantic message travels on.
* Value: **PDSCH**, single layer, PDSCH scrambling.
* Why: the gNB sends each user its semantic message on the beam it selected, i.e. on the downlink
  shared channel.
* How: `TBEncoder(channel_type="PDSCH")` and the PDSCH MCS table (P17).
* Source: 3GPP TS 38.211 / 38.212 / 38.214 as implemented by Sionna.

**P15. PDSCH symbols per slot (`radio.pdsch_symbols`)**
* Value: **12** of the 14 OFDM symbols of the slot.
* Why: the first OFDM symbols of a slot carry the downlink control channel (PDCCH CORESET); a
  2-symbol CORESET with the data channel on symbols 2-13 is a common configuration. This is a
  design choice, not a value prescribed by one table.
* How: fixed a priori (before any measurement), not tuned.
* Source: TS 38.211 (14 OFDM symbols per slot with normal cyclic prefix; PDSCH mapping type A may
  start at symbol 2 or 3).

**P16. DMRS overhead (`radio.dmrs_re_per_prb`)**
* Value: **12 resource elements per PRB per slot** (one front-loaded DMRS symbol, no data on it).
* Why: the receiver needs reference signals for channel estimation; one DMRS symbol is the minimum
  configuration, and leaving the whole symbol free of data is the common single-layer setting.
  Together with P15: 12 x 12 - 12 = **132 data REs per PRB**.
* How: fixed a priori, not tuned.
* Source: TS 38.211 clause 7.4.1.1 (PDSCH DM-RS).

**P17. MCS table (`radio.mcs_table`)**
* Value: **PDSCH MCS table 1** (QPSK to 64QAM, MCS 0-28).
* Why: table 1 is the default table (used unless 256QAM or low-SE tables are configured).
* Source: 3GPP TS 38.214 Table 5.1.3.1-1.

**P18. MCS of each curve (rule)**
* Value: **the lowest MCS index whose transport-block size holds the payload**; none fits => the
  message cannot be sent in one slot and Q = 0 at every SINR.
* Why: the message must fit in one slot; among the MCSs that fit, the lowest has the lowest code
  rate and modulation order, so it needs the least SINR. Note that the transport-block size is set by
  the MCS and the PRBs, not by the message, so small messages are padded (e.g. at K = 2 normal and
  background use the same MCS 0 block of 4104 bits).
* How: `measure_q.select_mcs`, TBS from TS 38.214 5.1.3.2 (`sionna.phy.nr.utils.calculate_tb_size`).
  At K = 8 (33 PRBs): critical MCS 15 (16QAM, R 0.60, TBS 10504, 2 code blocks), normal MCS 4
  (QPSK, R 0.30, TBS 2600), background MCS 0 (QPSK, R 0.12, TBS 1032). Test
  `test_selected_mcs_is_the_lowest_that_fits` re-derives every choice.
* Source: TS 38.214 clauses 5.1.3.1-5.1.3.2.

**P19. Channel decoder (`link_level.bp_iterations`)**
* Value: **LDPC belief propagation, 20 iterations, exact ("boxplus-phi") check-node update**.
* Why: 20 iterations is Sionna's default and a common choice; the exact update avoids the
  0.2-0.5 dB loss of min-sum approximations, so the curve is not pessimistic.
* Source: Sionna `TBDecoder` / `LDPC5GDecoder` defaults.

**P20. Monte-Carlo precision (`link_level.*`)**
* Values: SNR step **0.25 dB**; each point runs until **100 block errors or 1000 blocks** (batches
  of 100); a curve ends at the first point with BLER **< 2e-3**; random seed **7** (+ curve index).
* Why: the LDPC waterfall falls from BLER ~1 to ~0 within 1-2 dB. What matters is where the curve
  lies in SNR: with 100 errors the BLER is known to about +-20%, which moves a point on such a steep
  curve by only a few hundredths of a dB. 1000 blocks resolve Q to 0.001. On this CPU (no GPU used)
  100 blocks of a critical message take 5-20 s, so finer settings would cost days.
* How: "locate then refine" (`measure_q.measure_curve`): 1 dB steps of one batch from the Shannon
  SNR until fewer than half the blocks fail, then 0.25 dB steps from 1.5 dB below that point,
  walking down until BLER >= 0.99 and up until BLER < 2e-3.
* Evidence: three independent measurements of the same MCS 0 setup (1032-bit TB over 4356 QPSK
  symbols, different seeds) put BLER = 0.1 at -5.84, -5.85 and -5.90 dB: they agree within 0.06 dB.
* Robustness: every finished curve is saved at once to `data/q_curves_parts/` (file name =
  curve key + fingerprint of these settings) and a rerun skips saved curves. Each curve's seed is
  `seed + crc32(curve key)`, so a resumed run gives exactly the numbers of an uninterrupted one
  (checked: a deleted curve re-measured byte for byte identical).
* Source: design choice (standard Monte-Carlo practice).

**P21. Q outside the measured range (`quality.curve_on_grid`)**
* Value: **Q = 0 below the lowest measured point, Q = 1 above the highest**; in between linear
  in dB on a 0.01 dB grid, made non-decreasing.
* Why: the lowest point has BLER >= 0.99 and the highest BLER < 2e-3 (P20), and BLER can only fall
  as SINR rises, so 0 / 1 are the values the curve tends to. Making it non-decreasing removes
  Monte-Carlo wiggles, which the true curve cannot have.
* Source: design choice.

**P22. Carrier selection rule (`link_level.carrier_candidates`, `feasibility_target`, `select`)**
* What: how P2 was chosen.
* Value: candidates **20, 25, 30, 40, 50 MHz** (106, 133, 160, 216, 270 PRBs, all at 15 kHz SCS);
  target **90%**: the smallest carrier on which a critical message, sent with the TRUE best beam,
  reaches Q >= Q_min in at least 90% of critical slots at the default point. If none does, the best
  one is taken.
* Why: if the target is impossible even with perfect beamforming (Phase 1: 95.7% of critical slots
  were infeasible), no scheduler can matter; the deployment must be dimensioned so the target is
  normally achievable. 90% leaves room for deep fades. The rule was fixed before measuring.
* How: `scripts/measure_q.py select`, coarse curves (0.5 dB, 50 errors / 500 blocks, stop at
  BLER 1e-2), evaluated on the **validation seeds** (2000-2004), never the test seeds. Result
  (`data/carrier_selection.csv`): 20 MHz no MCS fits (0%); 25 MHz 1.9%; 30 MHz 23.3%;
  40 MHz 67.8%; **50 MHz 84.4%**. No candidate reached 90%, so 50 MHz (the largest 15 kHz carrier)
  was taken. No other parameter was changed to close the gap.
* Source: candidates from TS 38.101-1 Table 5.3.2-1; the rule and the 90% target are design choices.

**P24. Code rates below the mother-code rate: repetition (`measure_q.Chain`)**
* What: how low-rate MCSs on large allocations are simulated. 5G NR LDPC has two mother codes
  (TS 38.212 7.2.2 picks one from the TB size A and the MCS rate R): base graph 1 (large blocks,
  lowest rate 1/3) and base graph 2 (small blocks, lowest rate 1/5). Sionna's encoder stops at these
  rates; several of our curves need less (e.g. background at K = 8: 1032 bits over 8712 coded bits =
  0.12 on BG2; critical at K = 2: rate 0.30 on BG1).
* Value: the TB is encoded at the mother-code rate per code block (`base_coded_bits`), the remaining
  coded bits repeat those bits cyclically, and the receiver adds the LLRs of all copies before
  decoding. Used by 9 of the 18 curves: critical K = 2 (BG1, x1.10), normal K = 2, 4 and background
  at every K (BG2, x1.29-1.71). The base graph Sionna picks is checked against TS 38.212 7.2.2 for
  every curve (`Chain` raises on a mismatch).
* Why: this is how 5G NR itself reaches rates below 1/5: rate matching reads its circular buffer
  past the end and repeats bits (TS 38.212 clause 5.4.2). Repetition only occurs at MCS 0-4, all
  QPSK, which over AWGN acts on every bit independently, so the order and scrambling of the copies
  do not change the result (`Chain` refuses repetition with any other modulation).
* History: the first version used 1/5 for both graphs; the BG1 curve (critical, K = 2) then failed
  at start-up. Since then every run starts with a preflight that builds all 18 chains and checks
  0/8 errors at 40 dB and 8/8 at -20 dB.
* Approximation: 3GPP first sends the parity bits that lie between Sionna's 1/5 limit and the end of
  its circular buffer (at most 4 x the filler bits, ~5% of the codeword) before repeating; repeating
  earlier makes our curves very slightly pessimistic. Sanity check: the MCS 0 curve reaches BLER 0.1
  1.6 dB above the Shannon limit of its information rate, normal for a ~1000-bit LDPC block.
  NVIDIA's bundled tables start at MCS 3, so the MCS 0-2 curves cannot be cross-checked against them.
* Source: TS 38.212 clause 5.4.2 (rate matching, circular buffer); Sionna `LDPC5GEncoder`
  (rejects rates below 1/5).

**P23. Q_min (`quality.q_min`), unchanged**
* Value: **0.8**, from the original task spec (Eq. 34). With measured Q it means **BLER <= 0.2**:
  a critical message must get through at least 4 times out of 5 in a slot to count as served.

