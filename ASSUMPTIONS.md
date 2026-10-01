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
