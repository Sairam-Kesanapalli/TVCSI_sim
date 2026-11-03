# S2 — Analysis of the faculty implementation (TaskVoCSI_v2) results

Source data: the two full runs of S1 (`faculty_TaskVoCSI_v2/results/full_digits_20261103`,
`…/full_surrogate_20261103`), unmodified faculty code. Tables and figures: `S2_analysis/`
(produced by `S2_analysis/s2_faculty_analysis.py`; tables T1–T7, figures F1–F6). "digits" = the
measured digit-classification quality table; "surrogate" = the sigmoid quality model. Utility is the
importance-weighted task quality per slot (dimensionless). Baselines = periodic, age, uncertainty,
importance, channel.

## 1. Summary

1. **CTVC-DP beats all five baselines, robustly but modestly.** Default point, 20 held-out seeds:
   digits 3.598 vs best baseline 3.544 (importance), **+1.5 %**; surrogate 3.526 vs 3.392, **+3.9 %**.
   Against every baseline the seed-paired difference is significant after Holm correction
   (digits p ≤ 0.015, surrogate p ≤ 3e-7). CTVC-DP wins in 15–18 of 20 seeds on digits and in
   **20 of 20** on surrogate against every baseline.
2. **The gain appears only when not every user can be served.** With K = 4 users and 5 resource
   blocks (everyone served) the gain over the best baseline is +0.005 (digits) / +0.020 (surrogate);
   with K = 8 and 12 (5 served) it is +0.06…+0.19. Most likely mechanism: CTVC-DP chooses feedback and
   service *jointly*, so it does not spend bits on users it will not serve, while the baselines
   choose feedback first. **Our side track serves every user, so this mechanism does not exist there**
   (hypothesis, see §5 for the evidence and its limits).
3. **Importance weighting is essential** (removing it costs 0.64–0.71, −18 to −20 %, worse than every
   baseline); **the risk penalty does nothing measurable**; **exact DP = greedy**.
4. **The learned estimator works, weakly.** Test MAE 0.11–0.16; gain correlation (does it rank the
   value of feedback correctly?) 0.12–0.20 on test seeds, lower than on validation (0.21–0.27).
   Clearly better than our plug-in (≈ 0), far from accurate.
5. **The trained GRU is no better than persistence** (top-1 beam accuracy 0.643 vs 0.634, regret 0.193
   vs 0.195). The trained codec's highest resolution is not the best (regret 0.093 vs 0.086).
6. **Large headroom remains:** the genie (true outcomes) gets 3.89 using only 16–44 bits/slot; CTVC-DP
   spends ~140 of 144 bits/slot and reaches 90–92 % of the genie.

## 2. Setup reminder (what the faculty world is)
M = B = 16 ULA antennas/DFT beams, K = 8 users, **5 resource blocks** (≤ 5 users served per slot),
T = 140 slots (+10 warm-up), feedback modes 0/24/48/96 bits, budget 144 bits/slot, SNR 20 dB, 25 km/h,
critical share 0.25. Codec + GRU trained on seeds 101–106; calibrator (ExtraTrees) trained on the
same, validated on 201–202; test seeds 301–320. Quality: sigmoid surrogate or a digit-classification
accuracy table measured over *uncoded* BPSK. Sweeps: 5 seeds per point, models frozen (trained at the
default point), i.e. they measure robustness to an unadapted shift.

## 3. Default point (T1, T2, T3; F1, F2, F5)

| | digits | surrogate |
|---|---|---|
| CTVC-DP | 3.598 ± 0.064 | 3.526 ± 0.075 |
| best baseline (importance) | 3.544 ± 0.071 | 3.392 ± 0.083 |
| CTVC-DP − baseline (paired), range over 5 baselines | +0.054 … +0.112 | +0.133 … +0.266 |
| seeds won vs each baseline | 15–18 / 20 | 20 / 20 |
| genie | 3.892 (16 bits/slot) | 3.886 (44 bits/slot) |
| CTVC-DP share of genie utility | 92.5 % | 90.7 % |

Order of the baselines: importance and channel are strongest on digits, importance and age on
surrogate; uncertainty is weakest on surrogate. All baselines spend the full 144 bits; CTVC-DP spends
137–143. Outage (critical users below Q_min, or unserved) is ~0.43–0.47 for every scheme and 0.37 for
the genie, while the best-beam feasibility floor (`ideal_outage`) is 0.0002: in this world critical
failures come from scheduling (only 5 of 8 users served) and from estimates, not from physics.

## 4. Statistical validity
* **Unit of replication = seed (20 independent channel/importance realisations); differences are
  seed-paired**; slots within a seed are not treated as independent. Holm correction over 9 tests.
  This is the right design, and with 20 seeds all five baseline comparisons survive correction (the
  faculty's 10-seed notes reported that two did not).
* **Effect sizes are small relative to the spread between seeds** (per-seed CI of a scheme's mean
  ≈ ±0.07), but pairing removes most of it: paired CIs are ±0.03–0.06 on digits, ±0.03–0.06 on
  surrogate. On digits, CTVC-DP still loses in 2–5 of 20 seeds against each baseline (worst −0.10 vs
  channel).
* **Sweeps are exploratory**: 5 seeds per point, unpaired CIs; no sweep point has non-overlapping CIs
  (T4), so no sweep difference should be claimed as significant.
* Single training run (one training seed set): variability due to training randomness is not measured.

## 5. Where does the gain come from? (T1, T4, T7; F3)
* **Not from the risk term**: CTVC-DP vs no-risk: utility −0.007 ± 0.016 (digits) / +0.007 ± 0.017,
  outage +0.004 ± 0.006 / +0.002 ± 0.007, CVaR +0.04 ± 0.05 / +0.02 ± 0.04 — all indistinguishable
  from zero. The penalty (0.12 × predicted outage probability) is small next to v·Q differences.
* **Not from the exact optimiser**: greedy = DP (−0.012 ± 0.015, −0.004 ± 0.014).
* **Largely from importance**: no-importance loses 0.64–0.71 and falls below every baseline. (The
  baselines' service decision also uses importance, v·Q̂.)
* **From joint feedback + service decisions (hypothesis)**: the gain over the best baseline is
  ≈ 0 when K = 4 ≤ 5 resource blocks (every user is served; CTVC-DP 2.369 vs 2.364 digits, 2.294 vs
  2.274 surrogate) and appears once users compete for service (K = 8: +0.06 / +0.19; K = 12: +0.11 /
  +0.18). The baselines pick feedback before they know who will be served and may spend bits on users
  who are then not served; CTVC-DP prices both together. *Limits of this evidence*: 5 seeds per sweep
  point, K also changes the number of users to differentiate, and the per-user bit spending on
  unserved users is not logged by the faculty code. A direct check needs an instrumented re-run
  (feasible without modifying their code, by importing `taskv2` from a separate script).
* **Strongest relative gains** at low SNR (digits 12 dB: +0.23, surrogate +0.15) and mid budgets
  (96 bits: +0.15 / +0.23); **no gain** at a 48-bit budget (both runs, −0.01) and at 28 dB on digits
  (−0.01): when feedback is scarce or rarely needed, valuation matters little.

## 6. Estimator and learned models (T5, T6; F6)
* **Calibrator** (ExtraTrees per-mode expected quality + outage): validation MAE 0.117 / 0.164,
  outage Brier 0.075 / 0.096, gain correlation 0.21–0.27; on test seeds MAE 0.113 ± 0.023 /
  0.159 ± 0.026, gain correlation 0.12–0.20 with large spread between seeds (sd 0.05–0.11). It ranks
  feedback value only weakly, and generalises somewhat worse than validation suggests. It is the
  component that directly addresses our plug-in flaw (our gain correlation ≈ 0).
* **Inference cost**: calibrator ~20 ms per call, GRU ~2.5 ms, DP 1.3 ms, greedy 3.6 ms (CPU). The
  faculty code batches all schemes into one calibrator call per slot; without batching, our
  2000-slot × ~10-scheme simulations would be ~25× slower.
* **GRU vs persistence**: accuracy 0.643 vs 0.634, regret 0.193 vs 0.195 — no meaningful gain from
  the trained predictor. **Codec**: accuracy 0.747 / 0.783 / 0.790 for 24 / 48 / 96 bits, regret
  0.113 / 0.086 / 0.093 (the highest resolution is not the most accurate in regret: imperfect
  progressive training). The same networks serve both runs (training does not depend on the quality
  model), so these diagnostics are identical across runs.

## 7. Limitations of the faculty evaluation (as the faculty's own README also notes)
Synthetic channel; sigmoid or uncoded-BPSK quality (no coded 5G link); short trajectories (140
slots); a single training run; sweeps with 5 seeds and frozen models; the risk mechanism is a soft
penalty with no effect; reliability is not guaranteed; RB-limited service (5 RBs) is a modelling
choice that differs from our side track (every user served).

## 8. Implications for the side track (S3–S7)
1. **Expectations.** The learned estimator should lift our Proposed from its current failure mode
   (0 bits/slot, utility 1.25 vs periodic 2.73) substantially. Beating strong baselines may be harder
   than in the faculty world, because our side track has no service-selection decision, which is
   where the faculty gain appears. Report whatever results come out.
2. **S4 (estimator).** Port the per-mode expected-quality regression; track **test** gain
   correlation and MAE, not only validation; batch inference across schemes per slot (runtime);
   train on train seeds 3000–3019, validate on 2000–2004; keep the plug-in as an ablation.
3. **S5 (risk).** Do not copy the soft penalty as is: it is inert in the faculty world. Use a
   calibrated penalty or a virtual-queue (Lyapunov) weight on critical users, and report against the
   Phase 1 feasibility split (our infeasible share is 16 %, theirs ≈ 0). Adopt the five baselines and
   the seed-paired Holm comparison for reporting.
4. **S6 (codec, GRU).** Do not assume a gain: the faculty GRU matches persistence. Gate S6 on the
   diagnostic "GRU beats persistence on held-out seeds"; if it does not, report it and keep
   persistence.
5. **Optional extension.** If we want the faculty's main gain mechanism, the side track would need a
   service-selection (RB-limited) decision; that changes the problem definition and is out of the
   agreed S3–S6 scope.
6. **Optional check.** An instrumented re-run of the faculty code (separate script importing `taskv2`,
   logging bits spent on unserved users) would confirm or refute the §5 hypothesis.

## 9. Files
`S2_analysis/tables/T1_default_summary.csv` … `T7_risk_effect.csv`; figures
`S2_analysis/figures/F1_default_utility.png` (default utilities), `F2_paired_forest.png` (paired
differences), `F3_sweeps.png` (sweeps), `F4_utility_vs_bits.png`, `F5_per_seed_diff.png`,
`F6_calibration_diagnostics.png`; full console record `S2_analysis/s2_run.log`.
