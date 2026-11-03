# Final report: task-value-aware CSI feedback (stages S3–S7)

**Date:** 2026-11-03. **Branch:** `research/s3-s7-maruthi` (PR into `research/stages-s0-s7`).
**Final configuration:** `configs/final.yaml` (= `configs/default.yaml` + trained models + dated output folders).
**Final results:** `results/final_20261103/` (`summary.md`, CSVs) and `figures/final_20261103/`.
Every number below is on the **test seeds 1000–1009**, never used for training or tuning.

---

## 1. The answer in one paragraph

The faculty review found that our method lost to simple algorithms, that critical users failed
97–98 % of the time, that the models were placeholders, and that the quality curve was made up. Phase 2
(before this branch) replaced the quality curve with one measured on a 5G NR link. This branch fixes
the method itself: **the scheduler now learns what feedback is worth** (S4), **is compared with the
five faculty baselines with proper statistics** (S5), **protects critical users with a virtual queue**
(S5), and **uses a trained CSI codec** (S6). Result: **the Proposed scheme beats the best of eight
simple baselines at 28 of 29 operating points** (every point of the five sweeps except "unlimited
budget", where all schemes tie), on 9–10 of 10 seeds each time, with Holm-corrected significance. It
reaches **98 % of a genie that knows the true channel**. It keeps **the worst critical user's
avoidable failures below the 10 % target** wherever the budget is not tiny, and it reaches 90 % of
full-feedback utility with **2.4–3.4× fewer feedback bits** than the periodic baseline.

## 2. What was wrong, what was done

| Faculty finding | Fix | Stage | Status |
|---|---|---|---|
| Proposed loses to simple algorithms | learned per-mode quality estimator replaces the mean-regret plug-in (`tvcsi.py` → `estimator.py`) | S4 | ✅ fixed: 1.25 → 2.77 utility |
| need stronger baselines (age-, uncertainty-, feedback-gain-aware) | the five faculty baselines `fill_*` + seed-paired Holm statistics | S5 | ✅ all beaten |
| critical users violate 97–98 % | measured Q (Phase 2) + feasibility split (Phase 1) + virtual-queue risk scheme | Ph. 1–2, S5 | ✅ avoidable violations 5 % (target 10 %) |
| placeholder encoder / decoder / predictor ("why not AI?") | trained 4-bit progressive codec; trained GRU evaluated (fails its gate, so persistence kept) | S6 | ✅ codec / ⚠️ GRU |
| arbitrary logistic Q | Q = 1 − BLER measured on a 5G NR PDSCH link | Phase 2 | ✅ (before this branch) |
| missing constraints / feasibility oracle | true best-beam oracle; Eq. 39a/39i/39j as metrics | Phase 1 | ✅ (before this branch) |
| plumbing for all of the above | gNB-side state for the schedulers, bit-identical decisions | S3 | ✅ |

## 3. Main results (final configuration, test seeds)

### 3.1 Default point (K = 8, 30 km/h, 15 dB, 256 bits/slot)
| Scheme | Utility | Bits/slot | Critical violations, avoidable (feasible slots) | Worst critical user ≤ 10 %? |
|---|---|---|---|---|
| BestBeam oracle (perfect CSI, free) | 2.844 | – | 0 | – |
| Exact (genie scheduler) | 2.833 | 20 | 1.1 % | 10/10 seeds |
| full_feedback (4× the budget) | 2.798 | 1024 | 4.0 % | 10/10 |
| **Proposed** | **2.774** | 256 | **5.4 %** | **10/10** |
| **Proposed_risk** | 2.773 | 256 | **5.2 %** | **10/10** |
| Baseline (spec periodic, 64-bit mode) | 2.729 | 256 | 10.5 % | 1/10 |
| fill_importance (faculty) | 2.684 | 256 | 19.4 % | 0/10 |
| fill_age (faculty) | 2.674 | 256 | 15.8 % | 0/10 |
| fill_periodic (faculty) | 2.622 | 256 | 20.7 % | 0/10 |
| Proposed_plugin (old method) | 1.250 | **0** | 72.1 % | 0/10 |

Proposed − each baseline (seed-paired, Holm): +0.045 ± 0.007 (periodic) … +0.95 (fill_channel),
**10/10 seeds in all 8 comparisons**, Holm p ≤ 6e-7. 16.2 % of critical slots are physically
infeasible (even the true best beam misses Q_min); no scheme can avoid those, which is why the table
reports the avoidable part.

### 3.2 All sweep points (`research/logs/S7/best_baseline_per_point.csv`)
Proposed vs the **best** simple baseline at each point:

| Sweep | Points where Proposed wins significantly | Gain over the best baseline |
|---|---|---|
| budget 0.05 … 1.0 | 6 / 7 (tie at 1.0) | +4.6 % at 0.05, +3.5 % at 0.1, … +0.2 % at 0.75 |
| users K = 2 … 16 | 6 / 6 | +0.6 % (K = 2) … +1.7 % (K = 8) |
| speed 3 … 60 km/h | 6 / 6 | +0.3 % (3 km/h) … **+4.1 % (60 km/h)** |
| critical share 0 … 0.75 | 5 / 5 | +0.6 % … **+2.8 %** (more critical users → larger gain) |
| SNR 5 … 25 dB | 5 / 5 | **+8.8 % (5 dB)** … +0.2 % (25 dB) |

The gain grows exactly where the idea predicts: when feedback is scarce, channels change fast,
links are noisy, or more traffic is critical. It shrinks to ≈ 0 when feedback is plentiful.

* **Feedback savings (fig 10):** to reach 90 % of full-feedback utility, Proposed needs a budget of
  0.050 (≤ 30 km/h) to 0.082 (60 km/h) × K·C_3; the periodic baseline needs 0.166–0.196, i.e.
  **2.4× (60 km/h) to 3.4× (≤ 20 km/h) more bits**.
* **Reliability (fig 16):** the worst critical user's avoidable violation rate stays below 10 % for
  Proposed at every speed and for budgets ≥ 0.2. Periodic exceeds 10 % from 30 km/h on, the faculty
  baselines from 10–20 km/h on.
* **Objective of P1 (Eq. 39a, `results/final_20261103/sensitivity.md`):** over all 36 combinations
  of the penalty weights lambda_F, lambda_B, lambda_R, Proposed beats the periodic baseline in all 36
  and is the best implementable scheme in all 36.
* **Greedy = DP:** the greedy solver is within 0.003 % of the exact DP on the same estimates, and
  ~0.1 ms per slot decision at K = 8 (fig 14).

## 4. Comparison with the faculty implementation (main track, S1–S2)

| | Faculty TaskVoCSI_v2 (main track) | Our side track (this branch) |
|---|---|---|
| System | M = B = 16, 5 RBs for 8 users, joint feedback + service, T = 140 | M = B = 32, every user served, T = 2000, measured 5G NR Q |
| Default: proposed vs best baseline | +1.5 % (digits), +3.9 % (surrogate) | **+1.7 %** (periodic), +3.4 % vs best faculty baseline |
| Seeds won vs best baseline | 15–18 / 20 (digits), 20 / 20 (surrogate) | **10 / 10** |
| Share of genie utility | 91–92 % | **98 %** |
| Estimator gain correlation, test | 0.12–0.20 | 0.13–0.28 (Proposed trajectory) |
| Risk handling | soft penalty, no measurable effect | virtual queue, −23…−48 % worst-user violations at hard points (S5) |
| Trained GRU vs persistence | tie (0.643 vs 0.634 accuracy) | tie at the default point (5/10 seeds); helps at 3 / 60 km/h and 5 dB |
| Sweeps | 5 seeds, exploratory | 10 seeds, paired, Holm-corrected |

**Where the gain comes from differs.** In the faculty world the gain appears only when users compete
for the 5 resource blocks (S2 §5). Our side track serves every user, so the gain here comes from
**spending feedback bits on the users whose beam is most likely stale and whose message is most
valuable**, which is the TVCSI idea of the original formulation (Eq. 32–33). Both tracks agree that
the trained GRU does not beat persistence on these synthetic channels, and that importance weighting
matters (smaller in our world: −0.007 utility without it, S4).

## 5. Limitations (stated plainly)

1. **Synthetic channels.** The AR(1)/random-walk channel is close to memoryless given the last
   report, which is why no learned predictor beats persistence. Real (ray-traced or measured)
   trajectories may reward the GRU; `load_deepmimo` is still a stub.
2. **Only the feedback sub-problem of P1** is optimised: one user per resource block, no power
   allocation, single semantic representation; Eq. 39a/39i/39j are reported as metrics. The
   faculty's joint feedback + service decision is not part of the side track.
3. **Reliability is not guaranteed.** The virtual queue is a soft mechanism. At budget 0.1 every
   critical user meets the 10 % target in 6 of 10 seeds with Proposed_risk (3 / 10 with Proposed); at
   budget 0.05 in none. The chosen queue weight (0.2) was the only admissible value on validation
   data, i.e. fragile (ASSUMPTIONS P31).
4. **Estimator distribution shift.** Trained on random-exploration data, its gain correlation along
   the real Proposed trajectory (0.13–0.28) is lower than on validation (0.30–0.39).
5. **One training run** per model (fixed seeds): variability due to training randomness is not measured.
6. Uplink errors, headers and delay of the feedback reports are not modelled.

## 6. How to reproduce

```bash
python -m venv ../venvs/tvcsi_combined          # outside the repository and outside OneDrive
../venvs/tvcsi_combined/Scripts/python -m pip install -e . pytest scikit-learn joblib
../venvs/tvcsi_combined/Scripts/python -m pytest -q                       # 70 passed, 1 skipped (sionna)
python scripts/fit_estimator.py                                           # S4 estimator (default config)
python scripts/train_models.py                                            # S6, needs torch (CPU); weights are committed
python research/logs/S7/make_final_config.py                              # configs/final.yaml
python -m tvcsi.sweep --config configs/final.yaml                         # ~40 min on 8 cores
python -m tvcsi.plots --config configs/final.yaml
python scripts/make_summary.py --config configs/final.yaml
python scripts/sensitivity.py --config configs/final.yaml
```
The estimators are rebuilt automatically (deterministically) on first use if their cache files are
missing; the trained model weights (`data/models/s6_models.npz`) are in git.

## 7. Where to look
* Checkpoints: `research/checkpoints/S3_…S7_checkpoint.md`; evidence: `research/logs/S3 … S7/`.
* Every modelling choice and parameter value: `ASSUMPTIONS.md` items 53–74, P25–P35.
* Figures: `figures/final_20261103/` — fig01–fig16 (main schemes), supp01–05 (ablations),
  base01–05 (Proposed vs all simple baselines), fig16 (per-user reliability).
