# Checkpoint S5: five faculty baselines, risk handling for critical users, paired Holm statistics

**Status: COMPLETE (2026-11-03).** Next: S6 (trained ProgressiveCodec + BeamGRU).

Self-contained together with `research/HANDOFF.md`, `S4_checkpoint.md` and `ASSUMPTIONS.md` items 64–67, P30–P32.

## 1. In one paragraph (beginner version)
The faculty asked for stronger simple competitors. S5 adds their five: round-robin, oldest-report-
first, least-sure-beam-first, most-important-first and weakest-link-first, each giving the biggest
report that fits. It also adds `Proposed_risk`: Proposed plus a "debt counter" for every critical
user. The debt grows only when that user's *avoidable* chance of failing is above the 10 % target,
and then pushes feedback towards that user. **Result on the test seeds: Proposed beats all eight
simple baselines at all four tested settings, on 10 of 10 seeds every time (Holm-corrected p ≤ 1.2e-5).
`Proposed_risk` trades a little utility for fewer critical-message failures.**

## 2. What was done
Work by Maruthi with an AI assistant (Claude Opus 5.5).
1. `src/tvcsi/schemes.py`: `fill_largest` (faculty filling rule), `FILL_ORDERS` → `fill_periodic`,
   `fill_age`, `fill_uncertainty`, `fill_importance`, `fill_channel`; `risk_scheme` → `Proposed_risk`.
2. `src/tvcsi/stats.py` (new): `paired_holm`.
3. `configs/default.yaml`: `risk` section (eps_target 0.10, beta 0.2, grid, calibration points);
   the six new schemes added to `schemes.supplementary` (so S7 sweeps include them).
4. Calibration of beta on VALIDATION seeds (`research/logs/S5/calibrate_risk.py`):
   v1 exposed a queue design flaw (fixed, ASSUMPTIONS 66); v2 chose **beta = 0.2**.
5. Test evaluation (`research/logs/S5/evaluate_S5.py`) at four points on TEST seeds 1000–1009.
6. `tests/test_s5_baselines_risk.py` (15 tests).
7. The plug-in table cache key also ignores the new `risk` section (committed tables stay valid).

## 3. Results
### 3.1 Calibration of beta (validation seeds 2000–2004; `01_calibrate_risk_v2.log`)
| beta | mean utility (4 points) | worst point: worst critical user's feasible violation |
|---|---|---|
| 0 (= Proposed) | 2.767 | 0.135 (budget 0.1) |
| 0.05 | 2.737 | 0.103 |
| **0.2 (chosen, only admissible)** | 2.712 | **0.0997** |
| 1.0 | 2.664 | 0.101 |

### 3.2 Test seeds, utility and critical reliability (`02_evaluate_test.log`, `eval_20261103/`)
Utility (mean of 10 seeds); in brackets: worst critical user's violation rate in feasible slots.

| Scheme | default | budget 0.1 | 60 km/h | 50 % critical |
|---|---|---|---|---|
| Exact (genie) | 2.833 (0.013) | 2.821 (0.026) | 2.827 (0.018) | 2.931 (0.015) |
| **Proposed** | **2.779** (0.058) | **2.669** (0.140) | **2.733** (0.084) | **2.825** (0.094) |
| Proposed_risk | 2.771 (**0.045**) | 2.619 (**0.103**) | 2.657 (**0.044**) | 2.733 (**0.070**) |
| Baseline (spec periodic, mode 2) | 2.726 (0.119) | 2.001 (0.831) | 2.629 (0.207) | 2.756 (0.125) |
| fill_importance | 2.697 (0.199) | 2.586 (0.264) | 2.578 (0.323) | 2.697 (0.242) |
| fill_age | 2.684 (0.161) | 2.581 (0.264) | 2.512 (0.316) | 2.681 (0.172) |
| fill_periodic | 2.633 (0.210) | 2.582 (0.245) | 2.442 (0.367) | 2.600 (0.227) |
| fill_channel | 2.035 (0.487) | 1.923 (0.645) | 1.679 (0.687) | 1.911 (0.634) |
| fill_uncertainty | 1.969 (0.574) | 1.905 (0.612) | 1.691 (0.644) | 1.838 (0.734) |

**Proposed − each of the 8 simple baselines** (Baseline, five fill_*, importance_only, snr_threshold):
positive at every point, **10/10 seeds in all 32 comparisons**, Holm-adjusted p ≤ 1.2e-5. Smallest
margins: +0.053 (default, vs Baseline), +0.083 (budget 0.1, vs fill_importance), +0.104 (60 km/h, vs
Baseline), +0.069 (50 % critical, vs Baseline).

**Proposed_risk vs Proposed:** utility −0.008 (default), −0.050 (budget 0.1), −0.077 (60 km/h),
−0.093 (50 % critical). In return, the worst critical user's violations drop by 23–48 %.
"All critical users ≤ 10 %" (reliability_met) holds in 40 % of seeds at budget 0.1 (Proposed: 0 %) and in
100 % at 50 % critical (Proposed: 80 %). Proposed_risk still beats all baselines except one case:
at 50 % critical it is 0.023 below the spec Baseline (2/10 seeds; Holm p = 0.019).

### 3.3 Observations
* The faculty's "largest mode first" baselines are weaker in our system than the spec periodic
  Baseline with mode 2: two users at 128 bits help less than four users at 64 bits.
* `fill_uncertainty` and `fill_channel` are poor here. With the persistence predictor, the least
  sure / weakest *believed* links are often users whose estimate is stale or who are hard to serve.
* At budget 0.1 no implementable scheme meets the target for every critical user in every seed;
  even the best risk weight only reaches the limit (P31 caveat).

## 4. Tests
`python -m pytest -q` → **65 passed, 1 skipped** (`logs/S5/03_pytest_full.log`). New: fill rule
equals the faculty loop for 7 budgets × 20 orders; orders; budget respected and filled; beta = 0 is
exactly Proposed; a positive beta changes decisions; no true-channel leak (tamper) for Proposed_risk
and all fill baselines; Holm procedure.

## 5. Deviations and notes
1. Faculty baselines are re-stated, not imported: the faculty package imports torch at module level.
   The test compares against a literal copy of their loop.
2. beta was chosen after one design fix (v1 → v2). The selection rule itself was not changed, and
   both runs are logged. Test seeds were used only for the final evaluation.
3. Plot styles for the new schemes and their figures come in S7.

## 6. Integrity
Faculty code untouched: committed blobs 23/23 match (`logs/S5/integrity.log`).

## 7. Files changed
`src/tvcsi/schemes.py`, `src/tvcsi/stats.py` (new), `src/tvcsi/sweep.py` (cache key),
`configs/default.yaml`, `tests/test_s5_baselines_risk.py` (new), `ASSUMPTIONS.md` (64–67, P30–P32),
`research/logs/S5/`, this checkpoint, stage tables.

## 8. Next: S6
Train the faculty ProgressiveCodec + BeamGRU adapted to our system (M = B = 32, 4-bit 8/16/32-value
codes, a confidence output) and plug them in via `models: trained`. Gate: the GRU must beat the
persistence predictor on held-out seeds; otherwise report it and keep persistence.
