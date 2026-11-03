# Checkpoint S4: learned per-mode quality estimator (the fix for `tvcsi.py`)

**Status: COMPLETE (2026-11-03).** Gate passed. Next: S5 (risk handling + five faculty baselines).

Self-contained together with `research/HANDOFF.md`, `S3_checkpoint.md` and `ASSUMPTIONS.md` items 56–63, P25–P29.

## 1. In one paragraph (beginner version)
The old "Proposed" scheme guessed the value of feedback from the *average* beam error. With the
measured 5G quality curve that guess always said "feedback is worthless", so Proposed never asked
any user for CSI and scored 1.25 (round-robin: 2.73). S4 replaces the guess with a model that has
**learned from practice runs** what quality each feedback size actually gives, for each user,
from what the base station knows (its beam guess, confidence, time since the last report, last
report size, message type, SNR, speed, users). Trained on the train seeds, tuned on the validation
seeds, measured once on the test seeds. **Result: Proposed now beats round-robin on all 10 test
seeds and halves the critical-message failures.**

## 2. What was done
Work by Maruthi with an AI assistant (Claude Opus 5.5).

1. `src/tvcsi/estimator.py` (new): gNB-side `features`, data collection with exploration policies,
   an ExtraTrees forest with 8 outputs (E[Q] and Pr[Q < Q_min] for modes 0–3), leaf size chosen on
   validation, deterministic cache `results/estimators/estimator_<hash>.joblib` (+ `.json` report).
2. `src/tvcsi/sim.py`: `simulate(..., estimator=, observer=)`; **one batched estimator call per slot**
   for all learned schemes; the outage probability goes to the schemes as `DecisionInputs.p_out`
   (used from S5); `observer` is a read-only hook (data collection, diagnostics).
3. `src/tvcsi/schemes.py`: `Optimised`, `Proposed`, `Proposed_no_importance` use the learned
   estimator. The pre-S4 schemes are kept bit-identical as `Optimised_plugin`, `Proposed_plugin`,
   `Proposed_plugin_no_confidence`, `Proposed_plugin_no_importance`.
4. `configs/default.yaml`: new `estimator` section; supplementary schemes now
   `Proposed_no_importance, Proposed_plugin, Optimised_plugin`.
5. Sweeps (`sweep.py`) and `scripts/sensitivity.py` use the one frozen base-config estimator at every
   point; `scripts/fit_estimator.py` (new) and a step in `scripts/run_all.sh`; plot styles for the new
   names. The plug-in table cache key ignores the new `estimator` section (the committed tables stay valid).
6. Tests: the strict-xfail `test_perfect_csi_measured_q_estimator` now passes, mark removed (the gate).
   Tests that run the main schemes get a session `estimator` fixture (`tests/conftest.py`). The
   tamper test now covers the learned schemes too. The S3 golden test maps the old names to the
   `*_plugin` names and still requires bit-identical decisions and metrics.

## 3. Results
### 3.1 Validation (seeds 2000–2004, exploration data, 240 000 samples; `logs/S4/01_fit_estimator.log`)
| leaf size | MAE | gain correlation modes 1 / 2 / 3 |
|---|---|---|
| 25 | 0.1312 | 0.282 / 0.342 / 0.372 |
| 50 | 0.1317 | 0.287 / 0.346 / 0.375 |
| **100 (chosen)** | 0.1324 | **0.288 / 0.347 / 0.376** |

### 3.2 Test seeds 1000–1009, default point (`logs/S4/eval_20261103/`, `03_evaluate_test.log`)
| Scheme | Utility | Bits/slot | Regret | Critical violation (feasible slots) |
|---|---|---|---|---|
| BestBeam oracle (reference) | 2.844 | – | 0 | 0 |
| Exact (genie) | 2.833 | 22.2 | 0.258 | 0.010 |
| full_feedback (4× budget) | 2.809 | 1024 | 0.044 | 0.029 |
| **Proposed (learned)** | **2.779** | 256 | 0.180 | **0.051** |
| Optimised (learned, DP) | 2.778 | 256 | 0.180 | 0.052 |
| Proposed_no_importance | 2.768 | 256 | 0.171 | 0.046 |
| Baseline (periodic) | 2.726 | 256 | 0.129 | 0.108 |
| importance_only | 2.697 | 256 | 0.246 | 0.182 |
| snr_threshold | 2.239 | 256 | 0.348 | 0.334 |
| Proposed_plugin / Optimised_plugin (pre-S4) | 1.250 | 0 | 0.721 | 0.721 |

Paired differences (Proposed − other, 10 seeds, 95 % t-interval): Baseline **+0.053 ± 0.006
(10/10 wins)**, importance_only +0.082 ± 0.009 (10/10), Proposed_no_importance +0.011 ± 0.004 (10/10),
Proposed_plugin +1.53 ± 0.18 (10/10), Optimised +0.0005 ± 0.0012 (greedy = DP), Exact −0.054 ± 0.005.

* Proposed reaches **98.1 % of the genie** and is the best implementable scheme at the default point.
* **Critical violations in feasible slots: 5.1 % vs 10.8 % for round-robin**, below eps = 0.10,
  although no decision uses the outage estimate yet (that is S5).
* Importance weighting helps a little (+0.011); it mattered much more in the faculty world.

### 3.3 Estimator quality on the test seeds, along the Proposed trajectory
| | MAE | gain corr. modes 1 / 2 / 3 | mean est. gain vs true gain (mode 3) | outage Brier (critical) |
|---|---|---|---|---|
| learned | 0.111 | **0.104 / 0.177 / 0.227** | 0.060 vs 0.044 | 0.146 |
| plug-in | 0.196 | 0 / 0 / 0 (constant estimate) | 0 vs 0.517 | – |

**Caveat:** the test correlation is lower than on validation (0.29–0.38). The states the real
Proposed scheme visits (short feedback gaps for valuable users) differ from those of the random
exploration policies used for training. Collecting training data along scheme-like trajectories
(on train seeds) is a possible later improvement. It also slightly overestimates the gain of
feedback (0.060 vs 0.044 for mode 3).

## 4. Tests (gate)
`python -m pytest -q` → **50 passed, 1 skipped, 0 xfailed** (`logs/S4/02_pytest_full.log`, 1 min 46 s).
The former strict xfail `test_perfect_csi_measured_q_estimator` passes (gate). The skip is
`sionna` (optional, unchanged). The S3 golden regression still holds for every unchanged scheme
and for the renamed plug-in schemes.

## 5. Deviations and notes
1. **Environment rebuilt** on Python 3.12.7 with the group's exact library versions (numpy 2.5.4,
   scipy 1.18.1, pandas 3.1.0, scikit-learn 1.9.1, joblib 1.6.0; `logs/S4/00_env.txt`). numpy 2.5
   needs Python ≥ 3.12. The S3 golden file (made with numpy 2.4.6) still matches bit for bit.
2. **Speed:** ExtraTrees predicts with a thread pool when `n_jobs = -1` (the fitting setting). Per
   slot that cost 18 ms vs 4 ms single-threaded, and the test suite took 28 min instead of 1 min 46 s.
   Prediction now runs with `n_jobs = 1` (identical output).
3. The model file (~7 MB) is not committed (`.gitignore`); its `.json` report is. It is rebuilt
   identically by `python scripts/fit_estimator.py` (~1.5 min, 8 cores); tests build it on first use.
4. No learned "no confidence" ablation (would need a second model); the plug-in one is kept.
5. `scripts/make_summary.py` still reports the plug-in calibration section; it is updated in S7.

## 6. Integrity
Faculty code untouched: committed blobs 23/23 match (`logs/S4/integrity.log`, checked with
`research/logs/S3/verify_faculty_committed.py`; the working-tree tool shows CRLF-only differences on
Windows, see S3 checkpoint §5.3).

## 7. Files changed
`src/tvcsi/estimator.py` (new), `sim.py`, `schemes.py`, `sweep.py`, `plots.py`;
`configs/default.yaml`; `scripts/fit_estimator.py` (new), `scripts/sensitivity.py`, `scripts/run_all.sh`;
`tests/conftest.py`, `test_stage3_sim.py`, `test_phase1_oracle.py`, `test_s3_plumbing.py`;
`ASSUMPTIONS.md` (56–63, P25–P29); `.gitignore`; `results/estimators/estimator_f6d7d1a8a12b.json`;
`research/logs/S4/`; this checkpoint; stage tables.

## 8. Next: S5
Five faculty baselines (periodic, age, uncertainty, importance, channel; budget filled with the
largest affordable modes) and risk handling for critical users that uses `p_out`, reported against
the Phase 1 feasibility split, with seed-paired Holm-corrected comparisons.
