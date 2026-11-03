# Checkpoint S1 — Run the faculty implementation (TaskVoCSI_v2), unmodified

**Status: COMPLETE (2026-11-03, 13:08–13:45).** Next: S2 (analyse and document the faculty results), awaiting approval.

Self-contained: read together with `research/README.md` and `S0_checkpoint.md`.

## 1. What was run

All runs used the faculty code in `research/faculty_TaskVoCSI_v2/` **exactly as provided** (23 files,
byte-identical to `research/manifests/TaskVoCSI_faculty_original.csv` before and after), environment
`venvs\taskv2` (Python 3.13.2, torch 2.15.0+cpu, scikit-learn 1.9.1, numpy 2.5.4; full list in
`logs/S1/00_env.txt`), working directory `research/faculty_TaskVoCSI_v2`. Each run wrote to a new folder.

| Run | Command | Exit | Wall time | Output folder (`research/faculty_TaskVoCSI_v2/results/…`) | Log |
|---|---|---|---|---|---|
| smoke, sigmoid surrogate | `python -u run.py --quick --out results\smoke_surrogate_20261103` | 0 | 32 s | `smoke_surrogate_20261103` | `logs/S1/01_smoke_surrogate.log` |
| smoke, measured digits | `python -u run.py --quick --measured-task --out results\smoke_digits_20261103` | 0 | 104 s | `smoke_digits_20261103` | `logs/S1/02_smoke_digits.log` |
| **full, sigmoid surrogate** | `python -u run.py --out results\full_surrogate_20261103` | 0 | 826 s | `full_surrogate_20261103` | `logs/S1/04_full_surrogate.log` |
| **full, measured digits** | `python -u run.py --measured-task --out results\full_digits_20261103` | 0 | 784 s | `full_digits_20261103` | `logs/S1/05_full_digits.log` |

The two full runs ran in parallel as detached processes (PIDs 30100 / 25904, launched 13:17:48,
`logs/S1/06_full_launch.txt`); both `.err` logs are empty. A full run = train the quantised
ProgressiveCodec + BeamGRU (train seeds 101–106), fit the ExtraTrees quality/outage calibrator
(validated on seeds 201–202), test 11 schemes on 20 held-out seeds (301–320, T = 140 slots, warm-up 10),
paired seed-level comparisons with Holm correction, sweeps over budget/speed/K/SNR on the first 5 test
seeds with frozen models (143 rows), PDF plots, `model_diagnostics.json`.

Smoke runs are pipeline checks only (2 test seeds, 3 epochs) and are **not results**.

## 2. Results (full runs, 20 test seeds, mean utility per slot ± 95% t half-width)

| scheme | **digits** utility | bits/slot | outage | **surrogate** utility | bits/slot | outage |
|---|---|---|---|---|---|---|
| CTVC-DP (proposed) | **3.60 ± 0.06** | 137.5 | 0.43 | **3.53 ± 0.07** | 143.1 | 0.44 |
| CTVC-greedy | 3.61 ± 0.06 | 136.2 | 0.43 | 3.53 ± 0.07 | 142.6 | 0.45 |
| no-importance | 2.89 ± 0.06 | 133.0 | 0.48 | 2.88 ± 0.07 | 142.4 | 0.48 |
| no-risk | 3.60 ± 0.06 | 136.7 | 0.43 | 3.52 ± 0.07 | 143.0 | 0.44 |
| periodic | 3.49 ± 0.08 | 144 | 0.44 | 3.30 ± 0.08 | 144 | 0.47 |
| age | 3.49 ± 0.07 | 144 | 0.45 | 3.35 ± 0.08 | 144 | 0.46 |
| uncertainty | 3.52 ± 0.07 | 144 | 0.43 | 3.26 ± 0.09 | 144 | 0.47 |
| importance | 3.54 ± 0.07 | 144 | 0.44 | 3.39 ± 0.08 | 144 | 0.46 |
| channel | 3.54 ± 0.07 | 144 | 0.44 | 3.30 ± 0.09 | 144 | 0.47 |
| prediction only | 2.50 ± 0.25 | 0 | 0.55 | 2.05 ± 0.24 | 0 | 0.62 |
| genie (privileged) | 3.89 ± 0.04 | 16.1 | 0.37 | 3.89 ± 0.05 | 44.0 | 0.37 |

Paired differences CTVC-DP − baseline (20 seeds, `paired_comparisons.csv`):

| baseline | digits diff (± CI) | Holm p | surrogate diff (± CI) | Holm p |
|---|---|---|---|---|
| age | +0.11 ± 0.03 | 1.4e-5 | +0.17 ± 0.03 | 4.1e-10 |
| periodic | +0.11 ± 0.05 | 4.4e-4 | +0.23 ± 0.04 | 5.8e-9 |
| uncertainty | +0.08 ± 0.04 | 9.9e-4 | +0.27 ± 0.05 | 8.8e-9 |
| importance | +0.05 ± 0.03 | 5.8e-3 | +0.13 ± 0.03 | 7.6e-8 |
| channel | +0.06 ± 0.04 | 0.015 | +0.23 ± 0.06 | 2.6e-7 |
| no-importance | +0.71 ± 0.06 | 3e-15 | +0.64 ± 0.06 | 8e-15 |
| no-risk | −0.01 ± 0.02 | 0.38 (n.s.) | +0.01 ± 0.02 | 0.86 (n.s.) |
| CTVC-greedy | −0.01 ± 0.01 | 0.24 (n.s.) | −0.00 ± 0.01 | 0.86 (n.s.) |

Calibrator (validation seeds): digits MAE 0.117, RMSE 0.261, outage Brier 0.075, per-mode gain
correlation 0.21 / 0.21 / 0.24 (n = 6720); surrogate MAE 0.164, gain correlation 0.24 / 0.24 / 0.27.

**Reproducibility vs the faculty's `EXPERIMENT_NOTES.md`** (their 10-seed digits run): CTVC-DP 3.6018,
periodic 3.4647, age 3.4787, uncertainty 3.5009, importance 3.5321, channel 3.5277, genie 3.8497 — our
20-seed run gives 3.60 / 3.49 / 3.49 / 3.52 / 3.54 / 3.54 / 3.89: the numbers reproduce closely on a
different machine, Python 3.13 and newer library versions.

## 3. Observations recorded for S2 (facts only; interpretation is S2's job)
* With 20 seeds, CTVC-DP beats **all five** baselines significantly after Holm correction in both
  configurations (the faculty's 10-seed notes reported that two did not survive the correction).
* Margins are small on the digits task (+0.05 to +0.11, ≈ 1.5–3 %) and larger on the surrogate
  (+0.13 to +0.27, ≈ 4–8 %).
* Importance weighting carries most of the gain (no-importance −0.64 to −0.71); the risk penalty has no
  measurable effect (no-risk ≈ CTVC-DP, outage unchanged); exact DP ≈ greedy.
* The genie needs only 16–44 bits/slot vs ~140 for the learned policy: large headroom in *which* users
  to refresh.
* Calibrator gain correlation 0.21–0.27: clearly better than our plug-in (≈ 0), far from perfect.
* `ideal_outage` ≈ 0.0002: in this world almost no critical slot is infeasible with the best beam (our
  side track: 16.2 %); outage ≈ 0.43 is dominated by scheduling, not physics.
* Quick-mode calibrator results (MAE ≈ 0.49, correlation ≈ 0) are a training-size artefact and should
  not be quoted.

## 4. Integrity and safety evidence (`logs/S1/07_S1_closeout.log`)
* Faculty code in the repository: 23/23 files match the S0 manifest; the only extra files are the four
  results folders (72 files).
* Original `C:\Users\JSSR DURGESH\Documents\TVCSI_sim`: 426 files **UNCHANGED**.
* Pristine copy `TVCSI_research\main_track_faculty`: **UNCHANGED**.
* No file > 50 MB (largest: full-surrogate `calibrator.joblib`, 22.2 MB); all four results folders are
  committed, including `networks.pt` and `calibrator.joblib`, so collaborators can re-evaluate with the
  faculty's `resume.py` without retraining. Load `.joblib` files only from this repository (pickle).

## 5. Model allocation in S1
Sonnet sub-agent: preflight, smoke runs, output inventory, launching the full runs. Opus: gate review of
the smoke runs before the full runs, review of all evidence, integrity checks, this checkpoint, commit.

## 6. Next: S2 — analyse and document the faculty results (awaiting approval)
Plan: (Sonnet) tables and figures from `summary.csv`, `per_seed.csv`, `paired_comparisons.csv`,
`sweeps.csv`, `model_diagnostics.json`, `test_calibration.json` of both full runs; (Opus) interpretation:
where and why CTVC-DP wins, effect sizes vs CI, sweep behaviour (budget/speed/K/SNR, frozen-model shift),
calibrator quality, risk-term null effect, comparison of the faculty world with our side track (quality
model, feasibility, scale) and what it implies for S3–S6. Deliverable: `research/S2_faculty_results.md`
plus figures, `S2_checkpoint.md`.
