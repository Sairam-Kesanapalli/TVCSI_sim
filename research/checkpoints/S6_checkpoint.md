# Checkpoint S6: trained codec and GRU beam predictor (replacing the placeholders)

**Status: COMPLETE (2026-11-03).** Gate outcome: **GRU fails at the default point → persistence kept**
(as HANDOFF §6 prescribes); the trained codec is adopted. Next: S7 (final experiments and report).

Self-contained together with `research/HANDOFF.md`, `S5_checkpoint.md` and `ASSUMPTIONS.md` items 68–73, P33–P35.

## 1. In one paragraph (beginner version)
The faculty asked "why not AI?". S6 trains two small neural networks. A **codec** squeezes each
user's channel into 8, 16 or 32 numbers of 4 bits (exactly the 32 / 64 / 128-bit reports). A
**GRU** guesses the beam from the history of reports and also says how sure it is. The codec works
well (better than the old fake noise model for small reports). The GRU is **not better** than
"assume the beam did not change" at the default setting. It ties, as the faculty GRU did. It only
helps at low SNR and at extreme speeds. Following the agreed rule, we keep the simple predictor and
use the trained codec. With the trained codec the Proposed scheme still beats every simple baseline.

## 2. What was done
Work by Maruthi with an AI assistant (Claude Opus 5.5).
1. `src/tvcsi/models/trained.py` (new): numpy inference of `TrainedEncoder`, `TrainedDecoder`,
   `GRUPredictor` (learned confidence) and `Persistence` (persistence with the trained-model
   constructor signature).
2. `scripts/train_models.py` (new, torch, CPU): data from the simulator's own realisations; codec,
   then GRU histories using the trained codec; epochs chosen on validation; test diagnostics;
   writes `data/models/s6_models.npz` (190 kB) and `.json`.
3. `configs/default.yaml`: `trained` section filled (codec + Persistence), new `models_train`
   section. The default stays `models: dummy`, so all earlier tests and goldens are unchanged. The
   final runs (S7) use `models: trained`.
4. `src/tvcsi/estimator.py`: with dummy models the estimator cache hash ignores the `trained`
   section. The default estimator was refitted under its new name with **identical** validation
   numbers (deterministic fit); the stale cache file was removed.
5. `tests/test_s6_models.py` (5 tests); `research/logs/S6/evaluate_S6.py` (system check).

## 3. Results
### 3.1 Codec, test seeds 1000–1009 (`01_train_models.log`)
| report | trained regret / top-1 accuracy | old dummy codec |
|---|---|---|
| 32 bit (8 values) | **0.155 / 0.695** | 0.195 / 0.623 |
| 64 bit (16 values) | **0.079 / 0.784** | 0.086 / 0.764 |
| 128 bit (32 values) | 0.057 / 0.815 | 0.044 / 0.841 |

### 3.2 GRU vs persistence, test seeds, random feedback (P(mode 0) = 0.75) — **the S6 gate**
| setting | GRU regret / acc | persistence regret / acc | persistence − GRU regret (95 % CI) | seeds GRU better |
|---|---|---|---|---|
| **default (30 km/h, 15 dB)** | 0.2922 / 0.503 | 0.2921 / 0.510 | **−0.0001 ± 0.0044** | **5/10 → gate fails** |
| 3 km/h | 0.1015 / 0.738 | 0.1137 / 0.734 | +0.012 ± 0.008 | 8/10 |
| 60 km/h | 0.4275 / 0.361 | 0.4359 / 0.355 | +0.008 ± 0.005 | 9/10 |
| SNR 5 dB | 0.4858 / 0.348 | 0.5567 / 0.305 | +0.071 ± 0.009 | 10/10 |

Confidence Brier score 0.17–0.23. Why the GRU cannot win at the default point: the synthetic channel
is close to memoryless given the last report, so older history adds little; at low SNR it wins by
denoising the reports.

### 3.3 Whole system, test seeds, default point (`02_evaluate_test.log`, `eval_20261103/`)
| scheme | dummy models | **trained codec + persistence** | trained codec + GRU |
|---|---|---|---|
| Exact (genie) | 2.833 | 2.833 | 2.831 |
| **Proposed** | 2.779 | **2.774** | 2.769 |
| Proposed_risk | 2.771 | 2.773 | 2.766 |
| Baseline (periodic) | 2.726 | 2.729 | 2.725 |
| fill_importance | 2.697 | 2.684 | 2.683 |
| full_feedback (4× bits) | 2.809 | 2.798 | 2.798 |

Proposed − best simple baseline (spec periodic): **+0.046 ± 0.008, 10/10 seeds** with the trained
codec (Holm p < 1e-6 against all five compared baselines). Critical violations in feasible slots:
Proposed 5.4 %, Baseline 10.5 %. With the trained codec, Proposed_risk costs almost nothing (−0.001).

## 4. Tests
`python -m pytest -q` → **70 passed, 1 skipped** (`logs/S6/03_pytest_full.log`). New: code sizes and
4-bit levels, progressive prefix property, GRU interface, **numpy = torch to 1e-9** for codec and GRU,
simulator runs with both trained setups without true-channel leaks.

## 5. Deviations and notes
1. Trained on CPU (torch 2.15.0+cpu, reproducible). A GPU (RTX 3050) is available but not needed.
2. Faculty-size choices changed where our system differs (M = B = 32, 4-bit values): P33–P34.
3. The GRU is kept in the code (`GRUPredictor`) for the record and for low-SNR use; it is not
   used in the final configuration.

## 6. Integrity
Faculty code untouched: committed blobs 23/23 (`logs/S6/integrity.log`).

## 7. Files changed
`src/tvcsi/models/trained.py` (new), `src/tvcsi/estimator.py`, `src/tvcsi/plots.py` (S7 figures prepared),
`scripts/train_models.py` (new), `scripts/make_summary.py` (S7 report prepared),
`configs/default.yaml`, `tests/test_s6_models.py` (new), `data/models/s6_models.{npz,json}`,
`results/estimators/*.json` (refitted estimators, reports only), `ASSUMPTIONS.md` (68–73, P33–P35),
`research/logs/S6/`, this checkpoint, stage tables.

## 8. Next: S7
Full sweeps with `configs/final.yaml` (trained codec + persistence) into new dated folders, figures,
summary, and the comparison of both tracks in a final report.
