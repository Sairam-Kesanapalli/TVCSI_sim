# Codex review findings on PR #1 and PR #3: assessment and fixes (2026-11-04)

| # | PR | Finding (Codex) | Valid? | Action |
|---|---|---|---|---|
| A | #1, `scripts/measure_q.py:97` (P2) | repetition starts from K'/R_mother instead of the NR rate-matching buffer, so curves repeat early | **yes in principle, negligible in effect** | documented and quantified, curves **not** re-measured |
| B | #3, `src/tvcsi/estimator.py:255` (P1) | scikit-learn and joblib not declared, so a clean install fails | **yes** | added to `pyproject.toml` dependencies |
| C | #3, `src/tvcsi/estimator.py:224` (P2) | estimator cache key has the weights *path*, not contents, so a re-trained model reuses a stale estimator | **yes** (same bug for `data/q_curves.csv`) | key now includes file contents (text with LF line endings); 3 new tests |

## A: quantification (`ncb_check.py` → `01_ncb_check.log`)
Per curve: distinct coded bits per code block in our measurement (`ours` = K'/R_mother, Sionna's
limit) vs the NR circular buffer without filler bits (`nr_buffer` = 44 Zc + K' for BG1,
40 Zc + K' for BG2; TS 38.212 5.4.2.1, LBRM inactive). 9 of 18 curves use repetition; they send
40–808 fewer distinct parity bits than NR. Estimated SNR penalty (Gaussian-capacity approximation,
QPSK): 0.004–0.053 dB, and 0.151 dB for background at 16 PRB (only the K = 16 sweep point). Every
critical curve for K ≥ 4 has no repetition and is unaffected. The direction is pessimistic (our BLER
is slightly too high). All effects are below the 0.25 dB measurement step. Codex's example
(33-PRB background) quotes 5376 usable bits; by the spec it is 5536 (Zc = 112, K' = 1056: 40·112 + 1056).
Either way the conclusion is the same. Recorded in ASSUMPTIONS P24 and in the `measure_q.py` comments.

## C: verification (`refit_check.py` → `02_refit_check.log`)
The new keys renamed the cache files (default `8d2408872161` → `2d856f90b8cd`, final `266dd335e39c` →
`8c3b3538564b`). Both were re-fitted and their reports are identical to the old ones except for the
fit time, so no result changes. Old `.json` reports are kept as records (S6 logs refer to them).

## Tests
`03_pytest_full.log`: 74 passed (71 before, plus `tests/test_estimator_cache_key.py`).
