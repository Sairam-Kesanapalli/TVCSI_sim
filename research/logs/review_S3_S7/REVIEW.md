# Review of stages S3–S7 (PR #2 / #3) and the patches that followed

**Date:** 2026-11-03. **Reviewed:** commits `ce1bc73` (S3) … `bbc5fc8` (S7) by Maruthi, merged into
`main` as PR #3. **Reviewed on** a second computer (Python 3.13.2, numpy 2.5.4, scipy 1.18.1,
scikit-learn 1.9.1, `00_env.txt`); the work was produced on Python 3.11.9, numpy 2.4.6.

## 1. Verdict
All stages S0–S7 are complete and sound. No result changes. Four patches were made
(branch `research/review-patches`): two portability fixes, one documentation correction of the
train/validation/test protocol in S6, and one wording fix.

## 2. What was checked

| Check | Result | Evidence |
|---|---|---|
| Faculty code untouched | committed files 23/23 identical to the S0 manifest | `research/tools/verify_faculty.py` |
| Earlier (Phase 2) results overwritten? | no; S7 writes to `results/final_20261103/`, `figures/final_20261103/` | `git diff --name-status 2a55102 bbc5fc8` (only additions under results/figures) |
| Train / validation / test separation | S4 estimator: train + val seeds only; S5 risk weight: val seeds; S6 training and epoch choice: train + val | `src/tvcsi/estimator.py` (`build_dataset`), `research/logs/S5/calibrate_risk.py`, `scripts/train_models.py` |
| S6 model choices | **taken on test seeds** → re-checked on validation, same outcome (patch 3) | `research/logs/S6/04_gate_on_validation.log` |
| S3: decisions bit-identical | yes. On this computer, 3 of 6 golden cases failed only because the metrics differ in the last digit; the **unchanged pre-S3 code** shows the same differences here. Against a golden file written here by the pre-S3 code, all S3 tests pass | `01_golden_this_pc_vs_committed.log`, `golden_pre_S3_this_pc.npz` |
| S4 estimator rebuilds identically elsewhere | yes (refitted here: identical validation numbers; only `fit_seconds` differs) | `results/estimators/*.json` |
| S7 final numbers reproduce | yes: test seeds 1000 and 1005, six schemes, final estimator refitted from scratch here; differences 0 or ≤ 4.4e-16 | `02_reproduce_final.log` |
| Test suite, this computer | before the patches 68 passed, 3 failed (the golden last-digit issue above); **after: 71 passed** (sionna installed, so nothing skipped) | `03_pytest_full_after_patches.log` |

## 3. Patches
1. **`tests/test_s3_plumbing.py`**: decisions are still compared **exactly**; metrics now with an
   absolute tolerance of 1e-12 (observed cross-computer differences ≤ 4.4e-16). Before: the suite
   failed on any computer with a different numpy build.
2. **`research/tools/verify_faculty.py`** (written in the handoff commit): hashed only the working
   tree, so on Windows (`core.autocrlf=true`, LF → CRLF on checkout) it reported all 23 faculty files
   as changed. Now it checks the files as committed in git (authoritative) and the working tree with
   CRLF→LF normalisation, and names any file that really differs (tested by editing a file: detected,
   exit code 1). Supersedes `research/logs/S3/verify_faculty_committed.py` (kept as a record).
3. **S6 protocol**: the GRU-vs-persistence gate and the codec choice were read off test-seed
   diagnostics. Repeated on the validation seeds 2000–2004: persistence − GRU regret
   +0.0023 ± 0.0046 at the default point (not significant) → persistence kept; trained codec better at
   32/64 bit and worse at 128 bit, as on test → codec kept. Recorded in `S6_checkpoint.md` §5 item 4
   and in the S7 report header. No number changes.
4. **Wording** in `S7_final_report.md` §4: "importance weighting matters (−0.007)" →
   importance weighting is far less decisive here than in the faculty world (≈ 0.01 vs 18–20 %).
   Test counts in README / HANDOFF / report updated (70 passed, 1 skipped; 71 with sionna).

**Not changed (group decision pending):** the Phase 2 files `results/summary.md` and
`results/sensitivity.md` contain cp1252 bytes (e.g. "±"), shown as "�" in UTF-8 viewers
(`S7_checkpoint.md` §5 item 1). Re-encoding would not change any number but rewrites Phase 2 result
files, so it is left for the group.

## 4. Re-running this review
From the repository root, with the side-track environment:
* `python research/logs/review_S3_S7/compare_golden.py <a.npz> <b.npz>`: compares two golden files.
* `make_golden_local.py <out.npz>`: the S3 golden generator writing to a chosen file. Run it with the
  **pre-S3 code** (`git worktree add ../pre_s3 2a55102`, run there) to make a golden file for your own
  computer.
* `python research/logs/review_S3_S7/reproduce_final.py`: re-simulates two test seeds of the final
  configuration and compares with `results/final_20261103/default_runs.csv`.
* `python research/logs/S6/gate_on_validation.py`: the S6 gate on validation seeds (needs torch).
