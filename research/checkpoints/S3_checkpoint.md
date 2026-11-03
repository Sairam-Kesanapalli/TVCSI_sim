# Checkpoint S3: plumbing, gNB-side state for the schemes

**Status: COMPLETE (2026-11-03).** Gate passed. Next: S4 (learned per-mode quality estimator),
awaiting the group's approval.

Self-contained together with `research/HANDOFF.md`, `research/README.md` and `S2_checkpoint.md`.

## 1. In one paragraph (beginner version)
A scheduler (a "scheme") decides each slot which users send CSI feedback. Until now it saw only a
few numbers: estimated utilities, the believed SINR, importance, costs and budget. The S4 estimator
and the S5 baselines need more of what the base station already knows: its own beam guess, how
confident that guess is, how long ago each user last reported, how big that report was, and each
user's message type. S3 hands this information to every scheme **without changing any decision**.
We proved that by recording every decision of every scheme before the change and checking, after
the change, that they are exactly the same, bit for bit.

## 2. What was done
Work by Maruthi with an AI assistant (Claude Opus 5.5), following HANDOFF §5 exactly.

1. **Golden recording first, before editing any code.** `research/logs/S3/make_golden_pre_S3.py` ran
   the unchanged code (commit `2a55102`) and saved the `[T, K]` decisions of all 11 scheme names
   accepted by `build_scheme` and all 18 metrics of every scheme plus the `BestBeam_oracle` row, for
   test seeds 1000–1002 at two configurations: `default` and `K6_speed60_crit05`
   (K = 6, 60 km/h, crit_share 0.5), full length T = 2000 → `research/logs/S3/golden_pre_S3.npz`
   (138 arrays; the file stores its own cases, seeds, scheme and metric names).
2. **`src/tvcsi/schemes.py`:** `DecisionInputs` gets five fields appended after `budget`, each
   defaulting to `None`: `ghat0` [K, B] float, `conf` [K] float, `age` [K] float, `last_mode`
   [K] int64, `types` [K] int. Docstring updated. No scheme changed.
3. **`src/tvcsi/sim.py`:** per scheme, `age` (start `inf`) and `last_mode` (start 0) are kept in the
   simulator and updated after each decision with `age = where(modes > 0, 1, age + 1)`,
   `last_mode = modes`. All five fields are passed as copies.
4. **`tests/test_s3_plumbing.py`** (11 tests, the S3 gate), see §4.
5. **`ASSUMPTIONS.md`** items 53–55 (what a scheme may see, the age convention and how to convert
   to the faculty convention in S4, no new parameter values).

**Design decision recorded (as agreed in HANDOFF §5.2):** the bookkeeping lives in `sim.py`, not in
a `BeamPredictor.state()` method (the earlier idea in `S2_checkpoint.md` §5). Reason: one source of
truth that means the same for every predictor type, the S6 GRU included.

## 3. Results
* **Every decision of every scheme is bit-identical** to the pre-S3 code, at both configurations and
  all three seeds (6/6 golden cases). Every metric is identical too (exact, NaN-aware comparison, no
  tolerance).
* No result, figure or config default changed, so the Phase 2 numbers in `results/summary.md` stay
  valid.
* **The golden test can fail when it should.** `golden_sensitivity_check.py` shifted the periodic
  round-robin rule by one slot, in memory only. The comparison then flagged exactly the two schemes
  that use that rule (`Baseline`, `periodic`) and none of the other nine
  (`04_golden_sensitivity_check.log`).

## 4. Tests (gate)
`python -m pytest -q` → **49 passed, 1 skipped, 1 xfailed** (`logs/S3/02_pytest_full.log`).

| | Before S3 | After S3 |
|---|---|---|
| passed | 38 | 49 (38 + 11 new) |
| skipped | 1 | 1 (same test) |
| xfailed | 1 | 1 (same test: `test_perfect_csi_measured_q_estimator`, fixed in S4) |

The handoff expects "39 passed" before S3. On this computer one of those 39,
`test_phase2_quality.py:118`, is **skipped** because the optional package `sionna` (GPU link-level
tool, only for re-measuring Q) is not installed. It is unrelated to S3 and skipped both before and
after the change.

New tests (`logs/S3/03_pytest_s3_verbose.log`):

| Test | Checks |
|---|---|
| `test_golden_regression_bit_identical` ×6 | decisions (values and dtype) and all metrics equal the golden file exactly |
| `test_age_and_last_mode_consistent` | a probe scheme with a fixed mode pattern: `age` equals the value recomputed from its own decisions, `last_mode` equals its previous decision, and `conf == exp(-age/tau)` holds exactly (same age as `PersistencePredictor`) |
| `test_field_shapes_and_dtypes` | shapes/dtypes as specified, conf in [0, 1], types = `assign_types` |
| `test_inputs_are_copies` | a scheme that overwrites all five input arrays sees exactly what a well-behaved scheme sees later |
| `test_existing_scheme_unaffected_by_mutating_neighbour` | adding such a scheme next to Proposed/Baseline/Exact leaves their decisions unchanged |
| `test_new_fields_carry_no_true_channel` | with the true channel replaced by garbage (tamper hook), all five new fields stay identical |

The existing tamper test `test_no_true_channel_leak` still passes.

## 5. Deviations and notes
1. **Golden taken at `2a55102`, not `a20bcb5`.** `2a55102` (the handoff commit) only adds documents;
   `git diff a20bcb5 2a55102 -- src configs tests` is empty, so the code is the same.
2. **Estimator schemes in the golden file** use the committed default-point table
   `results/tvcsi_table.json` at both configurations, instead of refitting one per configuration.
   For a determinism check only the repeatability matters, and this keeps the test fast (~35 s).
3. **Integrity check false alarm on Windows.** `research/tools/verify_faculty.py` reports
   `match=0 mismatch=23` on this computer, although nothing was edited. Cause: Git for Windows'
   system setting `core.autocrlf=true` writes the files with Windows line endings (CRLF) on checkout,
   and the tool hashes the working-tree files. The committed files are untouched:
   `research/logs/S3/verify_faculty_committed.py` hashes the files as stored in git and reports
   **files=23 match=23 mismatch=0 missing=0**, and `git status` shows no change under
   `research/faculty_TaskVoCSI_v2/` (`logs/S3/integrity.log`).
   **Suggested fix for the group (not done here, outside S3):** add a `.gitattributes` line
   `research/faculty_TaskVoCSI_v2/** -text` (keep these files byte-exact on every OS), or let
   `verify_faculty.py` normalise CRLF to LF before hashing.
4. Environment: Python 3.11.9, numpy 2.4.6, scipy 1.17.1, pandas 3.1.0, scikit-learn 1.9.1,
   pytest 9.2.0 (`logs/S3/00_env.txt`); venv outside the repository and outside OneDrive.

## 6. Integrity
Faculty code untouched (committed blobs 23/23 match the S0 manifest; see §5.3). No result folder
was overwritten; S3 wrote only to `research/logs/S3/`.

## 7. Files changed
* `src/tvcsi/schemes.py`, `src/tvcsi/sim.py`: the plumbing (+22 / −2 lines).
* `tests/test_s3_plumbing.py`: new, the S3 gate.
* `ASSUMPTIONS.md`: items 53–55.
* `research/logs/S3/`: `make_golden_pre_S3.py`, `golden_pre_S3.npz`, `golden_sensitivity_check.py`,
  `verify_faculty_committed.py`, logs `00`–`04`, `integrity.log`, `commit_msg_S3.txt`.
* `research/checkpoints/S3_checkpoint.md` (this file); stage tables in `research/README.md` and
  `research/HANDOFF.md` §1.

## 8. Next: S4 (awaiting approval)
Learned per-mode quality estimator, frozen, replacing the mean-regret plug-in (HANDOFF §6). Its
features can now read `ghat0`, `conf`, `age`, `last_mode` and `types` from `DecisionInputs`
(convert `age` to the faculty convention as in ASSUMPTIONS item 54). Gate: the strict-xfail
`test_perfect_csi_measured_q_estimator` must pass and its `xfail` mark be removed.
