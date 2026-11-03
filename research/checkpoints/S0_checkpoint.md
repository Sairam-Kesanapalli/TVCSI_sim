# Checkpoint S0 — Backup, directory preparation, environment verification

**Status: COMPLETE (2026-11-03, 12:36–13:05).** Next: S1 (run the faculty implementation), awaiting approval.

This document is self-contained: a new session can continue from it alone.

## 1. Project structure (decided before S0)

| Track | Location | Rule |
|---|---|---|
| ORIGINAL TVCSI_sim (Phase 1+2 done) | `C:\Users\JSSR DURGESH\Documents\TVCSI_sim` | **frozen**, never modified; verified by SHA-256 manifest at every checkpoint |
| ORIGINAL faculty code | `...\TVCSI_sim\Group5_code_faculty_given\TaskVoCSI_v2` | **frozen** |
| MAIN TRACK (faculty, unmodified code) | `C:\Users\JSSR DURGESH\Documents\TVCSI_research\main_track_faculty\TaskVoCSI_v2` | code never edited; runs write only to new `results\<run>_<date>\` folders |
| SIDE TRACK (combined) | `C:\Users\JSSR DURGESH\Documents\TVCSI_research\side_track_combined\TVCSI_combined` | all S3–S6 work; git branch `combined-track` |

Stage names S0–S7 correspond to the user's Phases 0–7 (renamed to avoid confusion with the
finished "Phase 1 / Phase 2" of TVCSI_sim in ASSUMPTIONS.md):
S0 setup · S1 run faculty v2 · S2 analyse faculty results · S3 plumbing (3A) · S4 learned estimator
(frozen) · S5 risk + 5 baselines · S6 ProgressiveCodec + BeamGRU · S7 final experiments + report.

## 2. Decisions recorded
1. **Git**: commits allowed in the side-track copy and pushed. All stage work goes on the PR branch
   **`research/stages-s0-s7`** (created from `combined-track`); the user opens a PR from it to `main`
   and later stages continue on that branch. The branch must contain everything needed to continue
   on another computer (`research/` folder: README, checkpoints, logs, manifests, tools, unmodified
   faculty code). `main` on GitHub changes only by merging the PR. Author `SairamKesanapalli <sairamkesanapalli@gmail.com>`,
   trailer `Co-authored-by: Durgesh Janapareddy <durgeshjanapareddy9@gmail.com>` (the convention of
   the existing commits). **No AI co-author / attribution lines.** Identity is set in the copy's
   local git config only. Commit messages are passed with `git commit -F <file>` (PowerShell
   here-strings piped to `-F -` do not work).
2. **Execution**: Claude executes each stage itself and delegates to a Sonnet/Haiku sub-agent only
   where that is clearly better. One stage per session; stop at each checkpoint for approval.
3. **S1 scope**: run **both** faculty configurations (sigmoid surrogate and measured digits task).
4. **S4 approach**: frozen estimator (trained once, reused across sweep points).
5. **Model allocation**: follow the stage table in `research/README.md` — routine work (copying,
   installs, runs, scaffolding) goes to a Sonnet sub-agent; design, validation, debugging, review and
   checkpoints stay with Opus. (S0's routine steps were executed by Opus before this rule was
   stated; their evidence was reviewed and stands.)
6. **Main-track location from S1 on**: `research/faculty_TaskVoCSI_v2/` inside the repository (so
   results reach collaborators through the PR). `TVCSI_research\main_track_faculty\TaskVoCSI_v2` on
   this machine stays as an extra verified pristine copy.

## 3. What was done

| # | Task | Result | Log |
|---|---|---|---|
| 0.1 | No background jobs; original state recorded | 0 python processes; Phase 2 pipeline had finished 12:25:42; original tests **39 passed, 1 xfailed** | `logs\S0\01_original_state.log` |
| 0.2 | Workspace created | `TVCSI_research\{00_backups, main_track_faculty, side_track_combined, checkpoints, logs}` | — |
| 0.3 | Backups (zip) | `00_backups\TVCSI_sim_S0_20261103.zip` (16.3 MB, 474 entries, incl. `.git`), sha256 `79AF0B75…D477A`; `00_backups\TaskVoCSI_v2_original_S0_20261103.zip` (23 entries), sha256 `594E7620…124F1` | `logs\S0\02_backups.log` |
| 0.4 | SHA-256 manifests of every original file (excl. `__pycache__`, `.pyc`) | `00_backups\manifests\TVCSI_sim_original.csv` (426 files, 20.4 MB); `…\TaskVoCSI_faculty_original.csv` (23 files) | `logs\S0\02_backups.log` |
| 0.5 | Working copies (robocopy) | main track 23 files, side track 398 files (faculty folder, caches, egg-info excluded): **both byte-identical** to the originals | `logs\S0\03_copies.log` |
| 0.6 | Environments | `venvs\taskv2` (faculty `requirements.txt` + CPU torch); `venvs\tvcsi_combined` (editable install of the **copy**, CUDA torch, Sionna, scikit-learn, joblib) | `logs\S0\04_env_taskv2.log`, `05_env_combined.log` |
| 0.7 | Smoke checks | side track imports the **copy**; CUDA available; side-track tests **39 passed, 1 xfailed**; faculty tests **5 passed** | `logs\S0\06_smoke.log` |
| 0.8 | Git baseline | commit **`d639eff`** on `combined-track` (all Phase 1+2 work), pushed; `origin/main` still `25c516a` | `logs\S0\commit_msg_S0.txt` |
| 0.9 | Integrity re-check | both originals **UNCHANGED** vs manifests | `logs\S0\07_original_integrity.log` |

Tools: `00_backups\make_manifest.py` (`write` / `verify <root> <manifest.csv>`).

## 4. Environments

| | `C:\Users\JSSR DURGESH\venvs\taskv2` (main track) | `C:\Users\JSSR DURGESH\venvs\tvcsi_combined` (side track) |
|---|---|---|
| Python | 3.13 | 3.13 |
| torch | 2.15.0+cpu (the faculty code is CPU-only: `torch.set_num_threads(2)`) | 2.11.0+cu128 (RTX 4060 visible) |
| numpy / scipy / pandas | 2.5.4 / 1.18.1 / 3.1.0 | 2.5.4 / 1.18.1 / 3.1.0 |
| scikit-learn / joblib | 1.9.1 / 1.6.0 | 1.9.1 / 1.6.0 |
| matplotlib / pytest | 3.11.2 / 9.2.0 | 3.11.2 / 9.2.0 |
| other | — | sionna-no-rt 2.2.0; `tvcsi` editable → the copy |

`venvs\tvcsi` (older) is an editable install of the **original**; use it only to re-verify the original.

Machine: i7-13620H (16 threads), 16 GB RAM, RTX 4060 Laptop 8 GB (driver 592.82), ~105 GB free.

## 5. State of the work carried into the side track (Phase 2 results, default point)

From `results\summary.md` (measured 5G NR quality, 10 test seeds; mean):

| scheme | utility | crit. violation (all) | crit. violation (feasible) | bits/slot |
|---|---|---|---|---|
| Exact (genie) | 2.833 | 0.170 | 0.010 | 22 |
| Baseline (periodic) | 2.726 | 0.252 | 0.108 | 256 |
| Proposed | 1.250 | 0.763 | 0.721 | **0** |
| Optimised | 1.250 | 0.763 | 0.721 | **0** |
| BestBeam_oracle | — | 0.162 | 0.000 | — |

* Infeasible critical slots fell from **95.7 % (Phase 1, logistic Q, 20 MHz)** to **16.2 %**.
* **Proposed and Optimised request no feedback at all** (0 bits/slot): the mean-regret plug-in
  estimator sits on the saturated top of the measured Q curve. This is the flaw S4 must fix (also
  pinned by the strict-xfail test `tests/test_stage3_sim.py::test_perfect_csi_measured_q_estimator`,
  which must flip to passing in S4; its xfail mark must then be removed).
* The expected ordering Exact >= Optimised >= Proposed >= Baseline fails at 29 of 90 checks.

## 6. Resource / safety rules in force
* One heavy job at a time; check `Get-Process python` before launching; run long jobs as detached
  processes (`Start-Process … -RedirectStandardOutput`) so a session end cannot kill them.
* Faculty runs are CPU-only; GPU is reserved for S6 training and Sionna measurements.
* Never write into an existing results folder; never launch a full sweep before that stage's tests
  and a smoke run pass.
* At every checkpoint: `python -I 00_backups\make_manifest.py verify` on both originals.

## 7. Issues / deviations
* The first commit attempt passed the message via `-F -` from a PowerShell here-string, which git
  read as a pathspec; the branch was pushed at `25c516a` (no new content), then the real commit
  `d639eff` was made with `-F <file>` and pushed. No effect on `main`.
* Running the original's test suite in 0.1 (required to record its state) lets Python regenerate
  `__pycache__` there; these are git-ignored bytecode caches and are excluded from the manifest.

## 8. Repository layout for collaborators (PR branch `research/stages-s0-s7`)
`research/README.md` (set-up, stages, rules, commit convention, model allocation),
`research/checkpoints/`, `research/logs/S0/`, `research/manifests/`, `research/tools/make_manifest.py`,
`research/faculty_TaskVoCSI_v2/` (unmodified, hash-verified). Not committed: the backup zips
(they only duplicate the repository and the faculty code) and virtual environments.

## 9. Next: S1 — run the faculty implementation (awaiting approval)
Plan (all in `research/faculty_TaskVoCSI_v2` of the repository, env `venvs\taskv2`, code unmodified):
1. Smoke runs: `python run.py --quick --out results\smoke_surrogate_<date>` and
   `python run.py --quick --measured-task --out results\smoke_digits_<date>`; time them.
2. Full runs (detached, logged, both in parallel — each is ~2 CPU threads):
   `python run.py --out results\full_surrogate_<date>` and
   `python run.py --measured-task --out results\full_digits_<date>` (training, 20 test seeds, sweeps, plots).
3. If the unmodified code fails, do NOT edit it: record the failure and propose a separately named
   patched copy for approval.
4. Verify originals unchanged; write `S1_checkpoint.md`.
