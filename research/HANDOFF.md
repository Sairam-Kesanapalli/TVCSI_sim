# HANDOFF — start here (for collaborators and their AI agents)

**Last updated: 2026-11-03, after S6 (see `checkpoints/S6_checkpoint.md`). Next stage: S7.**

This file is the entry point for anyone continuing the project on another computer. It says what
the project is, what is finished, where every piece of earlier work lives, which rules apply, and
exactly what the next stages must do. A new AI session should read this file first, then the files
in the reading order below, and should not need anything else.

---

## 1. What the project is (2-minute version)

**TVCSI_sim** is our group 5G project: a simulator for *task-value-aware CSI feedback* in an FDD
massive-MIMO downlink. In every time slot each user can send no CSI feedback or a 32 / 64 / 128-bit
compressed description of its channel; the total feedback per slot is limited (256 bits). A
scheduler decides who sends how much, so that the users whose messages matter most (critical,
high-importance messages) get good beams. Utility = importance × task quality Q.

**The faculty review** said our first version had four problems: (a) our "Proposed" scheme lost to
simple baselines; (b) critical users violated the quality threshold 97–98 % of the time; (c) the
models were placeholders; (d) the quality curve Q was an arbitrary logistic function. They asked for:

| File | Requested fix | Status |
|---|---|---|
| `src/tvcsi/sim.py` | true best-beam feasibility oracle + the missing constraints | **done (Phase 1)** |
| `src/tvcsi/link.py` / `quality.py` | Q measured on a real 5G NR link instead of a logistic guess | **done (Phase 2)** |
| `src/tvcsi/tvcsi.py` | replace the mean-regret plug-in utility estimator | **S4** |
| `src/tvcsi/schemes.py` | age-, uncertainty- and feedback-gain-aware baselines | **S5** |
| placeholder encoder / decoder / predictor | trained models | **S6** |

The faculty then sent their own reference implementation, **TaskVoCSI_v2**. We now work on two
independent tracks:

* **MAIN TRACK** = faculty TaskVoCSI_v2, run **unmodified** → `research/faculty_TaskVoCSI_v2/`.
* **SIDE TRACK** = our TVCSI_sim (Phase 1 + 2 done) extended stage by stage with selected faculty
  components → the repository root (`src/tvcsi/`, `configs/`, `tests/`, …).

The work is split into **stages S0–S7** (one stage per session, each ending with tests, saved logs,
a checkpoint document and a stop for the group's approval):

| Stage | Goal | Status |
|---|---|---|
| S0 | backups, workspace, environments, integrity manifests | done |
| S1 | run the faculty code completely (both configurations), unmodified | done |
| S2 | analyse and document the faculty results | done |
| S3 | side track: plumbing — give schemes the gNB-side state (decisions bit-identical) | done |
| S4 | side track: learned per-mode quality estimator (frozen), replaces the plug-in | done |
| S5 | side track: calibrated risk handling + five faculty baselines + paired Holm statistics | done |
| S6 | side track: ProgressiveCodec + BeamGRU adapted to our system, trained | done (GRU fails gate → persistence kept; trained codec adopted) |
| **S7** | final experiments, comparison of both tracks, report | **next** |

---

## 2. Where everything is

Everything needed is on GitHub, repository `Sairam-Kesanapalli/TVCSI_sim`, branch
**`research/stages-s0-s7`** (`main` is the old pre-review version; do **not** work on `main`).

```bash
git clone https://github.com/Sairam-Kesanapalli/TVCSI_sim.git
cd TVCSI_sim
git checkout research/stages-s0-s7
```

### 2.1 Reading order for a new person / agent

1. **`research/HANDOFF.md`** — this file.
2. **`research/README.md`** — workflow rules, environment set-up, commit convention, model allocation.
3. **`research/checkpoints/S2_checkpoint.md`** — latest state (then `S1_…`, `S0_…` if needed).
4. **`research/S2_faculty_results.md`** — what the faculty method does well/badly; §8 lists what
   it means for S3–S6.
5. **`ASSUMPTIONS.md`** — every modelling choice and parameter value (what / why / how / where);
   Phase 1 and Phase 2 sections plus numbered items 37–52 and the parameter list P1, P2, … under
   "Parameter values: what, why, how, where".
6. **`docs/next_steps_plan.pdf`** — the beginner-friendly plan written after the faculty review
   (Phases 1–5; S3–S6 implement its Phases 3–5).
7. **`docs/5G_group_5.pdf`** — the original project specification (equation numbers such as
   Eq. 39 in the code refer to it).

### 2.2 Map of the repository

| Path | What it contains |
|---|---|
| `src/tvcsi/` | the side-track simulator (package `tvcsi`) |
| ├ `sim.py` | slot loop, Phase 1 best-beam oracle (`best_beam_oracle`, `EvalContext`), metrics |
| ├ `schemes.py` | `DecisionInputs` (what a scheduler may see) and all schemes |
| ├ `tvcsi.py` | the current (flawed) mean-regret plug-in estimator `expected_utility` — replaced in S4 |
| ├ `solvers.py` | multiple-choice knapsack DP (`dp_mckp`) and greedy (`greedy_tvcsi`) |
| ├ `quality.py` | Phase 2: `MeasuredQuality` (measured 5G NR curves), `LogisticQuality` (legacy) |
| ├ `link.py`, `channel.py`, `codebook.py`, `importance.py`, `metrics.py`, `config.py` | system model |
| ├ `models/` | `base.py` (Encoder / Decoder / BeamPredictor interfaces), `dummy.py` (placeholders incl. `PersistencePredictor`) — trained models plug in here in S6 |
| ├ `sweep.py`, `plots.py` | parameter sweeps and figures |
| `configs/default.yaml` | every parameter (sections `quality`, `radio`, `link_level`, `constraints`, `seeds`, …) |
| `data/q_curves.csv` (+ `_meta.json`, `q_validation.csv`, `carrier_selection.csv`) | Phase 2 measured quality curves (18 curves, Sionna 5G NR PDSCH, validated) |
| `scripts/measure_q.py` | re-measures the curves (needs GPU + torch + sionna; ~hours; normally **not** needed) |
| `scripts/` (others) | table fitting, summary, sensitivity, dataset export, `run_all.sh` |
| `tests/` | pytest suite: **39 passed, 1 xfailed** expected before S4 |
| `results/`, `figures/` | Phase 2 side-track results (default point and sweeps) |
| `research/faculty_TaskVoCSI_v2/` | **faculty code, unmodified** + its run results in `results/{smoke,full}_{surrogate,digits}_20261103/` (incl. trained `networks.pt`, `calibrator.joblib`) |
| `research/S2_analysis/` | S2 analysis script, tables T1–T7, figures F1–F6 |
| `research/checkpoints/` | one checkpoint per finished stage |
| `research/logs/S<n>/` | evidence (console logs, integrity checks, commit messages) of each stage |
| `research/manifests/` + `research/tools/make_manifest.py` | SHA-256 manifests proving the faculty code / originals are untouched |

Local Windows paths that appear in `S0_checkpoint.md` (`C:\Users\JSSR DURGESH\…`, `00_backups`,
`venvs\…`) exist only on Sairam's computer (backups and the frozen original). On your computer the
repository clone is all you need.

### 2.3 Key numbers to know (side track after Phase 2, default point, 10 test seeds)

| Scheme | Utility | Bits/slot | Note |
|---|---|---|---|
| Exact (genie DP with true utilities) | 2.833 | 22.2 | upper reference |
| Baseline (periodic, 64-bit mode) | 2.726 | 256 | strong simple baseline |
| Proposed / Optimised (plug-in estimator) | 1.250 | **0** | estimator sees no value in feedback → never requests it (the S4 problem) |

Full table: `results/summary.md`.

Critical-user slots that are physically infeasible (even the true best beam misses Q_min): 16.2 %
(was 95.7 % with the old logistic Q). Faculty main track (20 seeds): CTVC-DP beats all five
baselines significantly (+1.5 % digits, +3.9 % surrogate); its gain appears only when not all users
can be served (details: `S2_faculty_results.md`).

---

## 3. Rules (non-negotiable)

1. **Never modify the faculty code** in `research/faculty_TaskVoCSI_v2/`. Check it at every
   checkpoint with `python research/tools/verify_faculty.py` → expected
   `files=23 match=23 mismatch=0 missing=0` (the `extra` files are run outputs under `results/`).
2. **Never overwrite results**: every run writes to a new dated folder.
3. **Preserve Phase 1 and Phase 2** behaviour; the Phase 2 regression tests must keep passing.
4. **Validate before sweeping**: tests + a small smoke run before any full sweep.
5. **No train/test leakage**. Side-track seeds: **test 1000–1009, validation 2000–2004,
   train 3000–3019** (`configs/default.yaml → seeds`). Train on train, tune on validation, report
   on test once.
6. **One heavy job at a time**; run long jobs detached with log files.
7. **One stage per session**: finish the stage, write `research/checkpoints/S<n>_checkpoint.md`,
   update the stage table in `research/README.md` and in §1 of this file, commit, push, then
   **stop and ask the group for approval** before the next stage.
8. **Git**: work on `research/stages-s0-s7` (or your own branch `research/s<n>-<name>` with a PR into
   it); `git pull` first; never force-push; **you author your own commits** with your own git
   name/e-mail; no AI co-author / attribution lines in commit messages. On Windows PowerShell use
   `git commit -F <file>` for multi-line messages.
9. Every new parameter value goes into `ASSUMPTIONS.md` as what / why / how / where.

---

## 4. Environment set-up (short; full version in `research/README.md`)

Python 3.11+ (3.13 used). Keep virtual environments **outside** the repository.

```bash
python -m venv ../venvs/tvcsi_combined
../venvs/tvcsi_combined/Scripts/python -m pip install -e . pytest scikit-learn joblib   # Linux/macOS: bin/python
../venvs/tvcsi_combined/Scripts/python -m pytest -q          # expect: 39 passed, 1 xfailed
```

torch is needed only for S6 (training) or re-measuring Q (`pip install -e .[linklevel]`).
The faculty code has its own environment (`research/faculty_TaskVoCSI_v2/requirements.txt`, CPU
torch); it is needed only to re-run the main track or to import its components for reference.

---

## 5. Next stage in detail: S3 — plumbing (decisions must stay bit-identical)

**Goal.** Today a scheme sees only `DecisionInputs(t, U_hat, gamma_ref, v, costs, budget)`. The
S4 estimator and the S5 baselines need more of what the base station (gNB) already knows. S3 adds
that information **without changing any decision of any existing scheme**.

**Design (decided).**

1. Extend `DecisionInputs` in `src/tvcsi/schemes.py` with new fields **appended after `budget`,
   each defaulting to `None`** (so every existing constructor call keeps working):

   | Field | Shape / dtype | Meaning |
   |---|---|---|
   | `ghat0` | `[K, B]` float | the scheme's own predicted beam gains (from its predictor) |
   | `conf` | `[K]` float in [0, 1] | the predictor's confidence |
   | `age` | `[K]` float | slots since this scheme last let the user feed back (mode > 0): 1 if in the previous slot, `inf` if never |
   | `last_mode` | `[K]` int64 | mode this scheme chose for the user in the previous slot (0 at t = 0) |
   | `types` | `[K]` int | message type of each user (`importance.CRITICAL` etc.) |

2. **Bookkeeping lives in `sim.py`, not in the predictor.** Keep, per scheme, `age` (initial
   `np.full(K, np.inf)`) and `last_mode` (initial zeros); after each decision update
   `age = np.where(modes > 0, 1.0, age + 1.0)` and `last_mode = modes`. This is exactly the
   convention of `PersistencePredictor.age` and works for any future predictor (S6 GRU).
   (We decided against the earlier idea of a `BeamPredictor.state()` method: one source of truth,
   independent of the predictor type. Record this in the S3 checkpoint.)
   *Note:* the faculty `History` uses a different convention (age starts at 25, 0 right after
   feedback, capped at 100 in features). When S4 ports their features, convert explicitly
   (faculty_age = our_age − 1, with `inf` → 25) and document it.
3. Pass **copies** (`.copy()`) of all arrays into `DecisionInputs`, so a scheme cannot corrupt the
   simulator state. `gamma_ref` and `v` keep their current meaning.
4. Do **not** change any scheme, estimator, solver, metric or config default in S3.

**Tests to add** (`tests/test_s3_plumbing.py`) — the S3 gate:

1. **Golden regression (most important).** *Before* editing anything, run the current code
   (commit `a20bcb5`) and save, for every scheme name accepted by `build_scheme` (Baseline, Exact,
   Optimised, Proposed, Proposed_no_confidence, Proposed_no_importance, periodic, snr_threshold,
   importance_only, prediction_only, full_feedback) the `[T, K]` decisions
   (`simulate(..., record_decisions=True)`) and all metrics, for test seeds 1000–1002 at the default
   config and at least one non-default config (e.g. other K and speed). Store it as
   `research/logs/S3/golden_pre_S3.npz` (+ a small script that produced it). After the change, the
   test asserts **exact equality** of decisions and metrics (use `np.array_equal`, not tolerances;
   NaN-aware for metrics).
2. **Age consistency**: a probe `Scheme` (Scheme objects can be passed to `simulate`) records the
   `DecisionInputs` it receives; with dummy models, `age` must equal the `PersistencePredictor.age`
   that produced `conf` (i.e. `conf == exp(-age / tau)`), and `last_mode` must equal its own previous
   decision.
3. Shapes / dtypes of the new fields; arrays are copies (mutating them inside a scheme does not
   change later slots).
4. The existing tamper test (no true-channel information reaches non-genie schemes) still passes.

**Gate.** Full suite: previous 39 passed + 1 xfailed unchanged, plus the new tests passing; golden
regression exact. Then write `research/checkpoints/S3_checkpoint.md`, logs in `research/logs/S3/`,
update the stage tables, commit, push, stop for approval.

Model allocation for AI-assisted work: routine coding to the spec above can go to a lighter model;
the interface review and the bit-identical regression review need the strongest model.

---

## 6. Later stages (guides; refine at the start of each stage)

### S4 — learned per-mode quality estimator (the fix for `tvcsi.py`)
* **Problem**: the plug-in `Uhat = v·Q(gamma_ref·(1 − mean regret))` sits on the flat top of the
  measured Q curve, so feedback looks worthless and Proposed/Optimised send 0 bits.
* **Approach (decided: frozen)**: port the faculty idea — a regressor (ExtraTrees) that predicts,
  per user and per feedback mode, the expected quality (and outage probability) from gNB-side
  features only. Faculty reference: `research/faculty_TaskVoCSI_v2/taskv2/policy.py`
  (`observation` = features, `candidates` = counterfactual labels, calibrator fitting), features
  use `ghat0`, `last`, `age`, `last_mode`, critical flag, SNR, speed → these come from S3.
* Collect training data on **train seeds 3000–3019**, tune on **validation 2000–2004**, report on
  **test 1000–1009**. Train once and reuse (frozen) across sweep points. Batch one estimator call per
  slot for all schemes (~20 ms per call otherwise dominates runtime).
* Report test MAE and **gain correlation** (does it rank the value of feedback correctly?), not only
  validation numbers. Keep the plug-in as an ablation scheme.
* Gate: the strict-xfail test `test_perfect_csi_measured_q_estimator` in
  `tests/test_stage3_sim.py` must now pass → remove its `xfail` mark.
* Expectation (from S2): a big jump from 1.25, but margins over strong baselines may be small,
  because our side track serves every user (the faculty's gain comes from joint feedback + service).

### S5 — risk + five baselines (the fix for `schemes.py`)
* Add the faculty baselines **periodic, age, uncertainty, importance, channel** (reference:
  `taskv2/policy.py → baseline_modes`, which fills the budget with the largest affordable modes).
* Risk: the faculty soft penalty is **inert** (S2 §5). Use a calibrated penalty or a virtual-queue
  (Lyapunov) weight for critical users; evaluate against the Phase 1 feasibility split
  (`violation_rate_feasible`, `reliability_met`).
* Report with seed-paired differences and Holm correction (as in the faculty `taskv2/eval.py`).

### S6 — ProgressiveCodec + BeamGRU (replace the placeholders)
* Adapt to our feedback sizes (32/64/128 bits = 8/16/32 values × 4 bits; faculty uses 6-bit STE
  quantisation) and expose a confidence. Faculty reference: `taskv2/nn.py`; trained faculty weights
  in `results/full_*/networks.pt` (their system: M = B = 16, ours M = B = 32 → retraining needed).
* Plug in via `models: trained` in the config and the interfaces in `src/tvcsi/models/base.py`.
* **Gate**: the GRU must beat the persistence predictor on held-out seeds (the faculty GRU did not:
  0.643 vs 0.634 accuracy); otherwise report that and keep persistence. GPU training allowed; one
  heavy job at a time.

### S7 — final experiments and report
Full sweeps only after S4–S6 gates pass; compare both tracks; final report with honest limitations.

---

## 7. Checkpoint template (what each `S<n>_checkpoint.md` contains)

Status + date · what was done (and by whom / which model) · results with numbers · tests run and
their output · deviations from the plan and why · integrity check (faculty manifest) · files
changed · next stage. Look at `research/checkpoints/S2_checkpoint.md` as an example.

---

## 8. Prompt to give your AI agent

> You are continuing a staged research project in this repository (branch `research/stages-s0-s7`).
> Read `research/HANDOFF.md` first, then `research/README.md`, the latest checkpoint in
> `research/checkpoints/`, `research/S2_faculty_results.md` and `ASSUMPTIONS.md`. Follow every rule
> in HANDOFF §3. Do only stage **S<n>** as specified in HANDOFF §5/§6: implement, test (the stage
> gate), save logs to `research/logs/S<n>/`, write `research/checkpoints/S<n>_checkpoint.md`, update
> the stage tables, then stop and report. Never edit `research/faculty_TaskVoCSI_v2/`. Explain
> results in beginner-friendly language.
