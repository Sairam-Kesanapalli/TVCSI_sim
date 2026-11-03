# Research workflow: TVCSI_sim + faculty TaskVoCSI_v2 (stages S0–S7)

This folder makes the staged research work portable: anyone who checks out this branch can
continue from the latest checkpoint on their own computer.

**New here (person or AI agent)? Start with [`HANDOFF.md`](HANDOFF.md)**: project context,
where everything is, rules, and the detailed spec of the next stage.

## What lives where

| Path | What it is | Rule |
|---|---|---|
| repository root (`src/tvcsi`, `scripts/`, `configs/`, `data/`, `tests/`, …) | **side track**: our TVCSI_sim (Phase 1 + Phase 2 done) that S3–S6 extend with selected faculty components | edit only in S3–S6, one stage at a time |
| `research/faculty_TaskVoCSI_v2/` | **main track**: the faculty's TaskVoCSI_v2, **unmodified** | never edit the code; runs write only to new `results/<run>_<date>/` folders |
| `research/checkpoints/` | one self-contained `S<n>_checkpoint.md` per finished stage | read the latest one first |
| `research/logs/S<n>/` | console logs and evidence of each stage | append only |
| `research/manifests/` | SHA-256 of every original file (TVCSI_sim at Phase 2, faculty code) | used to prove the originals are untouched |
| `research/tools/make_manifest.py` | `write` / `verify <root> <manifest.csv>` | — |
| `research/tools/verify_faculty.py` | one-command check that the faculty code is untouched | run at every checkpoint |

The Phase 1 + Phase 2 baseline of the side track is commit `d639eff`. The faculty code in
`research/faculty_TaskVoCSI_v2/` is byte-identical to `research/manifests/TaskVoCSI_faculty_original.csv`.

## Stages

| Stage | Goal | Status |
|---|---|---|
| S0 | backups, directory set-up, environments, verification | **done** (`checkpoints/S0_checkpoint.md`) |
| S1 | run the faculty implementation completely (both configurations: sigmoid surrogate and measured digits task), unmodified | **done** (`checkpoints/S1_checkpoint.md`; results in `faculty_TaskVoCSI_v2/results/`) |
| S2 | analyse and document the faculty results | **done** (`S2_faculty_results.md`, `S2_analysis/`, `checkpoints/S2_checkpoint.md`) |
| S3 | side track: plumbing 3A (expose gNB-side state to schemes; decisions bit-identical) | **done** (`checkpoints/S3_checkpoint.md`; gate: golden regression bit-identical, 49 passed) |
| S4 | side track: learned per-mode quality estimator (frozen approach), validated | **done** (`checkpoints/S4_checkpoint.md`; Proposed 2.779 vs periodic 2.726, 10/10 test seeds; gate test passes, 50 passed) |
| S5 | side track: risk (outage) prediction + five baselines (periodic, age, uncertainty, importance, channel) | next |
| S6 | side track: ProgressiveCodec + BeamGRU adapted (4-bit codec, confidence) and trained | — |
| S7 | final experiments, comparison of both tracks, research report | — |

Working rules: one stage per session; every stage ends with tests, saved logs/configs, a checkpoint
document, and a stop for approval. The integration analysis behind S3–S6 (component map,
incompatibilities, risks) is summarised in `checkpoints/S0_checkpoint.md`.

## Set-up on a new computer

Python 3.11+ (3.13 used). Two separate environments, so the tracks cannot interfere:

```bash
# side track (repository root)
python -m venv ../venvs/tvcsi_combined
../venvs/tvcsi_combined/Scripts/python -m pip install -e . pytest        # Linux/macOS: bin/python
../venvs/tvcsi_combined/Scripts/python -m pip install scikit-learn joblib  # needed from S4
#   only for re-measuring the 5G quality curves or for S6 training:
#   pip install torch --index-url https://download.pytorch.org/whl/cu128   (or .../whl/cpu without a GPU)
#   pip install sionna-no-rt==2.2.0
../venvs/tvcsi_combined/Scripts/python -m pytest -q        # expect: 39 passed, 1 xfailed (until S4)

# main track (faculty code; CPU only by design)
python -m venv ../venvs/taskv2
../venvs/taskv2/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
../venvs/taskv2/Scripts/python -m pip install -r research/faculty_TaskVoCSI_v2/requirements.txt
cd research/faculty_TaskVoCSI_v2 && ../../../venvs/taskv2/Scripts/python -m pytest -q   # expect: 5 passed
```

Keep virtual environments **outside** the repository (and outside OneDrive / cloud-synced folders).
Check that the side-track environment imports this checkout:
`python -c "import tvcsi; print(tvcsi.__file__)"`.

## Rules

* **Never edit the faculty code** in `research/faculty_TaskVoCSI_v2/`. If it fails as provided,
  record the failure and propose a separately named patched copy.
* **Never overwrite results**: every run writes to a new, dated folder.
* **No full sweep before** the stage's tests and a small smoke run pass.
* **No train/test leakage**: train on train seeds, tune on validation seeds, report test seeds once.
  Side track seeds: test 1000–1009, validation 2000–2004, train 3000–3019.
* **One heavy job at a time.** Faculty runs are CPU-only; the GPU is for Sionna measurements and S6
  training. Run long jobs as detached processes with log files.
* At every checkpoint, verify the originals against `research/manifests/` (S0: on the machine that
  holds them) and record the result.

## Commits

* All stage work is committed and pushed to the branch **`research/stages-s0-s7`**; `main` changes
  only by merging the PR.
* **Authorship: everyone authors their own commits** under their own git name and e-mail
  (`git config user.name` / `user.email` in your clone). Commits made on behalf of Sairam and
  Durgesh use author `SairamKesanapalli <sairamkesanapalli@gmail.com>` with the trailer
  `Co-authored-by: Durgesh Janapareddy <durgeshjanapareddy9@gmail.com>` (the convention of the
  existing history). **No AI co-author or attribution lines** in any commit.
* Shared branch etiquette: `git pull` before starting, push right after finishing a stage, one
  person per stage at a time, **never force-push**. For extra safety, work on your own branch
  (e.g. `research/s3-<name>`) and open a PR into `research/stages-s0-s7`.
* One commit (or a few) per stage, ending with that stage's checkpoint document; update the stage
  status table above.
* On Windows PowerShell pass multi-line messages with `git commit -F <file>`.

## Model allocation (for AI-assisted sessions)

| Stage | Routine work → lighter model (Sonnet) | Reserved for the strongest model (Opus) |
|---|---|---|
| S0 | copying, zipping, environment installs, running tests | reviewing the hash and test evidence; the checkpoint |
| S1 | launching and monitoring the faculty runs, collecting outputs | diagnosing any failure (without editing their code) |
| S2 | tables and plots from the faculty outputs | interpreting results; statistical validity |
| S3 | the plumbing code, written to spec | interface design; bit-identical regression review |
| S4 | data-collection script, test scaffolding | estimator design and validation, leakage audit |
| S5 | the five baselines, plot styles | risk modelling, penalty calibration |
| S6 | training scripts, diagnostics runs | architecture adaptation (4-bit codec, confidence), debugging |
| S7 | running the final sweeps, assembling figures | final analysis, comparison, report |
