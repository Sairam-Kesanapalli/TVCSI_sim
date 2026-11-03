# Checkpoint S2 — Analyse and document the faculty results

**Status: COMPLETE (2026-11-03).** Next: S3 (plumbing 3A in the side track), awaiting approval.

Self-contained together with `research/README.md`, `S0_checkpoint.md` and `S1_checkpoint.md`.

## 1. What was done
* (Sonnet sub-agent) `research/S2_analysis/s2_faculty_analysis.py` reads the two S1 full runs and writes
  tables T1–T7 (`S2_analysis/tables/`), figures F1–F6 (`S2_analysis/figures/`, 200 dpi PNG) and the
  console record `S2_analysis/s2_run.log`. Read-only on the faculty folder; no simulation, no training.
  Re-run: from the repository root, `<venv tvcsi_combined>\Scripts\python -I research\S2_analysis\s2_faculty_analysis.py`.
* (Opus) Review: T1/T2 cross-checked against the raw `summary.csv` / `paired_comparisons.csv` read in
  S1; T3 (per-seed wins) recomputed independently from `per_seed.csv` (identical); sweep figure and
  the K-sweep service counts inspected directly. Interpretation written to
  **`research/S2_faculty_results.md`** (the S2 deliverable).

## 2. Key findings (details and caveats in `S2_faculty_results.md`)
1. CTVC-DP beats all five baselines, significantly after Holm correction with 20 paired seeds: +1.5 %
   over the best baseline on digits (3.598 vs 3.544), +3.9 % on surrogate (3.526 vs 3.392); wins
   15–18/20 seeds (digits), 20/20 (surrogate).
2. The gain is ≈ 0 when all users are served (K = 4 ≤ 5 RBs) and appears when users compete for
   service (K = 8, 12): likely from joint feedback + service decisions. **Our side track serves every
   user, so this mechanism is absent there** (hypothesis; 5-seed sweeps).
3. Importance weighting is essential (−18 to −20 % without it); the risk penalty and the exact DP add
   nothing measurable.
4. Learned calibrator: test MAE 0.11–0.16, test gain correlation 0.12–0.20 (validation 0.21–0.27):
   weak but real ranking of feedback value (ours ≈ 0). ~20 ms per call: batch across schemes.
5. Trained GRU ≈ persistence (accuracy 0.643 vs 0.634); codec 96-bit mode not best in regret.
6. Genie reaches 3.89 with 16–44 bits/slot; CTVC-DP uses ~140 bits and reaches 91–92 % of it.

## 3. Consequences recorded for the next stages
* S4: port the per-mode expected-quality estimator; judge it on **test** gain correlation and MAE;
  batch inference per slot; train 3000–3019, validate 2000–2004; keep the plug-in as an ablation.
  Expect a large jump over our current Proposed (0 bits/slot, utility 1.25) but possibly small or no
  margins over strong baselines.
* S5: the faculty soft risk penalty is inert — use a calibrated penalty or a virtual queue for critical
  users, reported against the Phase 1 feasibility split; adopt the five baselines and the seed-paired
  Holm comparison.
* S6: gate on "GRU beats persistence on held-out seeds".
* Optional (outside the agreed scope, needs approval): an RB-limited service decision in the side
  track; an instrumented re-run of the faculty code to test the joint-decision hypothesis.

## 4. Integrity
No file under `research/faculty_TaskVoCSI_v2/` was modified (verified at commit time, see
`logs/S2/integrity.log`); original `TVCSI_sim` unchanged.

## 5. Next: S3 — plumbing 3A in the side track (awaiting approval)
Expose gNB-side state to the schemes without changing any decision: optional `BeamPredictor.state()`
(age, last mode), `PersistencePredictor.state()`, extend `DecisionInputs` with `ghat0`, `age`,
`last_mode`, message types; `sim.py` passes them through. Gate: full test suite unchanged and every
decision of every scheme bit-identical to the current code on the default seed(s). (Sonnet writes the
code to spec; Opus designs the interface and reviews the bit-identical regression.)
