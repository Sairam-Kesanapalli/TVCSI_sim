# deprecated/: superseded files, kept for the record

Nothing in this folder is used by the code, the tests or the current results. It is kept so that
earlier stages stay traceable (git history has everything as well). Moved here on 2026-11-04
during the repository clean-up.

| Path | What it is | Superseded by |
|---|---|---|
| `phase2_results/results/` | side-track results of **Phase 2** (measured 5G NR quality, but the old mean-regret plug-in estimator and placeholder models): sweep CSVs, `default_runs.csv`, `summary.md`, sensitivity tables, and `tables/` (the per-sweep-point plug-in regret tables of that run) | `results/final_20261103/` (S7 final run) |
| `phase2_results/figures/` | the Phase 2 figures `fig01`–`fig15`, `supp01`–`supp05` (PNG; the PDF copies were removed) | `figures/final_20261103/`; side-by-side comparison in `figures/before_after/` |
| `docs/next_steps_plan.pdf` | the beginner-friendly plan written after the faculty review (its Phases 3–5 became stages S3–S6) | `research/HANDOFF.md`, `research/checkpoints/`, `research/S7_final_report.md` |
| `estimator_reports_old_cache_names/` | reports of the S4 estimators under their old cache names (`8d2408872161` default, `266dd335e39c` final, `4cb21e2e961f` codec + GRU); the S6 logs refer to these names | `results/estimators/estimator_2d856f90b8cd.json` (default), `estimator_8c3b3538564b.json` (final): identical fits, new names since the cache key includes file contents (ASSUMPTIONS item 73) |

The Phase 2 numbers quoted in the stage checkpoints (e.g. Proposed 1.250 vs periodic 2.726) come from
`phase2_results/results/`. Paths written in older checkpoints and logs (`results/summary.md`,
`figures/fig01_…`) refer to these files at their old location.
