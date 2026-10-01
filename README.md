# TVCSI simulator — Task-Value-Aware CSI Feedback

A pure-software simulator and evaluation harness for task-value-aware CSI feedback in an FDD
massive-MIMO downlink. In every slot the gNB chooses a CSI-feedback mode per user: m = 0 means
no feedback (the beam is predicted from history), and m = 1, 2, 3 means a semantic CSI code of
8/16/32 values at 4 bits (32/64/128 bits). The total feedback per slot is capped at C_FB_max,
and the scheduler maximises the task utility U_k = v_k Q(gamma_k). Notation and equation
numbers follow the formulation document (`docs/5G_group_5.pdf`). Every function that implements an
equation names it in its docstring.

> **Current status: dummy models.** The encoder/decoder and the beam predictor are placeholders
> (log-normal decoding noise, persistence prediction). Every figure is stamped
> "DUMMY MODELS". The numbers exercise the pipeline and the schedulers; they are not research
> results about learned CSI feedback.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e . pytest
bash scripts/run_all.sh          # tests, calibration, table, sweeps, figures, summary, dataset
```

`run_all.sh` took 12 minutes on a 16-thread CPU (about 140 CPU-minutes, so expect about 20 minutes on 8 threads). The sweep runs seeds in parallel
(`sim.n_jobs`, 0 = all cores). Individual steps:

| command | output |
|---|---|
| `python -m pytest -q` | the 10 spec tests plus unit tests |
| `python scripts/calibrate_q.py` | `figures/calib_q.png`, saturation/sensitivity check of Q |
| `python scripts/fit_tvcsi_table.py` | `results/tvcsi_table.json` (validation seeds) |
| `python -m tvcsi.sweep [--quick]` | `results/sweep_*.csv`, `default_runs.csv`, `default_loss_traces.npz`, `runtime.csv`, `tables/` |
| `python -m tvcsi.plots` | `figures/fig01..fig14`, `figures/supp01..supp05` (PNG 200 dpi + PDF) |
| `python scripts/make_summary.py` | `results/summary.md`, `results/ordering_gaps.csv` |
| `python scripts/export_dataset.py` | `results/dataset/{train,val,test}.npz` + `meta.json` |

`--quick` (T = 300, 2 seeds, written to `results/quick/`) only checks that the pipeline runs.

All parameters are in `configs/default.yaml`; nothing numeric is hard-coded in `src/`.
`ASSUMPTIONS.md` lists every interpretation, deviation and tuned constant.

## Layout

```
configs/default.yaml   every parameter
src/tvcsi/
  config.py            YAML loading, overrides, derived C_m, C_FB_max, rho
  channel.py           time-correlated ULA multipath channel, pilot estimate, DeepMIMO stub
  codebook.py          DFT codebook, beam gains g_kb = |h^H w_b|^2 (Eq. 3-4)
  link.py              beam selection (14), regret (15), SINR, Q(gamma) (20), utility (21)
  importance.py        message types and Markov importance v_k(t)
  models/              Encoder/Decoder/BeamPredictor interfaces, dummy implementations, factory
  tvcsi.py             regret lookup table, expected utility Uhat, TVCSI (Eq. 32)
  solvers.py           brute force, exact DP (multiple-choice knapsack), greedy TVCSI
  schemes.py           every scheduling scheme behind one decide() interface
  metrics.py           L_sem (36), CVaR (37), switching (38), t-intervals
  sim.py               slot loop, table fitting
  sweep.py             sweeps, default run, runtime benchmark (parallel)
  plots.py             all figures
scripts/               calibrate_q, fit_tvcsi_table, make_summary, export_dataset, run_all.sh
tests/                 pytest suite
results/  figures/     outputs
```

## What happens in a slot (spec 4)

For each slot t, and for every scheme on the same channel and importance realisation:

1. Channel step; true gains g_k(t); pilot estimate hhat = h + CN(0, M / snr_pilot).
2. Importance v_k(t).
3. Shared feedback outcomes for modes 1-3: encode hhat, decode to ghat^m, then beam, regret,
   SINR, Q and true utility. These depend only on the channel, so all schemes share them.
4. Per scheme: the predictor gives (ghat^0, confidence) from that scheme's own history; the
   mode-0 outcome follows.
5. Estimated utilities Uhat[k][m] from predictor output, importance and the regret table only.
6. The scheme picks modes. A hard assert enforces sum_k C_{x_k} <= C_FB_max (only
   `full_feedback` is exempt).
7. The chosen outcome is realised and logged.
8. The predictor history receives what the gNB actually holds: decoded gains if m > 0, its own
   prediction if m = 0. True gains never enter any history.

A test (spec test 10) replaces the true channel with garbage after step 3. The decisions of
every non-genie scheme stay bit-identical.

## Schemes

The four schemes in every main figure (same colour and marker in all figures):

| name | definition |
|---|---|
| **Exact** | Exact DP (multiple-choice knapsack) on the **true** utilities U[k][m] of the slot. A genie upper bound; not implementable. |
| **Optimised** | The same exact DP on the **estimated** utilities Uhat[k][m] of the TVCSI estimator. |
| **Proposed** | Greedy marginal-TVCSI upgrades on the same Uhat: per user, take the upper concave hull of (C_m, Uhat_m). Repeatedly apply the affordable upgrade with the largest positive slope (Uhat_m' - Uhat_m) / (C_m' - C_m). |
| **Baseline** | Periodic round-robin feedback (rule below). `baseline.kind: snr_threshold` swaps in the SNR-threshold baseline. |

Gaps: Exact - Optimised is the information gap, Optimised - Proposed the greedy gap, and
Proposed - Baseline the contribution. The ordering is not forced. Each scheme's predictor history
depends on its own decisions, so trajectories diverge. `results/summary.md` reports every point
where the expected ordering fails.

**Periodic baseline rule (exact).** Fixed mode m_fixed = 2. Period
P = ceil(K * C_mfixed / C_FB_max). User k uses mode m_fixed in slot t iff (t + k) mod P == 0;
everyone else uses mode 0. If the chosen users exceed C_FB_max, users are dropped in ascending
index order until the slot is feasible. If C_FB_max < C_mfixed, no one ever feeds back.

Supplementary schemes (only in `supp*` figures and the summary table):

| name | definition |
|---|---|
| prediction_only | every user mode 0 |
| importance_only | mode 3 to users in descending v_k until the budget is spent (ignores predictor confidence) |
| snr_threshold | mode m_fixed to users in ascending gamma_ref (weakest believed links first) until the budget is spent |
| full_feedback | every user mode 3; violates the budget; reference only |
| Proposed_no_confidence | Proposed with the confidence ignored (pooled regret per mode) |
| Proposed_no_importance | Proposed with v_k = 1 inside Uhat |

## TVCSI estimator (spec 6)

`r_hat[m][bin]` is the mean realised regret of mode m among samples whose predictor confidence
falls in a confidence-quantile bin. It is fitted on validation seeds under `prediction_only` and
random-mode policies. Then

```
gamma_ref_k = snr * max_b ghat0_kb / M
Uhat[k][m]  = v_k * Q_k( gamma_ref_k * (1 - r_hat[m][bin(c_k)]) )
TVCSI[k][m] = (Uhat[k][m] - Uhat[k][0]) / (C_m + eps)          (Eq. 32)
```

By default a table is fitted at every sweep point (ASSUMPTIONS.md, item 21).

## Figures

Figures 1-13: one parameter varied, the rest at the defaults (K = 8, M = B = 32, 30 km/h,
15 dB, crit_share 0.25, budget_frac 0.25, T = 2000 with 100 warm-up slots, 10 test seeds).
Each point is the mean over seeds with a 95 % t-interval band.

| # | file | shows |
|---|---|---|
| 1 | fig01_utility_vs_budget | utility (mean over slots of sum_k U_k) vs budget fraction of K*C_3 |
| 2 | fig02_utility_vs_users | utility per user vs K (budget scales with K) |
| 3 | fig03_utility_vs_speed | utility vs user speed: how fast prediction goes stale |
| 4 | fig04_utility_vs_critical_share | utility vs share of critical users |
| 5 | fig05_utility_vs_snr | utility vs SNR (pilot SNR moves with it) |
| 6 | fig06_regret_vs_speed | mean beam-semantic regret (Eq. 15) vs speed |
| 7 | fig07_regret_vs_budget | mean regret vs budget |
| 8 | fig08_switch_rate_vs_speed | served-beam switch rate (Eq. 38) vs speed |
| 9 | fig09_utility_per_bit_vs_budget | utility / mean feedback bits per slot: efficiency of the bits spent |
| 10 | fig10_bits_to_target_vs_speed | smallest budget fraction reaching 0.9 x full-feedback utility at each speed (interpolated on the budget grid; missing = never reached) |
| 11 | fig11_cvar_vs_budget | CVaR_0.95 of the per-slot semantic loss L_sem (Eqs. 36-37) |
| 12 | fig12_violation_vs_budget | fraction of (critical user, slot) pairs with Q < Q_min = 0.8 |
| 13 | fig13_loss_cdf | empirical CDF of per-slot L_sem at the default point, pooled over seeds |
| 14 | fig14_runtime_vs_K | wall time per slot decision of brute force (K <= 10), DP, greedy and the periodic rule (log scale). Solver curves, not the four schemes. |

`supp01..supp05` repeat figures 1-5. Panel (a) adds the supplementary baselines and the
full-feedback reference; panel (b) shows Proposed against its two ablations. `calib_q` is the Q
calibration plot.

## Plugging in trained models

Set `models: trained` and name the classes in the config:

```yaml
trained:
  encoder: mypkg.models:SemanticEncoder     # subclass of tvcsi.models.base.Encoder
  decoder: mypkg.models:SemanticDecoder     # subclass of Decoder
  predictor: mypkg.models:GRUPredictor      # subclass of BeamPredictor
  kwargs: {checkpoint_dir: /path/to/ckpt}
```

All interfaces are batched over users (see `src/tvcsi/models/base.py`). Train on
`results/dataset/*.npz`. The files hold the same channels, pilot estimates, importance and
message types the simulator uses, split by seed: train 3000-3019, val 2000-2004, test 1000-1009.
