"""S6 system-level check on the TEST seeds 1000-1009 (default point): dummy models vs the trained
codec with persistence (the S6 gate choice) vs the trained codec with the GRU.

From the repository root:  <venv>/Scripts/python research/logs/S6/evaluate_S6.py
Each model setup gets its own frozen S4 estimator (fitted on train / validation seeds for exactly
those models; cached by config hash). Writes research/logs/S6/eval_20261103/.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from tvcsi.config import apply_overrides, load_config
from tvcsi.estimator import estimator_path, get_estimator
from tvcsi.sim import metrics_rows, simulate
from tvcsi.stats import paired_holm

OUT = Path(__file__).with_name("eval_20261103")
SETUPS = {
    "dummy": {},
    "codec+persistence": {"models": "trained", "trained.predictor": "tvcsi.models.trained:Persistence"},
    "codec+GRU": {"models": "trained", "trained.predictor": "tvcsi.models.trained:GRUPredictor"},
}
SCHEMES = ["Exact", "Proposed", "Proposed_risk", "Baseline", "fill_importance", "fill_age", "fill_periodic",
           "importance_only", "prediction_only", "full_feedback"]


def job(args):
    setup, seed = args
    cfg = apply_overrides(load_config(), SETUPS[setup])
    res = simulate(cfg, seed, SCHEMES, estimator=get_estimator(cfg))
    return metrics_rows(res, setup=setup, seed=seed)


def main() -> None:
    base = load_config()
    for name, ov in SETUPS.items():
        c = apply_overrides(base, ov)
        est = get_estimator(c)                                   # fit (train/val seeds) if not cached
        print(f"{name}: estimator {estimator_path(c).name}, validation gain corr "
              f"{[round(x, 3) for x in est.meta['validation'][est.meta['chosen_min_samples_leaf']]['gain_corr']]}",
              flush=True)
    jobs = [(s, seed) for s in SETUPS for seed in base["seeds"]["test"]]
    with ProcessPoolExecutor(base["sim"]["n_jobs"] or os.cpu_count()) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    OUT.mkdir(parents=True, exist_ok=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_seed.csv", index=False)
    cols = ["utility", "bits", "regret", "violation_rate_feasible", "crit_worst_violation_feasible"]
    summ = df.groupby(["setup", "scheme"], sort=False)[cols].mean()
    summ.to_csv(OUT / "summary.csv")
    pd.set_option("display.width", 200)
    print(summ.round(4).to_string())
    for s in SETUPS:
        h = paired_holm(df[df["setup"] == s], "Proposed", "utility",
                        ["Baseline", "fill_importance", "fill_age", "fill_periodic", "importance_only"])
        h.to_csv(OUT / f"holm_{s.replace('+', '_')}.csv", index=False)
        print(f"\n== {s}: Proposed - other")
        print(h.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
