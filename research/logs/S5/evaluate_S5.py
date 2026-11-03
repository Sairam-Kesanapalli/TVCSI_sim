"""S5 evaluation on the TEST seeds (1000-1009): the default point and the three risk calibration
points, frozen S4 estimator, beta from the validation calibration (configs/default.yaml risk.beta).

From the repository root:  <venv>/Scripts/python research/logs/S5/evaluate_S5.py
Writes research/logs/S5/eval_20261103/: per_seed.csv, summary.csv, holm_<point>.csv.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from tvcsi.config import apply_overrides, load_config
from tvcsi.estimator import get_estimator
from tvcsi.sim import metrics_rows, simulate
from tvcsi.stats import paired_holm
from tvcsi.sweep import get_table

OUT = Path(__file__).with_name("eval_20261103")
SCHEMES = ["Exact", "Optimised", "Proposed", "Proposed_risk", "Proposed_no_importance", "Baseline",
           "fill_periodic", "fill_age", "fill_uncertainty", "fill_importance", "fill_channel",
           "importance_only", "snr_threshold", "full_feedback", "Proposed_plugin"]
BASELINES = ["Baseline", "fill_periodic", "fill_age", "fill_uncertainty", "fill_importance", "fill_channel",
             "importance_only", "snr_threshold"]
POINT_NAMES = ["default", "budget0.1", "speed60", "crit0.5"]


def job(args):
    i, overrides, seed = args
    base = load_config()
    cfg = apply_overrides(base, overrides)
    res = simulate(cfg, seed, SCHEMES, table=get_table(cfg), estimator=get_estimator(base))
    return metrics_rows(res, point=POINT_NAMES[i], seed=seed)


def main() -> None:
    base = load_config()
    get_estimator(base)
    points = base["risk"]["calib_points"]
    for p in points:
        get_table(apply_overrides(base, p))                      # cache tables before the pool
    jobs = [(i, p, s) for i, p in enumerate(points) for s in base["seeds"]["test"]]
    with ProcessPoolExecutor(base["sim"]["n_jobs"] or os.cpu_count()) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    OUT.mkdir(parents=True, exist_ok=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_seed.csv", index=False)
    cols = ["utility", "bits", "violation_rate_feasible", "crit_worst_violation_feasible", "reliability_met"]
    summ = df.groupby(["point", "scheme"], sort=False)[cols].mean()
    summ.to_csv(OUT / "summary.csv")
    pd.set_option("display.width", 220)
    print(summ.round(4).to_string())
    for name in POINT_NAMES:
        d = df[df["point"] == name]
        for ref in ("Proposed", "Proposed_risk"):
            h = paired_holm(d, ref, "utility", BASELINES + (["Proposed"] if ref == "Proposed_risk" else []))
            h.to_csv(OUT / f"holm_{name}_{ref}.csv", index=False)
            print(f"\n== {name}: {ref} - other (utility), Holm over {len(h)} comparisons")
            print(h.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
