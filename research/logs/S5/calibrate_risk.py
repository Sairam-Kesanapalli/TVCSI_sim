"""S5: choose the virtual-queue weight beta of Proposed_risk on the VALIDATION seeds.

From the repository root:  <venv>/Scripts/python research/logs/S5/calibrate_risk.py
Writes research/logs/S5/calib_v2_20261103/{calibration.csv, choice.json} (the first attempt,
before the queue fix, is in calib_v1_20261103/). Test seeds are not used.

Rule (fixed before looking at any result): a weight is admissible if, at every calibration point,
the mean over the validation seeds of the worst critical user's violation rate in feasible slots is
<= risk.eps_target. Among admissible weights take the one with the highest mean utility over the
points; if none is admissible, take the weight with the smallest worst-point violation.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from tvcsi.config import apply_overrides, load_config
from tvcsi.estimator import get_estimator
from tvcsi.schemes import risk_scheme
from tvcsi.sim import simulate

OUT = Path(__file__).with_name("calib_v2_20261103")


def job(args):
    point_id, overrides, seed = args
    base = load_config()
    est = get_estimator(base)                                  # frozen base-config estimator
    cfg = apply_overrides(base, overrides)
    r = base["risk"]
    schemes = [risk_scheme(f"beta={b}", float(b), float(r["eps_target"])) for b in r["beta_grid"]]
    res = simulate(cfg, seed, schemes, estimator=est)
    return [dict(point=point_id, seed=seed, beta=float(n.split("=")[1]), **{
        k: m[k] for k in ("utility", "bits", "violation_rate_feasible", "crit_worst_violation_feasible",
                          "reliability_met")}) for n, m in res.metrics.items() if n.startswith("beta=")]


def main() -> None:
    base = load_config()
    get_estimator(base)
    points = base["risk"]["calib_points"]
    jobs = [(i, p, s) for i, p in enumerate(points) for s in base["seeds"]["val"]]
    with ProcessPoolExecutor(base["sim"]["n_jobs"] or os.cpu_count()) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    OUT.mkdir(parents=True, exist_ok=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "calibration.csv", index=False)
    per = df.groupby(["beta", "point"])[["utility", "crit_worst_violation_feasible", "reliability_met"]].mean()
    eps = base["risk"]["eps_target"]
    worst = per["crit_worst_violation_feasible"].groupby("beta").max()
    util = per["utility"].groupby("beta").mean()
    ok = worst[worst <= eps].index
    beta = float(util.loc[ok].idxmax()) if len(ok) else float(worst.idxmin())
    table = pd.DataFrame({"mean_utility": util, "worst_point_violation": worst, "admissible": worst <= eps})
    choice = {"beta": beta, "admissible_betas": [float(b) for b in ok], "eps_target": eps,
              "points": points, "val_seeds": base["seeds"]["val"], "table": table.reset_index().to_dict("records")}
    (OUT / "choice.json").write_text(json.dumps(choice, indent=1))
    pd.set_option("display.width", 200)
    print(per.unstack("point").round(4).to_string())
    print(table.round(4).to_string())
    print("chosen beta:", beta)


if __name__ == "__main__":
    main()
