"""S4 evaluation on the TEST seeds (1000-1009), default configuration, frozen estimator.

From the repository root:  <venv>/Scripts/python research/logs/S4/evaluate_S4.py
Writes research/logs/S4/eval_20261103/: per_seed.csv, summary.csv, paired.csv, estimator_test.json.

1. All main + supplementary schemes, 10 test seeds: utility, bits, regret, violation rates.
2. Estimator quality on the test seeds along the Proposed trajectory (what the scheme actually
   sees): MAE per mode, gain correlation corr(Qhat_m - Qhat_0, Q_m - Q_0), outage Brier score on
   critical users; and the same for the plug-in (Proposed_plugin), for comparison.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tvcsi.config import load_config
from tvcsi.estimator import SlotContext, diagnostics, features, get_estimator
from tvcsi.sim import metrics_rows, simulate
from tvcsi.sweep import get_table

OUT = Path(__file__).with_name("eval_20261103")


def run_seed(seed: int):
    cfg = load_config()
    est, table = get_estimator(cfg), get_table(cfg)
    schemes = cfg["schemes"]["main"] + cfg["schemes"]["supplementary"]
    ctx, q_min = SlotContext.from_cfg(cfg), cfg["quality"]["q_min"]
    warmup = cfg["sim"]["warmup"]
    logs = {"Proposed": [], "Proposed_plugin": []}

    def observer(name, inp, Q_all, modes):
        if name in logs and inp.t >= warmup:
            logs[name].append((features(inp.ghat0, inp.conf, inp.age, inp.last_mode, inp.types, ctx),
                               inp.U_hat / inp.v[:, None], Q_all, inp.types))

    res = simulate(cfg, seed, schemes, table=table, estimator=est, observer=observer)
    diag = {}
    for name, rows in logs.items():
        X = np.vstack([r[0] for r in rows])
        Qh = np.vstack([r[1] for r in rows])                     # the Q estimate the scheme used
        Q = np.vstack([r[2] for r in rows])
        types = np.concatenate([r[3] for r in rows])
        p_out = est.predict(X)[1] if name == "Proposed" else np.zeros_like(Qh)
        diag[name] = diagnostics(Qh, p_out, np.hstack((Q, (Q < q_min).astype(float))), types)
        if name != "Proposed":
            diag[name]["outage_brier_critical"] = None             # the plug-in has no outage model
    return metrics_rows(res, seed=seed), diag


def main() -> None:
    cfg = load_config()
    get_estimator(cfg)
    get_table(cfg)                                                 # build caches before the pool
    seeds = cfg["seeds"]["test"]
    with ProcessPoolExecutor(cfg["sim"]["n_jobs"] or os.cpu_count()) as pool:
        out = list(pool.map(run_seed, seeds))
    OUT.mkdir(parents=True, exist_ok=False)                        # never overwrite a run
    df = pd.DataFrame([r for rows, _ in out for r in rows])
    df.to_csv(OUT / "per_seed.csv", index=False)

    cols = ["utility", "bits", "regret", "violation_rate", "violation_rate_feasible", "cvar", "switch_rate"]
    summ = df.groupby("scheme", sort=False)[cols].agg(["mean", "sem"])
    summ.to_csv(OUT / "summary.csv")
    piv = df.pivot(index="seed", columns="scheme", values="utility")
    rows = []
    for other in [c for c in piv.columns if c not in ("Proposed",)]:
        d = (piv["Proposed"] - piv[other]).to_numpy()
        half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        rows.append(dict(other=other, mean_diff=d.mean(), ci95=half, wins=int((d > 0).sum()), n=len(d)))
    pd.DataFrame(rows).to_csv(OUT / "paired.csv", index=False)

    agg = {}
    for name in ("Proposed", "Proposed_plugin"):
        ds = [d[name] for _, d in out]
        agg[name] = {k: (np.mean([x[k] for x in ds], axis=0).tolist() if ds[0][k] is not None else None)
                     for k in ("mae", "mae_per_mode", "gain_corr", "mean_true_gain", "mean_est_gain",
                               "outage_brier_critical")}
    (OUT / "estimator_test.json").write_text(json.dumps({"per_seed": [d for _, d in out], "mean": agg}, indent=1))

    pd.set_option("display.width", 200)
    print(df.groupby("scheme", sort=False)[["utility", "bits", "regret", "violation_rate_feasible"]].mean().round(4))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
