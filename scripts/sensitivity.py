"""Sensitivity of the conclusions to the constraint parameters that have no single correct value.

* eps (Eq. 39j) and lambda_F, lambda_B, lambda_R (Eq. 39a) only change how a finished run is
  scored, so they are re-evaluated from results/default_runs.csv without new simulations.
* The payload sizes (Eq. 39i) need the per-slot SINR, so the default point is re-simulated once
  per payload scale. The payload never changes a decision (tests/test_phase1_oracle.py), only
  the score.

Writes results/sensitivity.md and results/sensitivity_{eps,lambda,payload}.csv.
"""
from __future__ import annotations

import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from tvcsi.config import apply_overrides, load_config, resolve_path
from tvcsi.metrics import objective_p1, t_interval
from tvcsi.sim import ORACLE, simulate
from tvcsi.estimator import get_estimator
from tvcsi.sweep import get_table

# Implementable schemes only: Exact is a genie and full_feedback ignores the budget.
COMPARE = ["Proposed", "Optimised", "Baseline", "importance_only", "snr_threshold", "prediction_only"]


def fmt(m, h, nd=3):
    if not np.isfinite(m):
        return "n/a"
    return f"{m:.{nd}f} ± {h:.{nd}f}" if np.isfinite(h) else f"{m:.{nd}f}"


def eps_table(df: pd.DataFrame, grid: list[float], order: list[str]) -> pd.DataFrame:
    """Share of seeds in which every critical user meets Pr[Q < Q_min | feasible] <= eps."""
    rows = []
    for name in order:
        w = df[df["scheme"] == name]["crit_worst_violation_feasible"].dropna()
        for eps in grid:
            rows.append(dict(scheme=name, eps=eps, met_share=float((w <= eps).mean()) if len(w) else np.nan,
                             n_seeds=len(w)))
    return pd.DataFrame(rows)


def objective(df: pd.DataFrame, K: int, lF: float, lB: float, lR: float) -> pd.Series:
    return objective_p1(df["utility"], df["bits"], K * df["switch_rate"], df["cvar"], lF, lB, lR)


def lambda_tables(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(a) one lambda varied at a time, the others at their defaults; (b) the full grid."""
    con, K = cfg["constraints"], cfg["system"]["K"]
    sens = con["sensitivity"]
    default = dict(lambda_F=con["lambda_F"], lambda_B=con["lambda_B"], lambda_R=con["lambda_R"])
    one = []
    for lam in ("lambda_F", "lambda_B", "lambda_R"):
        for val in sens[lam]:
            L = {**default, lam: val}
            for name in ["Exact"] + COMPARE:
                g = df[df["scheme"] == name]
                m, h = t_interval(objective(g, K, L["lambda_F"], L["lambda_B"], L["lambda_R"]))
                one.append(dict(varied=lam, value=val, scheme=name, mean=m, half=h))
    grid = []
    for lF, lB, lR in itertools.product(sens["lambda_F"], sens["lambda_B"], sens["lambda_R"]):
        piv = df.assign(obj=objective(df, K, lF, lB, lR)).pivot_table(index="seed", columns="scheme", values="obj")
        diff = piv["Proposed"] - piv["Baseline"]
        m, h = t_interval(diff)
        means = piv[COMPARE].mean()
        grid.append(dict(lambda_F=lF, lambda_B=lB, lambda_R=lR, proposed_minus_baseline=m, half=h,
                         best_implementable=means.idxmax()))
    return pd.DataFrame(one), pd.DataFrame(grid)


def _payload_job(args):
    cfg, seed, schemes = args
    # payload_check_scale is evaluation-only, so every job maps to the base-config estimator
    res = simulate(cfg, seed, schemes, table=get_table(cfg), estimator=get_estimator(cfg))
    keys = ("payload_miss_rate", "payload_miss_rate_feasible", "payload_infeasible_share")
    return [dict(scheme=n, seed=seed, **{k: m[k] for k in keys}) for n, m in res.metrics.items()]


def payload_table(cfg: dict, n_jobs: int) -> pd.DataFrame:
    sens = cfg["constraints"]["sensitivity"]
    schemes = ["Exact", "Optimised", "Proposed", "Baseline"]
    get_estimator(cfg)                     # fit/cache once before the workers start
    jobs, scales = [], []
    for sc in sens["payload_scale"]:
        c = apply_overrides(cfg, {"constraints.payload_check_scale": sc})
        for seed in cfg["seeds"]["test"]:
            jobs.append((c, seed, schemes))
            scales.append(sc)
    with ProcessPoolExecutor(n_jobs) as pool:
        out = list(pool.map(_payload_job, jobs, chunksize=1))
    rows = [{**r, "scale": sc} for sc, rs in zip(scales, out) for r in rs]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    res = resolve_path(cfg, cfg["paths"]["results"])
    con = cfg["constraints"]
    sens = con["sensitivity"]
    df = pd.read_csv(res / "default_runs.csv")
    order = ["Exact"] + COMPARE + [ORACLE]

    eps = eps_table(df, sens["eps"], order)
    one, grid = lambda_tables(df, cfg)
    pay = payload_table(cfg, cfg["sim"]["n_jobs"] or os.cpu_count())
    eps.to_csv(res / "sensitivity_eps.csv", index=False)
    pd.concat([one.assign(kind="one_at_a_time"), grid.assign(kind="grid")]).to_csv(res / "sensitivity_lambda.csv", index=False)
    pay.to_csv(res / "sensitivity_payload.csv", index=False)

    L = ["# Sensitivity of the constraint metrics (default point)", "",
         "The constraint parameters of P1 (Eq. 39) have no single correct value. Defaults are taken "
         "from 3GPP where a standard exists (ASSUMPTIONS.md items 39-41); this file re-checks every "
         "conclusion over a range of values. Mean ± 95% t-interval over the test seeds.", ""]

    L += ["## eps (Eq. 39j): share of seeds where every critical user meets the tolerance", "",
          "Evaluated over each critical user's feasible slots. Default eps = "
          f"{con['eps_violation']:g} (3GPP TS 38.214 5.2.2.1 CQI target BLER). The URLLC target 1e-5 "
          "(TR 38.913 7.9) is not resolvable here: each critical user has about "
          f"{cfg['sim']['T'] - cfg['sim']['warmup']} slots per seed, of which only the feasible ones count.", "",
          "| scheme | " + " | ".join(f"eps = {e:g}" for e in sens["eps"]) + " |",
          "|---|" + "---|" * len(sens["eps"])]
    for name in order:
        e = eps[eps["scheme"] == name].set_index("eps")["met_share"]
        L.append(f"| {name} | " + " | ".join("n/a" if not np.isfinite(e[x]) else f"{e[x]:.1f}" for x in sens["eps"]) + " |")
    L += [""]

    L += ["## lambda (Eq. 39a): P1 objective, one weight varied at a time", "",
          f"Defaults: lambda_F = {con['lambda_F']:g}, lambda_B = {con['lambda_B']:g}, lambda_R = {con['lambda_R']:g}. "
          "No standard value exists for these weights.", ""]
    for lam in ("lambda_F", "lambda_B", "lambda_R"):
        sub = one[one["varied"] == lam]
        vals = sens[lam]
        L += [f"**{lam}**", "", "| scheme | " + " | ".join(f"{v:g}" for v in vals) + " |", "|---|" + "---|" * len(vals)]
        for name in ["Exact"] + COMPARE:
            r = sub[sub["scheme"] == name].set_index("value")
            L.append(f"| {name} | " + " | ".join(fmt(r.loc[v, "mean"], r.loc[v, "half"]) for v in vals) + " |")
        L += [""]
    n = len(grid)
    wins = int((grid["proposed_minus_baseline"] > 0).sum())
    sig_loss = int((grid["proposed_minus_baseline"] + grid["half"] < 0).sum())
    best = grid["best_implementable"].value_counts()
    L += ["### Full grid", "",
          f"Over all {n} combinations of the three weights: Proposed beats Baseline in mean in **{wins} of {n}**; "
          f"it is significantly worse (95% interval below zero) in **{sig_loss} of {n}**.", "",
          "Best implementable scheme (highest mean objective) per combination: " +
          ", ".join(f"{k} {v}" for k, v in best.items()) + ".", ""]

    L += ["## Payload size (Eq. 39i): every payload multiplied by a scale factor", "",
          "Base sizes (TS 22.185): " + ", ".join(f"{t} {b} B" for t, b in cfg["radio"]["payload_bytes"].items()) +
          f"; {cfg['radio']['carrier_prbs']} PRBs shared by K = {cfg['system']['K']} users. The scale "
          "applies to the Shannon check of Eq. 39i only; the measured Q curves keep the base sizes "
          "(a different payload would be a different measured curve).", "",
          "| scale | infeasible share (oracle) | " + " | ".join(f"{s} miss (feasible)" for s in ["Exact", "Optimised", "Proposed", "Baseline"]) + " |",
          "|---|---|" + "---|" * 4]
    for sc in sens["payload_scale"]:
        p = pay[pay["scale"] == sc]
        inf = fmt(*t_interval(p[p["scheme"] == ORACLE]["payload_infeasible_share"]))
        cells = [fmt(*t_interval(p[p["scheme"] == s]["payload_miss_rate_feasible"])) for s in ["Exact", "Optimised", "Proposed", "Baseline"]]
        L.append(f"| {sc:g} | {inf} | " + " | ".join(cells) + " |")
    L += [""]

    (res / "sensitivity.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {res / 'sensitivity.md'}")


if __name__ == "__main__":
    main()
