"""Write results/summary.md from the sweep CSVs plus two diagnostic runs at the default point.

Diagnostics (default configuration, test seeds):
  * greedy/DP ratio: along the Proposed trajectory, the estimated utility sum_k Uhat_k of the
    greedy allocation divided by that of the exact DP on the same Uhat;
  * estimator calibration: estimated gain Uhat[k][m] - Uhat[k][0] vs the true gain
    U[k][m] - U[k][0] on the same slots. The probe reads true utilities only to log them; its
    decisions are the greedy allocation on Uhat, i.e. exactly Proposed's.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from tvcsi.config import feedback_budget, feedback_costs, load_config, resolve_path, user_bandwidth_hz
from tvcsi.metrics import t_interval
from tvcsi.plots import MAIN
from tvcsi.schemes import Scheme
from tvcsi.quality import prbs_per_user
from tvcsi.sim import ORACLE, simulate
from tvcsi.solvers import dp_mckp, greedy_tvcsi
from tvcsi.estimator import get_estimator
from tvcsi.stats import paired_holm
from tvcsi.sweep import ONE_D_SWEEPS, get_table

SIMPLE_BASELINES = ["Baseline", "fill_periodic", "fill_age", "fill_uncertainty", "fill_importance",
                    "fill_channel", "importance_only", "snr_threshold"]

METRICS = [("utility", "utility", 3), ("regret", "regret", 3), ("cvar", "CVaR0.95", 3),
           ("violation_rate", "crit. violation", 3), ("bits", "bits/slot", 1),
           ("switch_rate", "switch rate", 3)]
# Phase 1: feasibility oracle and the constraints of P1 that the schemes do not optimise.
CONSTRAINT_METRICS = [("violation_rate", "crit. violation (all)", 3),
                      ("violation_rate_feasible", "crit. violation (feasible)", 3),
                      ("crit_worst_violation_feasible", "worst crit. user (feasible)", 3),
                      ("reliability_met", "39j met (share of seeds)", 2),
                      ("payload_miss_rate", "payload miss 39i (all)", 3),
                      ("payload_miss_rate_feasible", "payload miss (feasible)", 3),
                      ("objective_p1", "P1 objective 39a", 3)]
PAIRS = [("Exact", "Optimised", "information gap"), ("Optimised", "Proposed", "greedy gap"),
         ("Proposed", "Baseline", "contribution")]


def fmt(mean: float, half: float, nd: int) -> str:
    if not np.isfinite(mean):
        return "n/a"
    return f"{mean:.{nd}f} ± {half:.{nd}f}" if np.isfinite(half) else f"{mean:.{nd}f}"


def default_table(df: pd.DataFrame, order: list[str], metrics=METRICS) -> str:
    lines = ["| scheme | " + " | ".join(lbl for _, lbl, _ in metrics) + " |",
             "|---|" + "---|" * len(metrics)]
    for name in order:
        g = df[df["scheme"] == name]
        cells = [fmt(*t_interval(g[m]), nd) for m, _, nd in metrics]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paired_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Per (sweep, x, pair): mean and 95 % t-interval of the per-seed utility difference.
    Schemes share each seed's realisation, so differences are paired by seed."""
    rows = []
    for (sweep, x), g in df.groupby(["sweep", "x"], dropna=False):
        u = g.pivot_table(index="seed", columns="scheme", values="utility")
        for a, b, what in PAIRS:
            mean, half = t_interval(u[a] - u[b])
            rows.append(dict(sweep=sweep, x=x, pair=f"{a} - {b}", what=what, mean=mean, half=half))
    return pd.DataFrame(rows)


def _probe_job(args):
    cfg, seed = args
    costs, budget = feedback_costs(cfg), feedback_budget(cfg)
    rec = {"ratio": [], "est": [], "true": []}

    def decide(inp, U_true):
        m_g = greedy_tvcsi(inp.U_hat, costs, budget)
        m_dp = dp_mckp(inp.U_hat, costs, budget)
        users = np.arange(len(m_g))
        if inp.t >= cfg["sim"]["warmup"]:
            rec["ratio"].append(inp.U_hat[users, m_g].sum() / inp.U_hat[users, m_dp].sum())
            rec["est"].append(inp.U_hat[:, 1:] - inp.U_hat[:, :1])
            rec["true"].append(U_true[:, 1:] - U_true[:, :1])
        return m_g

    probe = Scheme("probe", decide, estimator="learned", uses_truth=True)   # S4: the learned estimator
    simulate(cfg, seed, [probe], table=get_table(cfg), estimator=get_estimator(cfg))
    return np.array(rec["ratio"]), np.concatenate(rec["est"]), np.concatenate(rec["true"])


def diagnostics(cfg: dict) -> dict:
    get_estimator(cfg)                     # fit / cache once before the workers start
    with ProcessPoolExecutor(cfg["sim"]["n_jobs"] or os.cpu_count()) as pool:
        out = list(pool.map(_probe_job, [(cfg, s) for s in cfg["seeds"]["test"]]))
    ratio = np.concatenate([o[0] for o in out])
    est, true = np.concatenate([o[1] for o in out]), np.concatenate([o[2] for o in out])
    seed_ratio = [o[0].mean() for o in out]
    cal = [dict(mode=m + 1, est=est[:, m].mean(), true=true[:, m].mean(),
                corr=np.corrcoef(est[:, m], true[:, m])[0, 1]) for m in range(est.shape[1])]
    return dict(ratio=ratio.mean(), ratio_ci=t_interval(seed_ratio)[1], ratio_min=ratio.min(),
                frac_equal=np.mean(ratio > 1 - 1e-12), calibration=cal)


def random_instance_ratio(n: int = 2000, seed: int = 0) -> float:
    """Greedy/DP value ratio on random utilities (same generator as tests/test_stage2)."""
    rng = np.random.default_rng(seed)
    costs = np.array([0, 32, 64, 128])
    r = []
    for i in range(n):
        K = rng.integers(1, 9)
        V = np.sort(rng.random((K, 4)), axis=1) if i % 2 else rng.random((K, 4))
        b = rng.uniform(0, K * 128)
        users = np.arange(K)
        r.append(V[users, greedy_tvcsi(V, costs, b)].sum() / V[users, dp_mckp(V, costs, b)].sum())
    return float(np.mean(r))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    res = resolve_path(cfg, cfg["paths"]["results"])
    order = cfg["schemes"]["main"] + cfg["schemes"]["supplementary"]

    default = pd.read_csv(res / "default_runs.csv")
    sweeps = pd.concat([pd.read_csv(res / f"sweep_{s}.csv") for s in ONE_D_SWEEPS], ignore_index=True)
    gaps_default = paired_gaps(default.assign(x=0))
    gaps = paired_gaps(sweeps)
    gaps.to_csv(res / "ordering_gaps.csv", index=False)
    diag = diagnostics(cfg)
    fig10 = pd.read_csv(res / "fig10_bits_to_target.csv")
    runtime = pd.read_csv(res / "runtime.csv").groupby(["solver", "K"])["seconds_per_decision"].mean().unstack(0) * 1e3

    n_seeds = default["seed"].nunique()
    L = []
    L += ["# TVCSI simulator — results summary", ""]
    if cfg["models"] == "dummy":
        L += ["> **DUMMY MODELS.** Every number below comes from the placeholder encoder/decoder "
              "(log-normal decoding noise) and the persistence beam predictor. They test the "
              "pipeline and the schedulers; they are **not research results** about learned "
              "semantic CSI feedback.", ""]
    else:
        L += [f"Models: **trained** (`{cfg['trained']['encoder']}`, `{cfg['trained']['decoder']}`, "
              f"`{cfg['trained']['predictor']}`; weights `{cfg['trained']['kwargs'].get('weights')}`, "
              "trained by scripts/train_models.py, research/checkpoints/S6_checkpoint.md).", ""]
    L += [f"Default configuration: K={cfg['system']['K']}, M=B={cfg['system']['M']}, "
          f"speed={cfg['channel']['speed_kmh']:g} km/h, SNR={cfg['link']['snr_db']:g} dB, "
          f"crit_share={cfg['importance']['crit_share']}, budget_frac={cfg['feedback']['budget_frac']} "
          f"(C_FB_max={feedback_budget(cfg):g} bits/slot), T={cfg['sim']['T']} slots "
          f"(first {cfg['sim']['warmup']} excluded), {n_seeds} test seeds. "
          "Values are mean ± 95% t-interval half-width over seeds.", ""]
    if cfg["quality"]["model"] == "measured":
        L += [f"Task quality: **measured**, Q = 1 - BLER of each message type's payload over "
              f"{prbs_per_user(cfg)} PRBs per user of a {cfg['radio']['carrier_prbs']}-PRB carrier "
              "(5G NR PDSCH link-level simulation, data/q_curves.csv; ASSUMPTIONS.md Phase 2).", ""]
    else:
        L += ["Task quality: legacy logistic (spec 3.6).", ""]
    L += ["## Default configuration, all schemes", "", default_table(default, order), "",
          "`full_feedback` ignores the budget (reference only). Utility = mean over slots of "
          "sum_k U_k. Violation = fraction of (critical user, slot) pairs with Q < Q_min.", ""]

    con = cfg["constraints"]
    orc = default[default["scheme"] == ORACLE]
    inf_share = orc["crit_infeasible_share"].mean()
    L += ["## Feasibility oracle and the constraints of P1 (default point)", "",
          f"`{ORACLE}` gives every user the beam with the largest **true** gain in every slot, for "
          "free: perfect CSI everywhere, no budget. No scheme can beat it in any slot. A critical slot "
          "is **feasible** if this oracle reaches Q >= Q_min there; otherwise every scheme violates. "
          "Violations among feasible slots are the part a scheduler can fix. All of this is computed "
          "after the decisions, from the true channel, and no scheme sees it.", "",
          f"* Critical slots that are infeasible even with the true best beam: **{100 * inf_share:.1f}%**. "
          f"Payload (39i) infeasible slots, all users: {100 * orc['payload_infeasible_share'].mean():.1f}%.",
          f"* Constraint parameters (3GPP-cited where a standard exists; ASSUMPTIONS.md items 39-41): "
          f"eps = {con['eps_violation']:g} (TS 38.214); payload {dict(cfg['radio']['payload_bytes'])} bytes (TS 22.185) over "
          f"{user_bandwidth_hz(cfg) / 1e6:g} MHz per user ({cfg['radio']['carrier_prbs']} PRBs / K, TS 38.101-1); "
          f"lambda_F = {con['lambda_F']:g}, lambda_B = {con['lambda_B']:g}, lambda_R = {con['lambda_R']:g} "
          "(no standard value). Sensitivity to all of them: results/sensitivity.md.", "",
          default_table(default, order + [ORACLE], CONSTRAINT_METRICS), "",
          "'39j met' is 1 for a seed when every critical user's violation rate over its feasible slots "
          "is <= eps. The P1 objective is utility - lambda_F bits - lambda_B sum_k s_k - lambda_R CVaR; "
          "it is undefined (n/a) for the oracle, which uses no feedback channel.", ""]

    present = [b for b in SIMPLE_BASELINES if b in set(default["scheme"])]
    if present:
        L += ["## Proposed vs every simple baseline (default point, seed-paired, Holm-corrected)", "",
              "| other | Proposed - other | wins | Holm p | Proposed_risk - other | wins | Holm p |",
              "|---|---|---|---|---|---|---|"]
        hp = paired_holm(default, "Proposed", "utility", present).set_index("other")
        hr = (paired_holm(default, "Proposed_risk", "utility", present).set_index("other")
              if "Proposed_risk" in set(default["scheme"]) else None)
        for b in present:
            a = hp.loc[b]
            row = f"| {b} | {fmt(a['mean_diff'], a['ci95'], 3)} | {a['wins']}/{a['n']} | {a['p_holm']:.2g} |"
            if hr is not None:
                r = hr.loc[b]
                row += f" {fmt(r['mean_diff'], r['ci95'], 3)} | {r['wins']}/{r['n']} | {r['p_holm']:.2g} |"
            L.append(row)
        L += ["", "fill_* are the five faculty baselines (largest mode that fits, ASSUMPTIONS item 64); "
              "Proposed_risk adds the critical-user virtual queue (items 65-66).", ""]

    L += ["## Gap decomposition at the default point (paired by seed)", "",
          "| pair | meaning | utility difference |", "|---|---|---|"]
    for _, r in gaps_default.iterrows():
        L.append(f"| {r['pair']} | {r['what']} | {fmt(r['mean'], r['half'], 3)} |")
    L += [""]

    L += ["## Greedy vs DP", "",
          f"* Random instances (2000, K<=8, random budgets): mean greedy/DP value ratio "
          f"**{random_instance_ratio():.4f}**.",
          f"* On the TVCSI estimates along the Proposed trajectory at the default point: mean "
          f"ratio of estimated utility **{diag['ratio']:.5f}** (± {diag['ratio_ci']:.5f} over seeds), "
          f"worst slot {diag['ratio_min']:.4f}, greedy exactly optimal in "
          f"{100 * diag['frac_equal']:.1f}% of slots.", ""]

    L += ["## Expected ordering Exact >= Optimised >= Proposed >= Baseline", "",
          "Checked at the default point and at every point of the five 1-D sweeps using the paired "
          "per-seed utility difference. A violation is listed when the mean difference is negative; "
          "'significant' means the 95% interval excludes zero.", ""]
    allg = pd.concat([gaps_default.assign(sweep="default", x=np.nan), gaps], ignore_index=True)
    viol = allg[allg["mean"] < 0]
    if viol.empty:
        L += ["No violations: the ordering holds in mean at every point.", ""]
    else:
        L += [f"**The expected ordering is violated at {len(viol)} of {len(allg)} (point, pair) checks.**", "",
              "| sweep | x | pair | mean difference | 95% CI excludes 0 |", "|---|---|---|---|---|"]
        for _, r in viol.iterrows():
            sig = "yes" if np.isfinite(r["half"]) and r["mean"] + r["half"] < 0 else "no"
            x = "default" if r["sweep"] == "default" else f"{r['x']:g}"
            L.append(f"| {r['sweep']} | {x} | {r['pair']} | {fmt(r['mean'], r['half'], 3)} | {sig} |")
        counts = viol.groupby("pair").size()
        L += ["", "Violations per pair: " + ", ".join(f"{p}: {n}" for p, n in counts.items()) + ".", ""]

    wins = gaps[(gaps["pair"] == "Proposed - Baseline") & (gaps["mean"] - gaps["half"] > 0)]
    L += ["### Where Proposed significantly beats Baseline", ""]
    if wins.empty:
        L += ["Nowhere: no sweep point has a paired 95% interval above zero.", ""]
    else:
        L += ["| sweep | x | Proposed - Baseline |", "|---|---|---|"]
        L += [f"| {r['sweep']} | {r['x']:g} | {fmt(r['mean'], r['half'], 3)} |" for _, r in wins.iterrows()]
        L += [""]

    budget = sweeps[sweeps["sweep"] == "budget"].groupby(["x", "scheme"])[["utility", "regret", "bits", "fb_share"]].mean()
    ex = default[default["scheme"] == "Exact"]
    ff = default[default["scheme"] == "full_feedback"]
    viol = default[default["scheme"].isin(MAIN)].groupby("scheme")["violation_rate"].mean()
    L += ["## Other observations (measured, stated plainly)", ""]
    drops = [(a, b) for a, b in zip(sorted(budget.index.levels[0])[:-1], sorted(budget.index.levels[0])[1:])
             if budget.loc[(b, "Proposed"), "utility"] < budget.loc[(a, "Proposed"), "utility"]]
    for a, b in drops:
        pa, pb = budget.loc[(a, "Proposed")], budget.loc[(b, "Proposed")]
        L.append(f"* Proposed utility is **not monotone in the budget**: {pa['utility']:.3f} at budget_frac {a:g} "
                 f"vs {pb['utility']:.3f} at {b:g}. Bits used grow {pb['bits'] / pa['bits']:.2f}x but the share of "
                 f"users fed back per slot only {pb['fb_share'] / pa['fb_share']:.2f}x ({pa['fb_share']:.2f} -> "
                 f"{pb['fb_share']:.2f}): the extra bits go mostly into higher modes for the same users, and mean "
                 f"regret moves {pa['regret']:.3f} -> {pb['regret']:.3f}.")
    above = ex["utility"].mean() > ff["utility"].mean()
    L.append(f"* Exact spends {ex['bits'].mean():.0f} of {feedback_budget(cfg):g} bits per slot on average and "
             f"reaches {ex['utility'].mean():.3f}, {'above' if above else 'below'} full feedback "
             f"({ff['utility'].mean():.3f}). The genie skips feedback whenever the predicted beam is already right "
             "and picks the cheapest mode whose decoded beam is right, so it is an upper bound on scheduling, "
             "not on CSI quality.")
    feas_viol = default[default["scheme"].isin(MAIN)].groupby("scheme")["violation_rate_feasible"].mean()
    L.append(f"* The critical violation rate is {viol.min():.3f}-{viol.max():.3f} (scheme means) for the four "
             f"main schemes at the default point. {100 * inf_share:.1f}% of critical slots are infeasible even with "
             "the true best beam, so every scheme violates there; over the feasible slots the rates are "
             + ", ".join(f"{n} {feas_viol[n]:.3f}" for n in MAIN) + " (fig 12b).")
    og = gaps[gaps["pair"] == "Optimised - Proposed"]
    n_below = int((og["mean"] < 0).sum())
    L.append(f"* Optimised (exact DP on Uhat) is below Proposed (greedy on the same Uhat) at {n_below} of "
             f"{len(og)} sweep points, although greedy is within {100 * (1 - diag['ratio']):.2f}% of the DP "
             "optimum on Uhat itself (above). Where Optimised loses, optimising an inaccurate estimate more "
             "exactly does not translate into higher true utility, and each scheme's predictor history "
             "follows its own decisions.")
    L += [""]

    L += ["## TVCSI estimator calibration (default point, Proposed trajectory)", "",
          "Mean estimated gain of feedback, Uhat[m] - Uhat[0], against the true gain U[m] - U[0] "
          "realised on the same user-slots, and their per-sample correlation.", "",
          "| mode | estimated gain | true gain | correlation |", "|---|---|---|---|"]
    for c in diag["calibration"]:
        L.append(f"| {c['mode']} | {c['est']:.4f} | {c['true']:.4f} | {c['corr']:.3f} |")
    L += ["",
          "Uhat = v E[Q | gNB-side features] from the frozen learned estimator (S4, ASSUMPTIONS items "
          "56-63), fitted on train seeds and chosen on validation seeds; these numbers are on the test "
          "seeds along the Proposed trajectory. The pre-S4 mean-regret plug-in (kept as Proposed_plugin) "
          "had a constant estimated gain of 0 here (correlation 0).", ""]

    L += ["## Feedback needed to reach "
          f"{cfg['sweeps']['bits_target_frac']:g} x full-feedback utility (fig 10)", "",
          "| speed [km/h] | " + " | ".join(MAIN) + " |", "|---|" + "---|" * len(MAIN)]
    for speed, g in fig10.groupby("x"):
        cells = []
        for name in MAIN:
            r = g[g["scheme"] == name].iloc[0]
            cells.append("not reached" if not np.isfinite(r["mean"]) else fmt(r["mean"], r["half"], 3))
        L.append(f"| {speed:g} | " + " | ".join(cells) + " |")
    L += ["", "Entries are budget fractions of K*C_3.", ""]

    L += ["## Solver runtime per slot decision [ms] (fig 14)", "",
          "| K | " + " | ".join(runtime.columns) + " |", "|---|" + "---|" * len(runtime.columns)]
    for K, r in runtime.iterrows():
        L.append(f"| {K} | " + " | ".join("—" if not np.isfinite(v) else f"{v:.3f}" for v in r) + " |")
    L += [""]

    (res / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {res / 'summary.md'}")


if __name__ == "__main__":
    main()
