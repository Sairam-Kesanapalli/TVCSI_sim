"""All figures (spec section 8): shared styling, the 14 main figures and the supplementary set.

python -m tvcsi.plots renders everything from the CSVs in results/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import load_config, resolve_path  # noqa: E402
from .metrics import t_interval  # noqa: E402

# Colours come from the validated categorical order of the reference palette (light mode).
# Exact is a genie upper bound, so it is drawn as a neutral black reference rather than a hue;
# the three implementable main schemes take the first three slots, which validate all-pairs.
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
STYLE = {
    "Exact":                  dict(color=INK,       marker="s", ls="--"),
    "Optimised":              dict(color="#2a78d6", marker="o", ls="-"),
    "Proposed":               dict(color="#eb6834", marker="^", ls="-"),
    "Baseline":               dict(color="#1baf7a", marker="D", ls="-"),
    "prediction_only":        dict(color="#eda100", marker="v", ls="-."),
    "importance_only":        dict(color="#e87ba4", marker="P", ls="-."),
    "snr_threshold":          dict(color="#008300", marker="X", ls="-."),
    "Proposed_no_confidence": dict(color="#4a3aa7", marker="<", ls=":"),
    "Proposed_no_importance": dict(color="#e34948", marker=">", ls=":"),
    "full_feedback":          dict(color="#8a8985", marker="*", ls=(0, (1, 1))),
}
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (6.4, 4.4),
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# Footer row below the axes (savefig uses bbox_inches="tight", so it is kept in the image).
FOOTER_Y = -0.01


def stamp_dummy(fig, cfg: dict) -> None:
    """Mark figures produced with placeholder models so they are never mistaken for results."""
    if cfg["models"] == "dummy":
        fig.text(0.99, FOOTER_Y, "DUMMY MODELS", ha="right", va="top", fontsize=9,
                 color="#e34948", fontweight="bold", alpha=0.85)


def footnote(fig, text: str) -> None:
    # one row below the stamp so long notes never collide with it
    fig.text(0.01, FOOTER_Y - 0.04, text, ha="left", va="top", fontsize=7.5, color=INK_2, linespacing=1.4)


def save(fig, out_dir: Path, name: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- figures

MAIN = ["Exact", "Optimised", "Proposed", "Baseline"]
SUPP_LEFT = MAIN + ["prediction_only", "importance_only", "snr_threshold", "full_feedback"]
SUPP_RIGHT = ["Proposed", "Proposed_no_confidence", "Proposed_no_importance", "Baseline"]
BUDGET_LABEL = "Feedback budget $C^{FB}_{max}$ [fraction of $K\\,C_3$]"
SPEED_LABEL = "User speed [km/h]"

# (file, sweep, metric, x label, y label, title)
LINE_FIGS = [
    ("fig01_utility_vs_budget", "budget", "utility", BUDGET_LABEL,
     "Utility $\\sum_k U_k$ per slot", "Task utility vs feedback budget"),
    ("fig02_utility_vs_users", "users", "utility_per_user", "Number of users $K$",
     "Utility per user per slot", "Per-user task utility vs number of users"),
    ("fig03_utility_vs_speed", "speed", "utility", SPEED_LABEL,
     "Utility $\\sum_k U_k$ per slot", "Task utility vs user speed"),
    ("fig04_utility_vs_critical_share", "critical_share", "utility", "Critical-user share $|K_c|/K$",
     "Utility $\\sum_k U_k$ per slot", "Task utility vs share of critical users"),
    ("fig05_utility_vs_snr", "snr", "utility", "SNR [dB]",
     "Utility $\\sum_k U_k$ per slot", "Task utility vs SNR"),
    ("fig06_regret_vs_speed", "speed", "regret", SPEED_LABEL,
     "Mean beam regret $d$ (Eq. 15)", "Beam-semantic regret vs user speed"),
    ("fig07_regret_vs_budget", "budget", "regret", BUDGET_LABEL,
     "Mean beam regret $d$ (Eq. 15)", "Beam-semantic regret vs feedback budget"),
    ("fig08_switch_rate_vs_speed", "speed", "switch_rate", SPEED_LABEL,
     "Beam switches [per user per slot]", "Beam-switch rate (Eq. 38) vs user speed"),
    ("fig09_utility_per_bit_vs_budget", "budget", "utility_per_bit", BUDGET_LABEL,
     "Utility per feedback bit [1/bit]", "Utility per mean feedback bit vs budget"),
    ("fig11_cvar_vs_budget", "budget", "cvar", BUDGET_LABEL,
     "CVaR$_{0.95}$ of $L^{sem}$ (Eq. 37)", "Tail semantic loss vs feedback budget"),
    ("fig12_violation_vs_budget", "budget", "violation_rate", BUDGET_LABEL,
     "Critical violation rate $\\Pr[Q_k<Q_{min}]$", "Critical-user violation rate vs budget"),
]
SUPP_FIGS = LINE_FIGS[:5]   # the five utility figures


def scheme_label(name: str, cfg: dict) -> str:
    if name == "Baseline":
        kind = cfg["baseline"]["kind"]
        return f"Baseline (periodic, m={cfg['baseline']['m_fixed']})" if kind == "periodic" else "Baseline (SNR threshold)"
    return {"Exact": "Exact (genie DP)", "Optimised": "Optimised (DP on $\\hat U$)",
            "Proposed": "Proposed (greedy TVCSI)"}.get(name, name.replace("_", " "))


def aggregate(df: pd.DataFrame, metric: str, keys=("x", "scheme")) -> pd.DataFrame:
    """Mean and 95 % t-interval half-width over seeds for every (x, scheme)."""
    rows = []
    for k, g in df.groupby(list(keys), sort=True):
        mean, half = t_interval(g[metric])
        rows.append({**dict(zip(keys, k)), "mean": mean, "half": half, "n": int(g[metric].notna().sum())})
    return pd.DataFrame(rows)


RATE_METRICS = {"regret", "switch_rate", "violation_rate"}   # bounded in [0, 1]


def draw_curves(ax, agg: pd.DataFrame, schemes: list[str], cfg: dict, unit_interval: bool = False) -> None:
    """Mean curve with a 95 % CI band per scheme; bands of [0, 1]-valued metrics are clipped."""
    lo, hi = (0.0, 1.0) if unit_interval else (-np.inf, np.inf)
    for name in schemes:
        a = agg[agg["scheme"] == name].sort_values("x")
        if a.empty or a["mean"].isna().all():
            continue
        st = STYLE[name]
        ax.plot(a["x"], a["mean"], color=st["color"], marker=st["marker"], ls=st["ls"],
                label=scheme_label(name, cfg), markeredgecolor="white", markeredgewidth=0.8)
        half = a["half"].fillna(0)
        ax.fill_between(a["x"], np.clip(a["mean"] - half, lo, hi), np.clip(a["mean"] + half, lo, hi),
                        color=st["color"], alpha=0.14, lw=0)


def _caption(fig, n_seeds: int, extra: str = "") -> None:
    footnote(fig, f"mean over {n_seeds} test seeds, band = 95% t-interval{extra}")


def load_sweep(res_dir: Path, sweep: str) -> pd.DataFrame:
    return pd.read_csv(res_dir / f"sweep_{sweep}.csv")


def line_figure(cfg, res_dir, fig_dir, fname, sweep, metric, xlabel, ylabel, title) -> None:
    df = load_sweep(res_dir, sweep)
    agg = aggregate(df, metric)
    fig, ax = plt.subplots()
    draw_curves(ax, agg, MAIN, cfg, unit_interval=metric in RATE_METRICS)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(loc="best")
    _caption(fig, df["seed"].nunique())
    stamp_dummy(fig, cfg)
    save(fig, fig_dir, fname, cfg["plots"]["dpi"])


def supp_figure(cfg, res_dir, fig_dir, idx, fname, sweep, metric, xlabel, ylabel, title) -> None:
    df = load_sweep(res_dir, sweep)
    agg = aggregate(df, metric)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    draw_curves(ax1, agg, SUPP_LEFT, cfg)
    ax1.set(xlabel=xlabel, ylabel=ylabel, title="(a) main schemes and supplementary baselines")
    ax1.legend(loc="best", fontsize=7.5)
    draw_curves(ax2, agg, SUPP_RIGHT, cfg)
    ax2.set(xlabel=xlabel, title="(b) ablations of the TVCSI estimator")
    ax2.legend(loc="best", fontsize=7.5)
    ax2.tick_params(labelleft=True)
    fig.suptitle(f"Supplementary: {title.lower()}", fontsize=11)
    _caption(fig, df["seed"].nunique(), "; full feedback ignores the budget (reference only)")
    fig.tight_layout()
    stamp_dummy(fig, cfg)
    save(fig, fig_dir, f"supp{idx:02d}_{fname.split('_', 1)[1]}", cfg["plots"]["dpi"])


def first_crossing(x: np.ndarray, y: np.ndarray, target: float) -> float:
    """Smallest x at which y first reaches target, linearly interpolated between grid points."""
    hit = np.flatnonzero(y >= target)
    if len(hit) == 0:
        return np.nan
    i = hit[0]
    if i == 0:
        return float(x[0])
    return float(x[i - 1] + (target - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))


def bits_to_target(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """Fig 10 data: per speed and scheme, the smallest budget fraction whose utility reaches
    frac x the full_feedback utility at that speed.

    The point estimate uses the seed-mean utility curve. The interval uses per-seed crossings
    and is only reported when every seed reaches the target.
    """
    rows = []
    for speed, g in df.groupby("x"):
        u_ff = g[g["scheme"] == "full_feedback"]
        target = frac * u_ff["utility"].mean()
        seed_targets = frac * u_ff.groupby("seed")["utility"].mean()
        for name in MAIN:
            s = g[g["scheme"] == name]
            curve = s.groupby("x2")["utility"].mean().sort_index()
            point = first_crossing(curve.index.values, curve.values, target)
            per_seed = []
            for seed, gs in s.groupby("seed"):
                c = gs.sort_values("x2")
                per_seed.append(first_crossing(c["x2"].values, c["utility"].values, seed_targets[seed]))
            per_seed = np.array(per_seed)
            half = t_interval(per_seed)[1] if np.all(np.isfinite(per_seed)) else np.nan
            rows.append(dict(x=speed, scheme=name, mean=point, half=half,
                             seeds_reaching=int(np.isfinite(per_seed).sum())))
    return pd.DataFrame(rows)


def fig10(cfg, res_dir, fig_dir) -> None:
    df = load_sweep(res_dir, "speed_budget")
    frac = cfg["sweeps"]["bits_target_frac"]
    agg = bits_to_target(df, frac)
    agg.to_csv(res_dir / "fig10_bits_to_target.csv", index=False)
    fig, ax = plt.subplots()
    draw_curves(ax, agg, MAIN, cfg)
    ax.set(xlabel=SPEED_LABEL, ylabel="Budget fraction needed [fraction of $K\\,C_3$]",
           title=f"Feedback needed to reach {frac:g} x full-feedback utility",
           ylim=(0, max(cfg["sweeps"]["budget"]["values"]) * 1.05))
    ax.legend(loc="best")
    footnote(fig, f"budget grid {cfg['sweeps']['budget']['values']}, linear interpolation; "
             "missing point = target never reached\nmean of {} test seeds; band = 95% t-interval "
             "of per-seed crossings, shown only when every seed reaches the target"
             .format(df["seed"].nunique()))
    stamp_dummy(fig, cfg)
    save(fig, fig_dir, "fig10_bits_to_target_vs_speed", cfg["plots"]["dpi"])


def fig13(cfg, res_dir, fig_dir) -> None:
    traces = np.load(res_dir / "default_loss_traces.npz")
    fig, ax = plt.subplots()
    for name in MAIN:
        loss = np.sort(traces[name].ravel())
        st = STYLE[name]
        ax.plot(loss, np.arange(1, len(loss) + 1) / len(loss), color=st["color"], ls=st["ls"],
                label=scheme_label(name, cfg))
    ax.set(xlabel="Per-slot semantic loss $L^{sem}(t)$ (Eq. 36)", ylabel="Empirical CDF",
           title="Distribution of per-slot semantic loss (default configuration)", ylim=(0, 1))
    ax.legend(loc="lower right")
    n_seeds, n_slots = traces[MAIN[0]].shape
    footnote(fig, f"pooled over {n_seeds} test seeds x {n_slots} slots after warm-up")
    stamp_dummy(fig, cfg)
    save(fig, fig_dir, "fig13_loss_cdf", cfg["plots"]["dpi"])


SOLVER_STYLE = {"Brute force": STYLE["Exact"], "DP": STYLE["Optimised"],
                "Greedy": STYLE["Proposed"], "Baseline": STYLE["Baseline"]}


def fig14(cfg, res_dir, fig_dir) -> None:
    df = pd.read_csv(res_dir / "runtime.csv")
    agg = aggregate(df.rename(columns={"K": "x", "solver": "scheme"}), "seconds_per_decision")
    fig, ax = plt.subplots()
    for name, st in SOLVER_STYLE.items():
        a = agg[agg["scheme"] == name].sort_values("x")
        ms, half = a["mean"] * 1e3, a["half"].fillna(0) * 1e3
        ax.plot(a["x"], ms, color=st["color"], marker=st["marker"], ls=st["ls"], label=name,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.fill_between(a["x"], np.maximum(ms - half, 1e-6), ms + half, color=st["color"], alpha=0.14, lw=0)
    ax.set_yscale("log")
    ax.set(xlabel="Number of users $K$", ylabel="Wall time per slot decision [ms]",
           title="Solver runtime vs K (solver curves, not the four schemes)")
    ax.legend(loc="upper left")
    footnote(fig, f"mean over {df['seed'].nunique()} seeds x {cfg['runtime']['n_instances']} "
             f"random instances, budget fraction {cfg['feedback']['budget_frac']}; brute force for "
             f"K <= {cfg['runtime']['brute_force_max_K']}")
    stamp_dummy(fig, cfg)
    save(fig, fig_dir, "fig14_runtime_vs_K", cfg["plots"]["dpi"])


def make_all(cfg: dict) -> None:
    apply_style()
    res_dir = resolve_path(cfg, cfg["paths"]["results"])
    fig_dir = resolve_path(cfg, cfg["paths"]["figures"])
    for spec in LINE_FIGS:
        line_figure(cfg, res_dir, fig_dir, *spec)
    fig10(cfg, res_dir, fig_dir)
    fig13(cfg, res_dir, fig_dir)
    fig14(cfg, res_dir, fig_dir)
    for i, spec in enumerate(SUPP_FIGS, start=1):
        supp_figure(cfg, res_dir, fig_dir, i, *spec)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render all figures from the CSVs in results/.")
    ap.add_argument("--config", default=None)
    make_all(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()
