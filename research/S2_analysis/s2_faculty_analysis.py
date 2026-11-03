import json, os, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = r"C:\Users\JSSR DURGESH\Documents\TVCSI_research\side_track_combined\TVCSI_combined"
RES = os.path.join(REPO, "research", "faculty_TaskVoCSI_v2", "results")
OUT = os.path.join(REPO, "research", "S2_analysis")
RUNS = {"digits": os.path.join(RES, "full_digits_20261103"), "surrogate": os.path.join(RES, "full_surrogate_20261103")}
TD = os.path.join(OUT, "tables"); FD = os.path.join(OUT, "figures")
os.makedirs(TD, exist_ok=True); os.makedirs(FD, exist_ok=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 500)

BASE = ["periodic", "age", "uncertainty", "importance", "channel"]
ORDER = ["CTVC-DP", "periodic", "age", "uncertainty", "importance", "channel", "genie", "prediction", "no-importance", "no-risk", "CTVC-greedy"]
COL = {"CTVC-DP": "#eb6834", "periodic": "#1baf7a", "age": "#2a78d6", "uncertainty": "#eda100", "importance": "#e87ba4",
       "channel": "#008300", "genie": "#0b0b0b", "prediction": "#8a8985", "no-importance": "#4a3aa7", "no-risk": "#e34948",
       "CTVC-greedy": "#7a4b2a"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

D = {}
for k, p in RUNS.items():
    D[k] = dict(summary=pd.read_csv(os.path.join(p, "summary.csv")), seed=pd.read_csv(os.path.join(p, "per_seed.csv")),
                paired=pd.read_csv(os.path.join(p, "paired_comparisons.csv")), sweeps=pd.read_csv(os.path.join(p, "sweeps.csv")),
                cal=json.load(open(os.path.join(p, "test_calibration.json"))), diag=json.load(open(os.path.join(p, "model_diagnostics.json"))),
                meta=json.load(open(os.path.join(p, "run_metadata.json"))))

def show(name, df):
    df.to_csv(os.path.join(TD, name), index=False)
    print(f"\n===== {name} =====\n" + df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

# T1
rows = []
for k, d in D.items():
    s = d["summary"].set_index("scheme")
    bb = s.loc[BASE, "utility"].max(); g = s.loc["genie", "utility"]
    for sc in s.index:
        r = s.loc[sc]
        rows.append(dict(run=k, scheme=sc, utility=r.utility, utility_ci95=r.utility_ci95, bits=r.bits, outage=r.outage,
                         ideal_outage=r.ideal_outage, regret=r.regret, cvar95=r.cvar95, switch=r.switch, served=r.served,
                         utility_per_100bits=(r.utility / r.bits * 100 if r.bits != 0 else np.nan),
                         rel_to_best_baseline_pct=100 * (r.utility / bb - 1), rel_to_genie_pct=100 * (r.utility / g - 1)))
show("T1_default_summary.csv", pd.DataFrame(rows))

# T2
rows = []
for k, d in D.items():
    s = d["summary"].set_index("scheme")["utility"]
    for _, r in d["paired"].iterrows():
        x = {"run": k, **r.to_dict()}; x["rel_diff_pct"] = r.mean_difference / s[r.baseline] * 100; rows.append(x)
show("T2_paired.csv", pd.DataFrame(rows))

# T3
rows = []
for k, d in D.items():
    p = d["seed"].pivot(index="seed", columns="scheme", values="utility")
    for b in BASE + ["prediction", "no-importance", "no-risk", "CTVC-greedy"]:
        df = p["CTVC-DP"] - p[b]
        rows.append(dict(run=k, baseline=b, n_seeds=len(df), n_wins=int((df > 0).sum()), frac_wins=(df > 0).mean(),
                         diff_min=df.min(), diff_median=df.median(), diff_max=df.max()))
show("T3_seed_wins.csv", pd.DataFrame(rows))

# T4
rows = []
for k, d in D.items():
    sw = d["sweeps"]
    for (ax, x), g in sw.groupby(["axis", "x"], sort=False):
        g = g.set_index("scheme"); bn = g.loc[BASE, "utility"].idxmax(); c = g.loc["CTVC-DP"]; b = g.loc[bn]
        rows.append(dict(run=k, axis=ax, x=x, ctvc_utility=c.utility, ctvc_ci95=c.utility_ci95, best_baseline=bn,
                         best_utility=b.utility, best_ci95=b.utility_ci95, diff=c.utility - b.utility, genie_utility=g.loc["genie", "utility"],
                         ctvc_bits=c.bits, best_bits=b.bits, ctvc_wins=bool(c.utility - b.utility > 0),
                         ci_separated=bool(c.utility - c.utility_ci95 > b.utility + b.utility_ci95)))
show("T4_sweeps_ctvc_vs_best.csv", pd.DataFrame(rows))

# T5
rows = []
for k, d in D.items():
    v = d["meta"]["calibration_validation"]; t = pd.DataFrame(d["cal"]["per_seed"])
    gc = np.array(t.gain_corr.tolist())
    r = dict(run=k, val_q_mae=v["val_q_mae"], val_q_rmse=v["val_q_rmse"], val_outage_brier=v["val_outage_brier"],
             **{f"val_gain_corr_m{i+1}": v["per_mode_gain_corr"][i] for i in range(3)},
             n_test_seeds=len(t), test_mae_mean=t.mae.mean(), test_mae_std=t.mae.std(), test_brier_mean=t.brier.mean(), test_brier_std=t.brier.std())
    for i in range(3):
        r[f"test_gain_corr_m{i+1}_mean"] = gc[:, i].mean(); r[f"test_gain_corr_m{i+1}_std"] = gc[:, i].std(ddof=1)
    rows.append(r)
show("T5_calibration.csv", pd.DataFrame(rows))

# T6
rows = []
for k, d in D.items():
    dg = d["diag"]
    for m in ["GRU", "persistence", "codec_1", "codec_2", "codec_3"]:
        rows.append(dict(run=k, model=m, accuracy=dg[m]["accuracy"], regret=dg[m]["regret"], timing_ms=np.nan))
    for kk, vv in dg["timings"].items():
        rows.append(dict(run=k, model=kk, accuracy=np.nan, regret=np.nan, timing_ms=vv))
show("T6_diagnostics.csv", pd.DataFrame(rows))

# T7
rows = []
for k, d in D.items():
    p = d["seed"]
    a = p[p.scheme == "CTVC-DP"].set_index("seed"); b = p[p.scheme == "no-risk"].set_index("seed")
    for m in ["utility", "outage", "bits", "cvar95"]:
        df = a[m] - b[m]; n = len(df); hw = stats.t.ppf(0.975, n - 1) * df.std(ddof=1) / np.sqrt(n)
        rows.append(dict(run=k, metric=m, n=n, ctvc_mean=a[m].mean(), norisk_mean=b[m].mean(), mean_diff=df.mean(), ci95_halfwidth=hw))
show("T7_risk_effect.csv", pd.DataFrame(rows))

# ---------------- figures
def save(fig, n): fig.savefig(os.path.join(FD, n), dpi=200, bbox_inches="tight"); plt.close(fig)
RL = list(RUNS)

# F1
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, k in zip(axs, RL):
    s = D[k]["summary"].set_index("scheme").loc[ORDER]
    ec = ["black" if n == "CTVC-DP" else "none" for n in ORDER]
    ax.bar(range(len(ORDER)), s.utility, yerr=s.utility_ci95, color=[COL[n] for n in ORDER], edgecolor=ec, linewidth=1.5, capsize=3)
    bb = s.loc[BASE, "utility"].max(); bn = s.loc[BASE, "utility"].idxmax()
    ax.axhline(bb, color="gray", ls=":", lw=1.2, label=f"best baseline ({bn}) = {bb:.3f}")
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=45, ha="right")
    ax.set_title(f"{k}: default setting (20 seeds, mean ± 95% CI)"); ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_ylim(0, s.utility.max() * 1.3)
axs[0].set_ylabel("Task utility (dimensionless)"); save(fig, "F1_default_utility.png")

# F2
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
lab = BASE + ["prediction", "no-importance", "no-risk", "CTVC-greedy"]
xmax = max((D[k]["paired"].mean_difference + D[k]["paired"].ci95_halfwidth).max() for k in RL)
xmin = min((D[k]["paired"].mean_difference - D[k]["paired"].ci95_halfwidth).min() for k in RL)
for ax, k in zip(axs, RL):
    pr = D[k]["paired"].set_index("baseline")
    for i, b in enumerate(lab):
        r = pr.loc[b]; ax.errorbar(r.mean_difference, i, xerr=r.ci95_halfwidth, fmt="o", color=COL[b], capsize=3)
        ax.text(xmax + 0.05 * (xmax - xmin), i, f"Holm p={r.p_holm:.2g}", va="center", fontsize=8)
    ax.axvline(0, color="k", lw=1); ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab); ax.invert_yaxis()
    ax.set_xlim(xmin - 0.05 * (xmax - xmin), xmax + 0.45 * (xmax - xmin)); ax.set_title(f"{k}: CTVC-DP minus baseline (paired, n=20)")
    ax.set_xlabel("Utility difference (mean ± 95% CI)")
save(fig, "F2_paired_forest.png")

# F3
axes = [("budget_bits", "Bit budget per slot (bits)"), ("speed_kmh", "User speed (km/h)"), ("K", "Number of users K"), ("snr_db", "SNR (dB)")]
fig, axs = plt.subplots(2, 4, figsize=(16, 7), sharey="row")
for i, k in enumerate(RL):
    sw = D[k]["sweeps"]
    for j, (ax_name, xl) in enumerate(axes):
        ax = axs[i, j]; g = sw[sw.axis == ax_name]
        for sc in ["CTVC-DP"] + BASE + ["genie"]:
            h = g[g.scheme == sc].sort_values("x")
            ax.errorbar(h.x, h.utility, yerr=h.utility_ci95, color=COL[sc], marker="o", ms=4, capsize=2,
                        ls="--" if sc == "genie" else "-", lw=2.2 if sc == "CTVC-DP" else 1.2, label=sc, zorder=5 if sc == "CTVC-DP" else 2)
        ax.set_xlabel(xl); ax.set_xticks(sorted(g.x.unique()))
        if j == 0: ax.set_ylabel(f"{k}\nTask utility (dimensionless)")
        if i == 0: ax.set_title(ax_name)
h, l = axs[0, 0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.legend(h, l, loc="upper center", ncol=7, frameon=False)
fig.text(0.5, 0.005, "5 test seeds per point; models frozen (trained at default)", ha="center", fontsize=9)
save(fig, "F3_sweeps.png")

# F4
fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
offs = {"CTVC-DP": (6, 6), "CTVC-greedy": (6, -10), "no-risk": (6, 0), "no-importance": (6, -4), "prediction": (6, 4),
        "periodic": (6, -4), "age": (6, 4), "uncertainty": (6, 4), "importance": (6, -8), "channel": (6, 0), "genie": (6, 4)}
for ax, k in zip(axs, RL):
    s = D[k]["summary"].set_index("scheme")
    for sc in ORDER:
        ax.errorbar(s.loc[sc, "bits"], s.loc[sc, "utility"], xerr=s.loc[sc, "bits_ci95"], yerr=s.loc[sc, "utility_ci95"],
                    fmt="o", color=COL[sc], capsize=2)
    clus = s.loc[[n for n in ORDER if s.loc[n, "bits"] > 100]].sort_values("utility")
    ys = np.linspace(clus.utility.min() - 0.15, clus.utility.max() + 0.25, len(clus))
    for (sc, r), y in zip(clus.iterrows(), ys):
        ax.annotate(sc, (r.bits, r.utility), xytext=(165, y), textcoords="data", fontsize=7.5, color=COL[sc], va="center",
                    arrowprops=dict(arrowstyle="-", color=COL[sc], lw=0.6))
    for sc in ["genie", "prediction"]:
        ax.annotate(sc, (s.loc[sc, "bits"], s.loc[sc, "utility"]), xytext=(6, 6), textcoords="offset points", fontsize=7.5, color=COL[sc])
    ax.set_xlim(-8, 215)
    ax.set_xlabel("Bits per slot (bits)"); ax.set_title(k)
axs[0].set_ylabel("Task utility (dimensionless)"); save(fig, "F4_utility_vs_bits.png")

# F5
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, k in zip(axs, RL):
    p = D[k]["seed"].pivot(index="seed", columns="scheme", values="utility")
    data = [(p["CTVC-DP"] - p[b]).values for b in BASE]
    bp = ax.boxplot(data, positions=range(len(BASE)), widths=0.5, showfliers=False, patch_artist=True)
    for patch, b in zip(bp["boxes"], BASE): patch.set_facecolor(COL[b]); patch.set_alpha(0.35)
    for med in bp["medians"]: med.set_color("k")
    rng = np.random.default_rng(0)
    for i, b in enumerate(BASE): ax.scatter(i + rng.uniform(-.12, .12, len(data[i])), data[i], s=14, color=COL[b], edgecolor="k", lw=.3, zorder=3)
    ax.axhline(0, color="k", lw=1); ax.set_xticks(range(len(BASE))); ax.set_xticklabels(BASE); ax.set_title(f"{k}: per-seed (n=20)")
    ax.set_xlabel("Baseline")
axs[0].set_ylabel("Utility difference (CTVC-DP − baseline)"); save(fig, "F5_per_seed_diff.png")

# F6
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axs[0]; w = 0.2; xs = np.arange(3)
cs = {"digits": "#2a78d6", "surrogate": "#eda100"}
for n, k in enumerate(RL):
    v = D[k]["meta"]["calibration_validation"]["per_mode_gain_corr"]
    t = np.array([c["gain_corr"] for c in D[k]["cal"]["per_seed"]])
    ax.bar(xs + (2 * n - 1.5) * w, v, w, color=cs[k], label=f"{k} validation")
    ax.bar(xs + (2 * n - 0.5) * w, t.mean(0), w, yerr=t.std(0, ddof=1), color=cs[k], alpha=0.5, hatch="//", capsize=2, label=f"{k} test (mean ± sd over seeds)")
ax.set_xticks(xs); ax.set_xticklabels([f"mode {i+1}" for i in range(3)]); ax.set_ylabel("Gain correlation (Pearson r)")
ax.set_title("(a) per-mode gain correlation"); ax.set_ylim(0, 0.55); ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1)
ax = axs[1]; ms = ["GRU", "persistence", "codec_1", "codec_2", "codec_3"]
for n, k in enumerate(RL):
    ax.bar(np.arange(5) + (n - 0.5) * 0.38, [D[k]["diag"][m]["accuracy"] for m in ms], 0.38, color=cs[k], label=k)
ax.set_xticks(range(5)); ax.set_xticklabels(ms, rotation=20); ax.set_ylabel("Beam accuracy (fraction)"); ax.set_title("(b) beam / codec accuracy")
ax.legend(frameon=False, loc="lower right"); ax.set_ylim(0, 1)
save(fig, "F6_calibration_diagnostics.png")
print("\nFigures:", sorted(os.listdir(FD)))
