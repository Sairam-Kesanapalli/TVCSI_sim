"""Calibration view of the task-quality model (figures/calib_q).

(a) Q_type(SINR) of every message type at the default point: the measured curves (Q = 1 - BLER,
    data/q_curves.csv) next to the legacy logistic curves, with the distribution of the true
    best-beam SINR and Q_min.
(b) Mean Q over channel realisations as a function of beam regret d, i.e. with the served SINR
    snr * max_b g * (1 - d) / M, for the configured model.

It also prints, per type, the SINR where Q crosses 0.5 and Q_min, and the share of slots in which
the true best beam reaches Q_min. The script reports only; it changes no parameter.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from tvcsi.channel import generate_channels
from tvcsi.codebook import beam_gains, dft_codebook
from tvcsi.config import apply_overrides, db2lin, lin2db, load_config, resolve_path
from tvcsi.importance import TYPE_NAMES
from tvcsi.plots import CATEGORICAL, INK_2, apply_style, footnote, save, stamp_dummy
from tvcsi.quality import build_quality, prbs_per_user


def crossing(x_db: np.ndarray, q: np.ndarray, level: float) -> float:
    """Smallest SINR [dB] at which the curve reaches `level` (NaN if it never does)."""
    hit = np.flatnonzero(q >= level)
    return float(x_db[hit[0]]) if len(hit) else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    M, B = cfg["system"]["M"], cfg["system"]["B"]
    snr_lin = db2lin(cfg["link"]["snr_db"])
    q_min = cfg["quality"]["q_min"]
    types = np.arange(len(TYPE_NAMES))                                   # one user of each type
    model = build_quality(cfg, types)
    legacy = build_quality(apply_overrides(cfg, {"quality.model": "logistic"}), types)

    h = generate_channels(cfg, np.random.default_rng(cfg["seeds"]["val"][0]), K=64)
    gmax = beam_gains(h, dft_codebook(M, B)).max(axis=-1).ravel()
    best_db = lin2db(snr_lin * gmax / M)

    x_db = np.linspace(-15, 30, 1801)
    gamma = np.repeat(db2lin(x_db)[:, None], len(types), axis=1)         # [points, K]
    q_meas, q_leg = model(gamma), legacy(gamma)

    print(f"model = {cfg['quality']['model']}, snr_db = {cfg['link']['snr_db']}, K = {cfg['system']['K']}, "
          f"{prbs_per_user(cfg)} PRBs per user; best-beam SINR p10 / median = "
          f"{np.percentile(best_db, 10):.1f} / {np.median(best_db):.1f} dB")
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6))
    ax1b = ax1.twinx()
    ax1b.hist(best_db, bins=80, range=(-15, 30), color="#d9d8d4", density=True)
    ax1b.set_yticks([])
    ax1b.spines["right"].set_visible(False)
    ax1.set_zorder(ax1b.get_zorder() + 1)
    ax1.patch.set_visible(False)
    for i, name in enumerate(TYPE_NAMES):
        ax1.plot(x_db, q_meas[:, i], color=CATEGORICAL[i], label=f"{name}: measured")
        ax1.plot(x_db, q_leg[:, i], color=CATEGORICAL[i], ls=":", lw=1.4, label=f"{name}: legacy logistic")
        feas = np.mean(model(np.repeat(db2lin(best_db)[:, None], len(types), axis=1))[:, i] >= q_min)
        print(f"  {name:10s} measured: Q=0.5 at {crossing(x_db, q_meas[:, i], 0.5):6.2f} dB, Q={q_min} at "
              f"{crossing(x_db, q_meas[:, i], q_min):6.2f} dB | legacy: Q={q_min} at "
              f"{crossing(x_db, q_leg[:, i], q_min):6.2f} dB | best beam reaches Q_min in {100 * feas:5.1f}% of slots")
    ax1.axhline(q_min, color=INK_2, lw=1, ls="--")
    ax1.text(-14.5, q_min + 0.02, f"$Q_{{min}}$ = {q_min:g}", color=INK_2, fontsize=8)
    ax1.set(xlabel="SINR $\\gamma$ [dB]", ylabel="task quality $Q$", ylim=(0, 1.02),
            title="(a) Q vs SINR; grey: true best-beam SINR distribution")
    ax1.legend(loc="lower right", fontsize=7.2)

    d = np.linspace(0, 0.99, 100)
    g_d = snr_lin * gmax[None, :] * (1 - d[:, None]) / M                # [d, samples]
    for i, name in enumerate(TYPE_NAMES):
        qd = np.array([model(np.column_stack([g] * len(types)))[:, i].mean() for g in g_d])
        ax2.plot(d, qd, color=CATEGORICAL[i], label=name)
    ax2.axvline(0.5, color="#8a8985", lw=1, ls=":")
    ax2.text(0.51, 0.02, "3 dB beam loss", color=INK_2, fontsize=8, va="bottom")
    ax2.set(xlabel="beam regret $d$ (Eq. 15)", ylabel="mean task quality $Q$", ylim=(0, 1.02),
            title=f"(b) mean Q vs beam regret ({cfg['quality']['model']} model)")
    ax2.legend(loc="upper right")
    fig.suptitle(f"Task-quality model at the default point (SNR = {cfg['link']['snr_db']:g} dB, "
                 f"K = {cfg['system']['K']}, {prbs_per_user(cfg)} PRBs per user)", fontsize=11)
    footnote(fig, "measured: Q = 1 - BLER of the type's payload, 5G NR PDSCH link-level simulation "
                  "(scripts/measure_q.py); legacy: logistic of spec 3.6")
    fig.tight_layout()
    stamp_dummy(fig, cfg)
    save(fig, resolve_path(cfg, cfg["paths"]["figures"]), "calib_q", cfg["plots"]["dpi"])


if __name__ == "__main__":
    main()
