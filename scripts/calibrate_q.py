"""Calibration check for the task-quality curves (spec 3.6).

Plots the mean Q_type over channel realisations as a function of beam regret d, i.e. with the
served SINR snr * max_b g * (1 - d) / M, for every message type at the configured SNR. It then
applies the spec's two checks and prints the verdict:
  * saturation: mean Q at d = 0 must lie in [0.05, 0.95] for every type;
  * sensitivity: a 3 dB beam loss (d = 0.5) must change the critical users' mean Q by >= 0.1.
The script does not change any parameter; it only reports, so the choice is recorded by hand
in ASSUMPTIONS.md.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from tvcsi.channel import generate_channels
from tvcsi.codebook import beam_gains, dft_codebook
from tvcsi.config import db2lin, load_config, resolve_path
from tvcsi.importance import TYPE_NAMES
from tvcsi.link import task_quality
from tvcsi.plots import CATEGORICAL, apply_style, save, stamp_dummy

SAT_LO, SAT_HI, MIN_DELTA_Q = 0.05, 0.95, 0.10


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    M, B = cfg["system"]["M"], cfg["system"]["B"]
    snr_lin = db2lin(cfg["link"]["snr_db"])

    h = generate_channels(cfg, np.random.default_rng(cfg["seeds"]["val"][0]), K=64)
    gmax = beam_gains(h, dft_codebook(M, B)).max(axis=-1).ravel()
    d = np.linspace(0, 0.99, 100)
    gamma = snr_lin * gmax[None, :] * (1 - d[:, None]) / M

    apply_style()
    fig, ax = plt.subplots()
    ok = True
    print(f"snr_db = {cfg['link']['snr_db']}, s = {cfg['quality']['s_db']} dB")
    for i, name in enumerate(TYPE_NAMES):
        g0 = cfg["quality"]["gamma0_db"][name]
        q = task_quality(gamma, g0, cfg["quality"]["s_db"]).mean(axis=1)
        q0, q3 = q[0], np.interp(0.5, d, q)
        saturated = not (SAT_LO <= q0 <= SAT_HI)
        ok &= not saturated
        if name == "critical":
            ok &= (q0 - q3) >= MIN_DELTA_Q
        print(f"  {name:10s} gamma0={g0:5.1f} dB  mean Q(d=0)={q0:.3f}  mean Q(3 dB loss)={q3:.3f}  "
              f"dQ={q0 - q3:.3f}{'  SATURATED' if saturated else ''}")
        ax.plot(d, q, color=CATEGORICAL[i], label=f"{name} ($\\gamma_0$={g0:g} dB)")
    ax.axvline(0.5, color="#8a8985", lw=1, ls=":")
    ax.text(0.51, 0.02, "3 dB beam loss", color="#52514e", fontsize=8, va="bottom")
    ax.set(xlabel="beam regret $d$ (Eq. 15)", ylabel="mean task quality $Q$",
           title=f"Task quality vs beam regret (SNR = {cfg['link']['snr_db']:g} dB)", ylim=(0, 1))
    ax.legend(loc="upper right")
    stamp_dummy(fig, cfg)
    save(fig, resolve_path(cfg, cfg["paths"]["figures"]), "calib_q", cfg["plots"]["dpi"])
    print("calibration OK: defaults kept" if ok else "calibration FAILED: adjust snr_db / gamma0")


if __name__ == "__main__":
    main()
