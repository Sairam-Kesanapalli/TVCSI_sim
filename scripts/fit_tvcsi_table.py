"""Fit the TVCSI regret lookup table at the configured (default) operating point (spec 6).

Runs the prediction_only policy and the random-mode policies (tvcsi.fit_p0) on the validation
seeds, which are disjoint from the test seeds, collects (confidence, regret of every mode) per
user and slot, and writes r_hat[m][bin] to tvcsi.table_path (results/tvcsi_table.json).
The sweep fits the same kind of table at each sweep point when tvcsi.refit_per_point is true.
"""
from __future__ import annotations

import argparse

import numpy as np

from tvcsi.config import load_config, resolve_path
from tvcsi.sim import fit_policies
from tvcsi.sweep import get_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert not set(cfg["seeds"]["val"]) & set(cfg["seeds"]["test"]), "validation and test seeds overlap"

    table = get_table(cfg)   # also caches it under results/tables/ for the sweep
    out = resolve_path(cfg, cfg["tvcsi"]["table_path"])
    table.to_json(out, meta={"val_seeds": cfg["seeds"]["val"], "policies": fit_policies(cfg),
                             "speed_kmh": cfg["channel"]["speed_kmh"], "snr_db": cfg["link"]["snr_db"],
                             "K": cfg["system"]["K"], "T": cfg["sim"]["T"]})
    np.set_printoptions(precision=3, suppress=True)
    print(f"wrote {out}")
    print("confidence bin edges:", table.edges)
    print("r_hat[m][bin] (rows m = 0..3):")
    print(table.r_hat)
    print("samples per bin:", table.counts[0])


if __name__ == "__main__":
    main()
