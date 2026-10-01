"""Export the exact channel/importance realisations used by the simulator for model training.

For every seed in seeds.train / seeds.val / seeds.test the realisation is regenerated with the
same random streams as tvcsi.sim, so a separately trained encoder or GRU predictor sees exactly
the data the simulator evaluates. One file per split, results/dataset/<split>.npz, with seeds
stacked along the user axis:
  h      complex64 [users, time, M]  true channel h_k(t)
  hhat   complex64 [users, time, M]  pilot estimate hhat_k(t) (encoder input)
  v      float32   [users, time]     semantic importance v_k(t)
  types  int8      [users]           message type (0 critical, 1 normal, 2 background)
  seed   int64     [users]           seed the row came from
  user   int64     [users]           user index within that seed
plus meta.json with the configuration snapshot.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from tvcsi.config import load_config, resolve_path
from tvcsi.importance import TYPE_NAMES
from tvcsi.sim import make_realization, rng_streams


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve_path(cfg, cfg["paths"]["results"]) / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    K = cfg["system"]["K"]

    for split in ("train", "val", "test"):
        seeds = cfg["seeds"][split]
        parts = [make_realization(cfg, rng_streams(s)) for s in seeds]
        np.savez(out / f"{split}.npz",
                 h=np.concatenate([p.h for p in parts]).astype(np.complex64),
                 hhat=np.concatenate([p.hhat for p in parts]).astype(np.complex64),
                 v=np.concatenate([p.v for p in parts]).astype(np.float32),
                 types=np.concatenate([p.types for p in parts]).astype(np.int8),
                 seed=np.repeat(seeds, K), user=np.tile(np.arange(K), len(seeds)))
        print(f"{split}: {len(seeds)} seeds -> {len(seeds) * K} user rows of shape "
              f"[{cfg['sim']['T']}, {cfg['system']['M']}]")
    with open(out / "meta.json", "w") as f:
        json.dump({"config": cfg, "type_names": TYPE_NAMES,
                   "layout": "seeds stacked along the user axis; see scripts/export_dataset.py"}, f, indent=1)


if __name__ == "__main__":
    main()
