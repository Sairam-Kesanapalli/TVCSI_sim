"""S3 golden regression: record every scheme's decisions and metrics with the pre-S3 code.

Run once, BEFORE any S3 change, from the repository root:
    <venv>/Scripts/python research/logs/S3/make_golden_pre_S3.py

Writes research/logs/S3/golden_pre_S3.npz. The file describes itself (cases, schemes, metric
names), so tests/test_s3_plumbing.py can re-run exactly the same simulations after the change and
require bit-identical decisions and metrics.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tvcsi.config import ROOT, load_config, resolve_path
from tvcsi.sim import simulate
from tvcsi.tvcsi import TVCSITable

OUT = Path(__file__).with_name("golden_pre_S3.npz")

# Every scheme name accepted by build_scheme (the random_p0=* fitting policies excluded).
SCHEMES = ["Baseline", "Exact", "Optimised", "Proposed", "Proposed_no_confidence",
           "Proposed_no_importance", "periodic", "snr_threshold", "importance_only",
           "prediction_only", "full_feedback"]
SEEDS = [1000, 1001, 1002]
# The default point plus one non-default point (other K, speed and critical share).
CASES = {
    "default": {},
    "K6_speed60_crit05": {"system.K": 6, "channel.speed_kmh": 60.0, "importance.crit_share": 0.5},
}


def run_case(overrides: dict, seed: int):
    """One simulation of all schemes. The estimator schemes use the committed default-point table
    (results/tvcsi_table.json) at every case: only determinism matters here, not calibration."""
    cfg = load_config(overrides=overrides)
    table = TVCSITable.from_json(resolve_path(cfg, cfg["tvcsi"]["table_path"]))
    return simulate(cfg, seed, SCHEMES, table=table, record_decisions=True)


def main() -> None:
    data: dict[str, np.ndarray] = {}
    metric_names = None
    for case, overrides in CASES.items():
        for seed in SEEDS:
            res = run_case(overrides, seed)
            for name in SCHEMES:
                data[f"{case}/{seed}/{name}/decisions"] = res.decisions[name]
            if metric_names is None:
                metric_names = sorted(res.metrics["Proposed"])
            for name, m in res.metrics.items():          # includes the BestBeam_oracle row
                data[f"{case}/{seed}/{name}/metrics"] = np.array([m[k] for k in metric_names], dtype=float)
            print(f"{case} seed {seed}: done", flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    meta = {"commit": commit, "cases": CASES, "seeds": SEEDS, "schemes": SCHEMES,
            "metric_names": metric_names, "numpy": np.__version__, "python": sys.version.split()[0]}
    data["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(OUT, **data)
    print(f"wrote {OUT} ({len(data) - 1} arrays) at commit {commit}")


if __name__ == "__main__":
    main()
