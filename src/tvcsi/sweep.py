"""Parameter sweeps, the default-configuration run and the solver runtime benchmark (spec 8).

python -m tvcsi.sweep [--quick] writes results/sweep_<name>.csv, results/default_runs.csv,
results/default_loss_traces.npz, results/runtime.csv and the per-point TVCSI tables.
"""
from __future__ import annotations

import os

# One BLAS thread per worker: parallelism comes from running seeds in separate processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import apply_overrides, feedback_budget, feedback_costs, load_config, resolve_path  # noqa: E402
from .schemes import periodic_modes  # noqa: E402
from .sim import fit_tvcsi_table, metrics_rows, simulate  # noqa: E402
from .solvers import brute_force, dp_mckp, greedy_tvcsi  # noqa: E402
from .tvcsi import TVCSITable  # noqa: E402

ONE_D_SWEEPS = ("budget", "users", "speed", "critical_share", "snr")


def sweep_points(cfg: dict) -> list[dict]:
    """Every simulated configuration: the five 1-D sweeps, the default point, and the
    speed x budget grid needed for fig 10. Each point varies only the listed parameters."""
    all_schemes = cfg["schemes"]["main"] + cfg["schemes"]["supplementary"]
    sw = cfg["sweeps"]
    points = [dict(sweep="default", param="", x=np.nan, x2=np.nan, overrides={}, schemes=all_schemes)]
    for name in ONE_D_SWEEPS:
        for x in sw[name]["values"]:
            points.append(dict(sweep=name, param=sw[name]["param"], x=x, x2=np.nan,
                               overrides={sw[name]["param"]: x}, schemes=all_schemes))
    for s in sw["speed"]["values"]:
        for b in sw["budget"]["values"]:
            points.append(dict(sweep="speed_budget", param=f"{sw['speed']['param']}|{sw['budget']['param']}",
                               x=s, x2=b, overrides={sw["speed"]["param"]: s, sw["budget"]["param"]: b},
                               schemes=cfg["schemes"]["main"] + ["full_feedback"]))
    return points


def config_key(cfg: dict) -> str:
    """Stable hash of everything that affects a simulation (identical configs share runs)."""
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _table_cfg(cfg: dict) -> dict:
    """The parts of a config that a table fit depends on (schemes/sweeps/paths do not matter)."""
    return {k: v for k, v in cfg.items() if k not in ("sweeps", "schemes", "paths", "plots", "runtime")}


def get_table(cfg: dict) -> TVCSITable:
    """Fit (or load from cache) the table for this configuration on the validation seeds."""
    tables_dir = resolve_path(cfg, cfg["paths"]["results"]) / "tables"
    path = tables_dir / f"{config_key(_table_cfg(cfg))}.json"
    if path.exists():
        return TVCSITable.from_json(path)
    table = fit_tvcsi_table(cfg)
    table.to_json(path, meta={"val_seeds": cfg["seeds"]["val"]})
    return table


def _fit_job(cfg: dict) -> None:
    get_table(cfg)


def _sim_job(args) -> tuple[str, int, list[dict], dict | None]:
    cfg, key, seed, schemes, keep, table = args
    res = simulate(cfg, seed, schemes, table=table or get_table(cfg), keep_traces=keep)
    traces = {n: tr["loss"] for n, tr in res.traces.items()} if keep else None
    return key, seed, metrics_rows(res), traces


def run_sweeps(cfg: dict, n_jobs: int) -> None:
    out = resolve_path(cfg, cfg["paths"]["results"])
    out.mkdir(parents=True, exist_ok=True)
    points = sweep_points(cfg)

    # Identical configurations (e.g. K=8 in the user sweep and 30 km/h in the speed sweep are
    # both the default point) are simulated once and reported in every sweep they belong to.
    unique: dict[str, tuple[dict, list[str]]] = {}
    for p in points:
        pcfg = apply_overrides(cfg, p["overrides"])
        p["key"] = config_key(pcfg)
        prev = unique.get(p["key"], (pcfg, []))[1]
        unique[p["key"]] = (pcfg, sorted(set(prev) | set(p["schemes"]), key=_scheme_order(cfg)))

    t0 = time.time()
    with ProcessPoolExecutor(n_jobs) as pool:
        if cfg["tvcsi"]["refit_per_point"]:
            fixed_table = None   # each worker loads its point's table from the cache
            fit_cfgs = {config_key(_table_cfg(c)): c for c, _ in unique.values()}
            list(pool.map(_fit_job, fit_cfgs.values()))
        else:
            fixed_table = get_table(cfg)   # one table, fitted at the default configuration
        print(f"tables ready for {len(unique)} configurations ({time.time() - t0:.0f}s)", flush=True)

        default_key = next(p["key"] for p in points if p["sweep"] == "default")
        jobs = [(c, k, seed, schemes, k == default_key, fixed_table)
                for k, (c, schemes) in unique.items() for seed in cfg["seeds"]["test"]]
        results = list(pool.map(_sim_job, jobs, chunksize=1))
    print(f"{len(jobs)} simulations done ({time.time() - t0:.0f}s)", flush=True)

    by_key: dict[str, list[dict]] = {}
    traces: dict[str, list] = {}
    for key, seed, rows, tr in results:
        by_key.setdefault(key, []).extend({**r, "seed": seed} for r in rows)
        if tr is not None:
            for name, loss in tr.items():
                traces.setdefault(name, []).append(loss)

    frames: dict[str, list[pd.DataFrame]] = {}
    for p in points:
        df = pd.DataFrame(by_key[p["key"]])
        df = df[df["scheme"].isin(p["schemes"])]
        frames.setdefault(p["sweep"], []).append(df.assign(sweep=p["sweep"], param=p["param"], x=p["x"], x2=p["x2"]))
    for name, fs in frames.items():
        df = pd.concat(fs, ignore_index=True)
        lead = ["sweep", "param", "x", "x2", "seed", "scheme"]
        df = df[lead + [c for c in df.columns if c not in lead]]
        fname = "default_runs.csv" if name == "default" else f"sweep_{name}.csv"
        df.to_csv(out / fname, index=False)
    np.savez_compressed(out / "default_loss_traces.npz", **{n: np.array(v) for n, v in traces.items()})


def _scheme_order(cfg: dict):
    order = cfg["schemes"]["main"] + cfg["schemes"]["supplementary"]
    return lambda n: order.index(n) if n in order else len(order)


def run_runtime(cfg: dict) -> None:
    """Wall time per slot decision of each solver on random instances (fig 14).

    Instances use uniform random utilities and the default budget fraction. The brute-force
    enumeration of mode vectors is cached (it is fixed per K) and excluded from the timing.
    """
    rt = cfg["runtime"]
    rows = []
    for seed in range(rt["n_seeds"]):
        rng = np.random.default_rng(10_000 + seed)
        for K in rt["K_values"]:
            kcfg = apply_overrides(cfg, {"system.K": K})
            costs, budget = feedback_costs(kcfg), feedback_budget(kcfg)
            V = rng.random((rt["n_instances"], K, len(costs)))
            solvers = {"DP": lambda v: dp_mckp(v, costs, budget),
                       "Greedy": lambda v: greedy_tvcsi(v, costs, budget),
                       "Baseline": lambda v: periodic_modes(seed, K, cfg["baseline"]["m_fixed"], costs, budget)}
            if K <= rt["brute_force_max_K"]:
                brute_force(V[0], costs, budget)   # warm the enumeration cache
                solvers["Brute force"] = lambda v: brute_force(v, costs, budget)
            for name, solve in solvers.items():
                t0 = time.perf_counter()
                for v in V:
                    solve(v)
                rows.append(dict(K=K, seed=seed, solver=name,
                                 seconds_per_decision=(time.perf_counter() - t0) / len(V)))
    out = resolve_path(cfg, cfg["paths"]["results"])
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "runtime.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true", help="T=300, 2 test/val seeds (pipeline check only)")
    ap.add_argument("--skip-runtime", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.quick:
        cfg = apply_overrides(cfg, {"sim.T": 300, "sim.warmup": 50, "seeds.test": cfg["seeds"]["test"][:2],
                                    "seeds.val": cfg["seeds"]["val"][:2], "runtime.n_seeds": 2,
                                    "runtime.n_instances": 5, "paths.results": "results/quick"})
    n_jobs = cfg["sim"]["n_jobs"] or os.cpu_count()
    run_sweeps(cfg, n_jobs)
    if not args.skip_runtime:
        t0 = time.time()
        run_runtime(cfg)
        print(f"runtime benchmark done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
