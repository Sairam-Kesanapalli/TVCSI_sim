"""Stage S4: learned per-mode task-quality estimator (replaces the mean-regret plug-in for the
'Optimised' / 'Proposed' schemes; the plug-in stays available as the *_plugin ablations).

Idea (ported from the faculty TaskVoCSI_v2 calibrator, taskv2/policy.py): instead of plugging the
MEAN beam regret into Q, which sits on the flat top of the measured Q curve and makes feedback look
worthless, learn E[Q_k^m | gNB-side features] directly for every feedback mode m, together with
the outage probability Pr[Q_k^m < Q_min], from simulated samples.

* Features (`features`) use only what the gNB holds at decision time: its own predicted beam gains,
  the predictor confidence, feedback age, last mode, message type, plus the configuration (SNR,
  speed, PRBs per user). No true-channel quantity.
* Labels are the counterfactual per-mode outcomes Q_all[k, m] that the simulator computes for
  evaluation anyway (the realised Q had mode m been used in that slot).
* Data: exploration policies (random modes) on the TRAIN seeds, over a small grid of
  configurations around the default point, so one frozen model serves every sweep point.
  Hyper-parameters are chosen on the VALIDATION seeds. Test seeds are never touched here.
* One ExtraTrees forest with 2 (M_f + 1) outputs: [E[Q^0..Q^Mf], Pr[outage^0..outage^Mf]].

The fitted model is cached under estimator.cache_dir with a hash of everything it depends on;
fitting is deterministic (fixed seeds), so the cache can always be rebuilt.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import apply_overrides, db2lin, lin2db, resolve_path
from .importance import CRITICAL
from .quality import prbs_per_user

FEATURE_NAMES = ("gref_db", "margin", "peak_share", "top2_db", "age", "log_age", "never_fed", "conf",
                 "last_mode", "type", "snr_db", "speed_kmh", "n_prb")


@dataclass(frozen=True)
class SlotContext:
    """Configuration-level constants that enter the features."""
    snr_lin: float
    snr_db: float
    speed_kmh: float
    n_prb: int
    M: int
    age_cap: float

    @classmethod
    def from_cfg(cls, cfg: dict) -> "SlotContext":
        return cls(snr_lin=float(db2lin(cfg["link"]["snr_db"])), snr_db=float(cfg["link"]["snr_db"]),
                   speed_kmh=float(cfg["channel"]["speed_kmh"]), n_prb=prbs_per_user(cfg),
                   M=int(cfg["system"]["M"]), age_cap=float(cfg["estimator"]["age_cap"]))


def features(ghat0: np.ndarray, conf: np.ndarray, age: np.ndarray, last_mode: np.ndarray,
             types: np.ndarray, ctx: SlotContext) -> np.ndarray:
    """gNB-side feature matrix [K, len(FEATURE_NAMES)] (float32). Inputs as in DecisionInputs."""
    g = np.sort(np.asarray(ghat0, dtype=float), axis=1)
    top1, top2 = g[:, -1], g[:, -2]
    gref = ctx.snr_lin * top1 / ctx.M                      # believed SINR of the predicted beam
    age_c = np.minimum(np.where(np.isfinite(age), age, ctx.age_cap), ctx.age_cap)
    K = len(top1)
    return np.column_stack((
        lin2db(gref),
        (top1 - top2) / (top1 + 1e-12),                    # how clearly the best beam wins
        top1 / (g.sum(axis=1) + 1e-12),                    # how concentrated the gains are
        lin2db(ctx.snr_lin * top2 / ctx.M),
        age_c,
        np.log1p(age_c),
        (~np.isfinite(age)).astype(float),
        conf,
        last_mode.astype(float),
        types.astype(float),
        np.full(K, ctx.snr_db),
        np.full(K, ctx.speed_kmh),
        np.full(K, float(ctx.n_prb)),
    )).astype(np.float32)


class QualityEstimator:
    """Frozen learned estimator. predict(X) -> (Q_hat [N, M_f+1], p_out [N, M_f+1]), both in [0, 1]."""

    def __init__(self, model, n_modes: int, meta: dict):
        self.model = model
        self.n_modes = n_modes
        self.meta = meta

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Prediction is called once per slot on a few dozen rows: a thread pool per call (n_jobs=-1,
        # used for fitting) would cost far more than the prediction itself. Results are identical.
        self.model.n_jobs = 1
        y = np.clip(self.model.predict(X), 0.0, 1.0)
        return y[:, : self.n_modes], y[:, self.n_modes:]


# ----------------------------------------------------------------------------- data collection

def _fit_policies(cfg: dict) -> list[str]:
    return [f"random_p0={p}" for p in cfg["estimator"]["fit_p0"]]


def collect(cfg: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Run the exploration policies on one seed; return (X [N, F], Y [N, 2 (M_f+1)]).

    Every `slot_stride`-th slot of every user and policy is kept (consecutive slots are strongly
    correlated, so this loses little information and keeps the training set small).
    """
    from .sim import simulate   # local import: sim imports this module

    ctx = SlotContext.from_cfg(cfg)
    stride = int(cfg["estimator"]["slot_stride"])
    q_min = cfg["quality"]["q_min"]
    X, Y = [], []

    def observer(name, inp, Q_all, modes):
        if inp.t % stride:
            return
        X.append(features(inp.ghat0, inp.conf, inp.age, inp.last_mode, inp.types, ctx))
        Y.append(np.hstack((Q_all, (Q_all < q_min).astype(float))))

    simulate(cfg, seed, _fit_policies(cfg), observer=observer)
    return np.vstack(X), np.vstack(Y).astype(np.float32)


def training_configs(cfg: dict) -> list[dict]:
    """The default point plus one-at-a-time variations listed in estimator.train_grid."""
    out = [cfg]
    for param, values in cfg["estimator"]["train_grid"].items():
        for v in values:
            c = apply_overrides(cfg, {param: v})
            if c != cfg:
                out.append(c)
    return out


def _seed_split(seeds: list[int], j: int, n: int) -> list[int]:
    """n seeds for configuration j, cycling through the seed list."""
    return [seeds[(j * n + r) % len(seeds)] for r in range(n)]


def build_dataset(cfg: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Samples from every training configuration on the given seed split ('train' or 'val')."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    seeds = cfg["seeds"][split]
    n = int(cfg["estimator"]["seeds_per_config"])
    jobs = [(c, s) for j, c in enumerate(training_configs(cfg)) for s in _seed_split(seeds, j, min(n, len(seeds)))]
    n_jobs = cfg["sim"]["n_jobs"] or os.cpu_count()
    with ProcessPoolExecutor(n_jobs) as pool:          # results come back in job order (deterministic)
        parts = list(pool.map(_collect_job, jobs))
    return np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts])


def _collect_job(args):
    return collect(*args)


# ----------------------------------------------------------------------------- fitting

def diagnostics(Q_hat: np.ndarray, p_hat: np.ndarray, Y: np.ndarray, types: np.ndarray) -> dict:
    """MAE per mode, gain correlation corr(Qhat_m - Qhat_0, Q_m - Q_0) per mode m > 0, and the
    Brier score of the outage probability on critical users."""
    nm = Q_hat.shape[1]
    Q, out = Y[:, :nm], Y[:, nm:]
    gain_corr = []
    for m in range(1, nm):
        a, b = Q_hat[:, m] - Q_hat[:, 0], Q[:, m] - Q[:, 0]
        gain_corr.append(float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else 0.0)
    crit = types == CRITICAL
    return {"n": int(len(Y)),
            "mae_per_mode": [float(x) for x in np.abs(Q_hat - Q).mean(axis=0)],
            "mae": float(np.abs(Q_hat - Q).mean()),
            "gain_corr": gain_corr,
            "mean_true_gain": [float(x) for x in (Q[:, 1:] - Q[:, :1]).mean(axis=0)],
            "mean_est_gain": [float(x) for x in (Q_hat[:, 1:] - Q_hat[:, :1]).mean(axis=0)],
            "outage_brier_critical": float(((p_hat[crit] - out[crit]) ** 2).mean()) if crit.any() else None}


def _forest(cfg: dict, leaf: int):
    from sklearn.ensemble import ExtraTreesRegressor
    e = cfg["estimator"]
    return ExtraTreesRegressor(n_estimators=int(e["n_estimators"]), min_samples_leaf=int(leaf),
                               max_features=float(e["max_features"]), random_state=int(e["random_state"]),
                               n_jobs=-1)


def fit_estimator(cfg: dict, log=print) -> QualityEstimator:
    """Collect train/val data, pick min_samples_leaf on validation, refit on train, freeze."""
    t0 = time.time()
    nm = len(cfg["feedback"]["code_len"]) + 1
    Xtr, Ytr = build_dataset(cfg, "train")
    Xva, Yva = build_dataset(cfg, "val")
    log(f"estimator data: train {len(Xtr)} samples, val {len(Xva)} samples ({time.time() - t0:.0f}s)")
    type_col = FEATURE_NAMES.index("type")
    scores = {}
    best = None
    for leaf in cfg["estimator"]["min_samples_leaf"]:
        model = _forest(cfg, leaf).fit(Xtr, Ytr)
        est = QualityEstimator(model, nm, {})
        d = diagnostics(*est.predict(Xva), Yva, Xva[:, type_col].astype(int))
        scores[int(leaf)] = d
        log(f"  leaf {leaf}: val MAE {d['mae']:.4f}  gain corr {np.round(d['gain_corr'], 3).tolist()}")
        # selection rule: highest mean validation gain correlation (ranking the value of feedback
        # is what the scheduler needs); MAE breaks near-ties
        key = (round(float(np.mean(d["gain_corr"])), 3), -d["mae"])
        if best is None or key > best[0]:
            best = (key, int(leaf), est)
    _, leaf, est = best
    est.meta = {"chosen_min_samples_leaf": leaf, "validation": scores, "n_train": int(len(Xtr)),
                "n_val": int(len(Xva)), "features": list(FEATURE_NAMES), "fit_seconds": time.time() - t0,
                "train_configs": len(training_configs(cfg))}
    log(f"chosen min_samples_leaf = {leaf} ({time.time() - t0:.0f}s)")
    return est


# ----------------------------------------------------------------------------- cache

def _fit_spec(cfg: dict) -> dict:
    """Everything the fitted model depends on (the hash names the cache file)."""
    keep = ("system", "channel", "link", "quality", "radio", "importance", "feedback", "dummy",
            "models", "trained", "estimator", "sim")
    spec = {k: cfg[k] for k in keep if not (k == "trained" and cfg["models"] == "dummy")}
    spec["seeds"] = {"train": cfg["seeds"]["train"], "val": cfg["seeds"]["val"]}
    spec["estimator"] = {k: v for k, v in cfg["estimator"].items() if k != "cache_dir"}
    return spec


def estimator_key(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(_fit_spec(cfg), sort_keys=True, default=str).encode()).hexdigest()[:12]


def estimator_path(cfg: dict) -> Path:
    return resolve_path(cfg, cfg["estimator"]["cache_dir"]) / f"estimator_{estimator_key(cfg)}.joblib"


_LOADED: dict[str, QualityEstimator] = {}


def load_estimator(path: str | Path) -> QualityEstimator:
    """Load a cached estimator file once per process (sweep workers call this for every job)."""
    import joblib
    key = str(path)
    if key not in _LOADED:
        _LOADED[key] = joblib.load(path)
    return _LOADED[key]


def get_estimator(cfg: dict, log=print) -> QualityEstimator:
    """Load the frozen estimator for this configuration from the cache, or fit and cache it.

    Pass the BASE configuration: the model is frozen and reused at every sweep point (ASSUMPTIONS
    item 58); a sweep-point configuration would hash to a different file and trigger a new fit."""
    import joblib
    path = estimator_path(cfg)
    if path.exists():
        return load_estimator(path)
    est = fit_estimator(cfg, log=log)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(est, path, compress=3)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(est.meta, f, indent=1)
    return est
