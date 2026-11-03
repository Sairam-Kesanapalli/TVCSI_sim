"""Slot-level simulator (spec section 4).

All schemes run on the same channel/importance realisation of a seed. Each scheme owns its beam
predictor, because what the gNB knows depends on which users that scheme let feed back.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable

import numpy as np

from .channel import generate_channels, pilot_estimate
from .codebook import beam_gains, dft_codebook
from .config import db2lin, feedback_budget, feedback_costs
from .importance import CRITICAL, assign_types, gamma0_per_user, generate_importance
from .link import beam_regret, select_beam, sinr, task_quality, task_utility
from .metrics import cvar, semantic_loss, switch_indicator
from .models import build_models
from .schemes import DecisionInputs, Scheme, build_scheme
from .tvcsi import TVCSITable, expected_utility, fit_table, gamma_ref

_STREAMS = ("channel", "importance", "pilot", "encoder", "policy")


@dataclass
class Realization:
    """Everything shared by all schemes for one seed."""
    h: np.ndarray       # [K, T, M] true channel
    hhat: np.ndarray    # [K, T, M] pilot estimate
    v: np.ndarray       # [K, T] importance
    types: np.ndarray   # [K] message type
    gamma0: np.ndarray  # [K] quality threshold [dB]


@dataclass
class SimResult:
    metrics: dict = field(default_factory=dict)    # scheme -> {metric: value}
    traces: dict = field(default_factory=dict)     # scheme -> {logged quantity: [T]}
    decisions: dict = field(default_factory=dict)  # scheme -> [T, K] modes
    samples: tuple | None = None                   # (confidence [N], regret per mode [N, M_f+1])


def rng_streams(seed: int) -> dict[str, np.random.Generator]:
    """Independent generators per random source, so adding a scheme never shifts the channel."""
    return {n: np.random.default_rng(s) for n, s in zip(_STREAMS, np.random.SeedSequence(seed).spawn(len(_STREAMS)))}


def make_realization(cfg: dict, rngs: dict[str, np.random.Generator]) -> Realization:
    K = cfg["system"]["K"]
    h = generate_channels(cfg, rngs["channel"])
    snr_pilot = db2lin(cfg["link"]["snr_db"] + cfg["link"]["pilot_snr_offset_db"])
    hhat = pilot_estimate(h, snr_pilot, rngs["pilot"])
    types = assign_types(K, cfg["importance"]["crit_share"])
    v = generate_importance(cfg, types, rngs["importance"])
    return Realization(h, hhat, v, types, gamma0_per_user(cfg, types))


def simulate(cfg: dict, seed: int, scheme_names: list[str | Scheme], table: TVCSITable | None = None,
             tamper: Callable[[SimpleNamespace], None] | None = None, collect_samples: bool = False,
             keep_traces: bool = False, record_decisions: bool = False) -> SimResult:
    """Run T slots for every scheme on the realisation of `seed`.

    scheme_names holds scheme names or ready-made Scheme objects (used for diagnostic probes).

    tamper(shared), if given, is called after the shared per-mode outcome tables of a slot are
    built and may overwrite shared.g_true; it exists to test that no true-channel quantity
    reaches the decisions of non-genie schemes.
    """
    K, M, B = cfg["system"]["K"], cfg["system"]["M"], cfg["system"]["B"]
    T, warmup = cfg["sim"]["T"], cfg["sim"]["warmup"]
    n_fb = len(cfg["feedback"]["code_len"])
    costs, budget = feedback_costs(cfg), feedback_budget(cfg)
    snr_lin, eps = db2lin(cfg["link"]["snr_db"]), cfg["link"]["eps"]
    s_db, q_min = cfg["quality"]["s_db"], cfg["quality"]["q_min"]
    L_h = cfg["feedback"]["history_len"]

    W = dft_codebook(M, B)
    rngs = rng_streams(seed)
    real = make_realization(cfg, rngs)
    encoder, decoder, make_predictor = build_models(cfg, W, rngs["encoder"])
    schemes = [n if isinstance(n, Scheme) else build_scheme(n, cfg, rngs["policy"]) for n in scheme_names]
    if table is None and any(s.estimator for s in schemes):
        raise ValueError("a TVCSI table is required for estimator-based schemes")
    predictors = [make_predictor(K) for _ in schemes]
    crit = real.types == CRITICAL
    users = np.arange(K)
    g0_col = real.gamma0[:, None]

    # Raw per-slot records; every metric is computed from them after the loop.
    rec = {s.name: dict(modes=np.zeros((T, K), dtype=np.int8), beam=np.zeros((T, K), dtype=np.int16),
                        Q=np.zeros((T, K)), d=np.zeros((T, K))) for s in schemes}
    samples_c, samples_d = [], []
    g_true_all = beam_gains(real.h, W)                                 # [K, T, B]
    beams = np.empty((K, n_fb + 1), dtype=np.int64)
    d_all = np.empty((K, n_fb + 1))
    Q_all = np.empty((K, n_fb + 1))

    for t in range(T):
        # Step 1-3: shared quantities. Fresh-feedback outcomes depend only on the channel, so
        # they are computed once for all modes 1..M_f and reused by every scheme.
        hist = real.hhat[:, max(0, t - L_h + 1): t + 1]
        ghat_fb = np.stack([decoder.decode(encoder.encode(hist, m), m) for m in range(1, n_fb + 1)])
        g_true = g_true_all[:, t]
        b_fb = select_beam(ghat_fb).T                                  # [K, M_f]
        d_fb = beam_regret(g_true[:, None, :].repeat(n_fb, 1), b_fb, eps)
        Q_fb = task_quality(snr_lin * np.take_along_axis(g_true, b_fb, 1) / M, g0_col, s_db)
        shared = SimpleNamespace(t=t, g_true=g_true, ghat_fb=ghat_fb, b_fb=b_fb, d_fb=d_fb, Q_fb=Q_fb)
        if tamper is not None:
            tamper(shared)
        v = real.v[:, t]

        for i, scheme in enumerate(schemes):
            # Step 4: this scheme's prediction and the mode-0 outcome it would realise.
            ghat0, conf = predictors[i].predict()
            b0 = select_beam(ghat0)
            beams[:, 0], beams[:, 1:] = b0, shared.b_fb
            d_all[:, 0], d_all[:, 1:] = beam_regret(shared.g_true, b0, eps), shared.d_fb
            Q_all[:, 0] = task_quality(sinr(shared.g_true, b0, snr_lin, M), real.gamma0, s_db)
            Q_all[:, 1:] = shared.Q_fb

            # Step 5-6: estimate utilities from gNB-side information only, then decide.
            gref = gamma_ref(ghat0, snr_lin, M)
            U_hat = None
            if scheme.estimator:
                U_hat = expected_utility(table, gref, conf, v, real.gamma0, s_db,
                                         use_confidence=scheme.estimator != "no_confidence",
                                         use_importance=scheme.estimator != "no_importance")
            U_true = v[:, None] * Q_all if scheme.uses_truth else None
            modes = scheme.decide(DecisionInputs(t, U_hat, gref, v, costs, budget), U_true)
            if not scheme.budget_exempt:
                assert costs[modes].sum() <= budget + 1e-9, f"{scheme.name} exceeded the feedback budget at t={t}"

            # Step 7: realise the chosen outcome.
            r = rec[scheme.name]
            r["modes"][t], r["beam"][t] = modes, beams[users, modes]
            r["Q"][t], r["d"][t] = Q_all[users, modes], d_all[users, modes]
            if collect_samples:
                samples_c.append(conf)
                samples_d.append(d_all.copy())

            # Step 8: the gNB keeps what it actually has: decoded gains or its own prediction.
            held = np.where((modes > 0)[:, None], shared.ghat_fb[np.maximum(modes - 1, 0), users], ghat0)
            predictors[i].update(held, modes > 0)

    res = SimResult(decisions={n: r["modes"] for n, r in rec.items()} if record_decisions else {})
    for s in schemes:
        log = slot_log(rec[s.name], real.v.T, costs, crit, q_min)
        res.metrics[s.name] = summarize(log, warmup, K, int(crit.sum()), cfg["metrics"]["cvar_alpha"])
        if keep_traces:
            res.traces[s.name] = {q: a[warmup:] for q, a in log.items()}
    if collect_samples:
        res.samples = (np.concatenate(samples_c), np.concatenate(samples_d))
    return res


def slot_log(r: dict, v: np.ndarray, costs: np.ndarray, crit: np.ndarray, q_min: float) -> dict:
    """Per-slot system quantities from the raw records of one scheme (v is [T, K])."""
    switch = np.zeros(len(v))
    switch[1:] = switch_indicator(r["beam"][1:], r["beam"][:-1]).mean(axis=1)   # Eq. 38
    return {
        "utility": task_utility(v, r["Q"]).sum(axis=1),                          # sum_k U_k (Eq. 21)
        "loss": semantic_loss(v, r["Q"]),                                        # Eq. 36
        "regret": r["d"].mean(axis=1),
        "switch": switch,
        "bits": costs[r["modes"]].sum(axis=1).astype(float),
        "violations": (r["Q"][:, crit] < q_min).sum(axis=1).astype(float),       # Eq. 34 events
        "fb_users": (r["modes"] > 0).sum(axis=1).astype(float),
    }


def summarize(log: dict, warmup: int, K: int, n_crit: int, alpha: float) -> dict:
    """Per-run metrics over the slots after warm-up (spec 3.9 and section 8)."""
    w = slice(warmup, None)
    utility, bits = log["utility"][w].mean(), log["bits"][w].mean()
    n_slots = len(log["utility"][w])
    return {
        "utility": utility,
        "utility_per_user": utility / K,
        "regret": log["regret"][w].mean(),
        "switch_rate": log["switch"][w].mean(),
        "bits": bits,
        "utility_per_bit": utility / bits if bits > 0 else np.nan,
        "loss": log["loss"][w].mean(),
        "cvar": cvar(log["loss"][w], alpha),
        "violation_rate": log["violations"][w].sum() / (n_crit * n_slots) if n_crit else np.nan,
        "fb_share": log["fb_users"][w].mean() / K,
    }


def fit_policies(cfg: dict) -> list[str]:
    """Policies whose runs on validation seeds provide the TVCSI table samples (spec 6)."""
    return ["prediction_only"] + [f"random_p0={p}" for p in cfg["tvcsi"]["fit_p0"]]


def fit_tvcsi_table(cfg: dict) -> TVCSITable:
    """Fit the regret lookup table on the validation seeds, which are disjoint from test seeds.

    Every (user, slot) sample records the confidence at decision time and the regret each mode
    would have realised in that slot (mode 0 from the policy's own predictor; modes > 0 from the
    shared decoded feedback), so all modes are observed at every confidence level.
    """
    conf, regret = [], []
    for seed in cfg["seeds"]["val"]:
        res = simulate(cfg, seed, fit_policies(cfg), collect_samples=True)
        conf.append(res.samples[0])
        regret.append(res.samples[1])
    return fit_table(np.concatenate(conf), np.concatenate(regret), cfg["tvcsi"]["n_bins"])


def metrics_rows(res: SimResult, **meta) -> list[dict]:
    """One flat record per scheme, tagged with run metadata (sweep, x, seed, ...)."""
    return [{**meta, "scheme": name, **m} for name, m in res.metrics.items()]
