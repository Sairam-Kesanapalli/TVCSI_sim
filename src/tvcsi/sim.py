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
from .config import db2lin, feedback_budget, feedback_costs, user_bandwidth_hz
from .importance import CRITICAL, assign_types, generate_importance, per_type
from .link import beam_regret, deliverable_bits, select_beam, sinr, task_utility
from .metrics import cvar, objective_p1, semantic_loss, switch_indicator
from .models import build_models
from .quality import QualityModel, build_quality
from .schemes import DecisionInputs, Scheme, build_scheme
from .tvcsi import TVCSITable, expected_utility, fit_table, gamma_ref

_STREAMS = ("channel", "importance", "pilot", "encoder", "policy")

# Name of the evaluation-only best-beam oracle row in every result (not a scheme: it never decides).
ORACLE = "BestBeam_oracle"


@dataclass
class EvalContext:
    """Evaluation-only quantities of one run (true channel + constraint parameters).

    Built after the slot loop from the true channel. No scheme ever receives it.
    """
    crit: np.ndarray              # [K] bool, critical users
    q_min: float
    eps: float                    # epsilon_k of Eq. 34 / 39j
    crit_feasible: np.ndarray     # [T, K] bool: critical AND the true best beam reaches Q >= Q_min
    payload_bits: np.ndarray      # [K] semantic payload S_k per slot (Eq. 39i)
    payload_feasible: np.ndarray  # [T, K] bool: the true best beam can deliver S_k in the slot
    bandwidth_hz: float
    Ts_s: float
    lambdas: tuple[float, float, float]   # (lambda_F, lambda_B, lambda_R) of Eq. 39a


def best_beam_oracle(g_true_all: np.ndarray, snr_lin: float, M: int,
                     quality: QualityModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True best-beam oracle: for every user and slot, the codebook beam with the largest TRUE gain.

    This is perfect, free, instantaneous CSI for everyone, so no scheme can do better in any slot
    (Q is increasing in gamma, gamma in the beam gain). Returns (beam [T, K], gamma [T, K], Q [T, K]).
    """
    beam = g_true_all.argmax(axis=-1).T
    gamma = (snr_lin * g_true_all.max(axis=-1) / M).T
    return beam, gamma, quality(gamma, user_axis=-1)


def eval_context(cfg: dict, real: "Realization", gamma_best: np.ndarray, Q_best: np.ndarray) -> EvalContext:
    con = cfg["constraints"]
    crit = real.types == CRITICAL
    q_min = cfg["quality"]["q_min"]
    pay = 8.0 * con["payload_check_scale"] * per_type(cfg["radio"]["payload_bytes"], real.types)  # S_k [bits]
    W, Ts = user_bandwidth_hz(cfg), cfg["channel"]["Ts_s"]
    return EvalContext(
        crit=crit, q_min=q_min, eps=con["eps_violation"],
        crit_feasible=crit[None, :] & (Q_best >= q_min),
        payload_bits=pay, payload_feasible=deliverable_bits(gamma_best, W, Ts) >= pay[None, :],
        bandwidth_hz=W, Ts_s=Ts, lambdas=(con["lambda_F"], con["lambda_B"], con["lambda_R"]))


@dataclass
class Realization:
    """Everything shared by all schemes for one seed."""
    h: np.ndarray       # [K, T, M] true channel
    hhat: np.ndarray    # [K, T, M] pilot estimate
    v: np.ndarray       # [K, T] importance
    types: np.ndarray   # [K] message type


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
    return Realization(h, hhat, v, types)


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
    L_h = cfg["feedback"]["history_len"]

    W = dft_codebook(M, B)
    rngs = rng_streams(seed)
    real = make_realization(cfg, rngs)
    quality = build_quality(cfg, real.types)                         # Q_k(gamma) of every user
    encoder, decoder, make_predictor = build_models(cfg, W, rngs["encoder"])
    schemes = [n if isinstance(n, Scheme) else build_scheme(n, cfg, rngs["policy"]) for n in scheme_names]
    if table is None and any(s.estimator for s in schemes):
        raise ValueError("a TVCSI table is required for estimator-based schemes")
    predictors = [make_predictor(K) for _ in schemes]
    # S3: per-scheme feedback bookkeeping, kept here (not in the predictor) so it is the same for
    # every predictor type. age = slots since the user's last mode > 0 feedback under this scheme
    # (1 = previous slot, inf = never), the convention of PersistencePredictor.age.
    ages = [np.full(K, np.inf) for _ in schemes]
    last_modes = [np.zeros(K, dtype=np.int64) for _ in schemes]
    users = np.arange(K)

    # Raw per-slot records; every metric is computed from them after the loop.
    rec = {s.name: dict(modes=np.zeros((T, K), dtype=np.int8), beam=np.zeros((T, K), dtype=np.int16),
                        Q=np.zeros((T, K)), d=np.zeros((T, K)), gamma=np.zeros((T, K))) for s in schemes}
    samples_c, samples_d = [], []
    g_true_all = beam_gains(real.h, W)                                 # [K, T, B]
    beams = np.empty((K, n_fb + 1), dtype=np.int64)
    d_all = np.empty((K, n_fb + 1))
    Q_all = np.empty((K, n_fb + 1))
    gamma_all = np.empty((K, n_fb + 1))

    for t in range(T):
        # Step 1-3: shared quantities. Fresh-feedback outcomes depend only on the channel, so
        # they are computed once for all modes 1..M_f and reused by every scheme.
        hist = real.hhat[:, max(0, t - L_h + 1): t + 1]
        ghat_fb = np.stack([decoder.decode(encoder.encode(hist, m), m) for m in range(1, n_fb + 1)])
        g_true = g_true_all[:, t]
        b_fb = select_beam(ghat_fb).T                                  # [K, M_f]
        d_fb = beam_regret(g_true[:, None, :].repeat(n_fb, 1), b_fb, eps)
        gamma_fb = snr_lin * np.take_along_axis(g_true, b_fb, 1) / M
        Q_fb = quality(gamma_fb, user_axis=0)
        shared = SimpleNamespace(t=t, g_true=g_true, ghat_fb=ghat_fb, b_fb=b_fb, d_fb=d_fb, Q_fb=Q_fb,
                                 gamma_fb=gamma_fb)
        if tamper is not None:
            tamper(shared)
        v = real.v[:, t]

        for i, scheme in enumerate(schemes):
            # Step 4: this scheme's prediction and the mode-0 outcome it would realise.
            ghat0, conf = predictors[i].predict()
            b0 = select_beam(ghat0)
            beams[:, 0], beams[:, 1:] = b0, shared.b_fb
            d_all[:, 0], d_all[:, 1:] = beam_regret(shared.g_true, b0, eps), shared.d_fb
            gamma_all[:, 0], gamma_all[:, 1:] = sinr(shared.g_true, b0, snr_lin, M), shared.gamma_fb
            Q_all[:, 0] = quality(gamma_all[:, 0])
            Q_all[:, 1:] = shared.Q_fb

            # Step 5-6: estimate utilities from gNB-side information only, then decide.
            gref = gamma_ref(ghat0, snr_lin, M)
            U_hat = None
            if scheme.estimator:
                U_hat = expected_utility(table, gref, conf, v, quality,
                                         use_confidence=scheme.estimator != "no_confidence",
                                         use_importance=scheme.estimator != "no_importance")
            U_true = v[:, None] * Q_all if scheme.uses_truth else None
            inp = DecisionInputs(t, U_hat, gref, v, costs, budget,
                                 ghat0=ghat0.copy(), conf=conf.copy(), age=ages[i].copy(),
                                 last_mode=last_modes[i].copy(), types=real.types.copy())
            modes = scheme.decide(inp, U_true)
            if not scheme.budget_exempt:
                assert costs[modes].sum() <= budget + 1e-9, f"{scheme.name} exceeded the feedback budget at t={t}"

            # Step 7: realise the chosen outcome.
            r = rec[scheme.name]
            r["modes"][t], r["beam"][t] = modes, beams[users, modes]
            r["Q"][t], r["d"][t] = Q_all[users, modes], d_all[users, modes]
            r["gamma"][t] = gamma_all[users, modes]
            if collect_samples:
                samples_c.append(conf)
                samples_d.append(d_all.copy())

            # Step 8: the gNB keeps what it actually has: decoded gains or its own prediction.
            held = np.where((modes > 0)[:, None], shared.ghat_fb[np.maximum(modes - 1, 0), users], ghat0)
            predictors[i].update(held, modes > 0)
            ages[i] = np.where(modes > 0, 1.0, ages[i] + 1.0)
            last_modes[i] = np.asarray(modes, dtype=np.int64).copy()

    # Evaluation only, after every decision has been taken: the true best-beam oracle and the
    # constraints of P1 that the schemes do not optimise (39a penalties, 39i payload, 39j).
    b_best, gamma_best, Q_best = best_beam_oracle(g_true_all, snr_lin, M, quality)
    ev = eval_context(cfg, real, gamma_best, Q_best)
    rec[ORACLE] = dict(modes=None, beam=b_best, Q=Q_best, d=np.zeros((T, K)), gamma=gamma_best)

    alpha = cfg["metrics"]["cvar_alpha"]
    res = SimResult(decisions={s.name: rec[s.name]["modes"] for s in schemes} if record_decisions else {})
    for name in [s.name for s in schemes] + [ORACLE]:
        log = slot_log(rec[name], real.v.T, costs, ev)
        res.metrics[name] = summarize(log, warmup, K, ev, alpha)
        if keep_traces:
            res.traces[name] = {q: a[warmup:] for q, a in log.items()}
    if collect_samples:
        res.samples = (np.concatenate(samples_c), np.concatenate(samples_d))
    return res


def slot_log(r: dict, v: np.ndarray, costs: np.ndarray, ev: EvalContext) -> dict:
    """Per-slot system quantities from the raw records of one scheme (v is [T, K]).

    r["modes"] is None for the oracle row, which uses no feedback channel: its bits are NaN.
    """
    T = len(v)
    switch = np.zeros(T)
    switch[1:] = switch_indicator(r["beam"][1:], r["beam"][:-1]).mean(axis=1)   # Eq. 38
    viol = (r["Q"] < ev.q_min) & ev.crit[None, :]                                 # Eq. 34 events
    payload_ok = deliverable_bits(r["gamma"], ev.bandwidth_hz, ev.Ts_s) >= ev.payload_bits[None, :]
    if r["modes"] is None:
        bits, fb_users = np.full(T, np.nan), np.full(T, np.nan)
    else:
        bits = costs[r["modes"]].sum(axis=1).astype(float)
        fb_users = (r["modes"] > 0).sum(axis=1).astype(float)
    return {
        "utility": task_utility(v, r["Q"]).sum(axis=1),                          # sum_k U_k (Eq. 21)
        "loss": semantic_loss(v, r["Q"]),                                        # Eq. 36
        "regret": r["d"].mean(axis=1),
        "switch": switch,
        "bits": bits,
        "violations": viol.sum(axis=1).astype(float),
        "violations_feasible": (viol & ev.crit_feasible).sum(axis=1).astype(float),
        "payload_miss": (~payload_ok).sum(axis=1).astype(float),                 # Eq. 39i events
        "payload_miss_feasible": (~payload_ok & ev.payload_feasible).sum(axis=1).astype(float),
        "fb_users": fb_users,
        "viol_feasible_k": viol & ev.crit_feasible,                              # [T, K], for 39j per user
    }


def summarize(log: dict, warmup: int, K: int, ev: EvalContext, alpha: float) -> dict:
    """Per-run metrics over the slots after warm-up (spec 3.9 and section 8, plus Eq. 39 checks).

    Feasible = the true best beam meets the requirement in that slot. A violation in an
    infeasible slot is unavoidable for every scheme; violations among feasible slots are the
    scheduler's (or the CSI's) fault.
    """
    w = slice(warmup, None)
    utility, bits = log["utility"][w].mean(), log["bits"][w].mean()
    n_slots = len(log["utility"][w])
    n_crit = int(ev.crit.sum())
    feas = ev.crit_feasible[w]
    n_feas = int(feas.sum())
    pay_feas = int(ev.payload_feasible[w].sum())

    # Eq. 39j is per critical user: Pr[Q_k < Q_min] <= eps_k. Evaluated over each user's feasible slots.
    fk = feas[:, ev.crit].sum(axis=0)
    vk = log["viol_feasible_k"][w][:, ev.crit].sum(axis=0)
    rates_k = vk[fk > 0] / fk[fk > 0]

    switch_rate = log["switch"][w].mean()
    c = cvar(log["loss"][w], alpha)
    return {
        "utility": utility,
        "utility_per_user": utility / K,
        "regret": log["regret"][w].mean(),
        "switch_rate": switch_rate,
        "bits": bits,
        "utility_per_bit": utility / bits if bits > 0 else np.nan,
        "loss": log["loss"][w].mean(),
        "cvar": c,
        "violation_rate": log["violations"][w].sum() / (n_crit * n_slots) if n_crit else np.nan,
        "fb_share": log["fb_users"][w].mean() / K,
        # --- Phase 1: feasibility oracle and the missing constraints of P1 ---
        "crit_infeasible_share": 1.0 - n_feas / (n_crit * n_slots) if n_crit else np.nan,
        "violation_rate_feasible": log["violations_feasible"][w].sum() / n_feas if n_feas else np.nan,
        "crit_worst_violation_feasible": rates_k.max() if len(rates_k) else np.nan,
        "reliability_met": float(np.all(rates_k <= ev.eps)) if len(rates_k) else np.nan,
        "payload_miss_rate": log["payload_miss"][w].sum() / (K * n_slots),
        "payload_infeasible_share": 1.0 - pay_feas / (K * n_slots),
        "payload_miss_rate_feasible": log["payload_miss_feasible"][w].sum() / pay_feas if pay_feas else np.nan,
        "objective_p1": objective_p1(utility, bits, K * switch_rate, c, *ev.lambdas),
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
