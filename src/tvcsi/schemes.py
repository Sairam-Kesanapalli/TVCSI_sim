"""Feedback-mode scheduling schemes (spec sections 2 and 7).

A scheme sees only DecisionInputs, which hold gNB-side information: estimated utilities, the
gNB's SINR belief, importance, costs and budget, and (since S3) its predicted beam gains, the
predictor's confidence, each user's feedback age and last mode, and the message types. None of
them is derived from the true channel. Only 'Exact' additionally receives the true
per-mode utilities (its genie role); the simulator passes None to every other scheme.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .importance import CRITICAL
from .solvers import dp_mckp, greedy_tvcsi


@dataclass
class DecisionInputs:
    t: int
    U_hat: np.ndarray | None   # [K, M_f + 1] estimated utilities (None if the scheme needs none)
    gamma_ref: np.ndarray      # [K] predictor-based SINR belief
    v: np.ndarray              # [K] semantic importance
    costs: np.ndarray          # [M_f + 1] bits per mode
    budget: float              # C_FB_max in bits
    # S3: further gNB-side state. All optional, so existing constructor calls keep working. The
    # simulator passes copies, so a scheme cannot change the simulator's own state.
    ghat0: np.ndarray | None = None      # [K, B] float: this scheme's predicted beam gains
    conf: np.ndarray | None = None       # [K] float in [0, 1]: the predictor's confidence
    age: np.ndarray | None = None        # [K] float: slots since this scheme last let the user feed
                                         #   back (mode > 0); 1 = previous slot, inf = never
    last_mode: np.ndarray | None = None  # [K] int64: mode this scheme chose in the previous slot (0 at t = 0)
    types: np.ndarray | None = None      # [K] int: message type (importance.CRITICAL, NORMAL, BACKGROUND)
    # S4: outage probability Pr[Q_k^m < Q_min] per mode from the learned estimator (None otherwise)
    p_out: np.ndarray | None = None      # [K, M_f + 1] float in [0, 1]


@dataclass(frozen=True)
class Scheme:
    name: str
    decide: Callable[[DecisionInputs, np.ndarray | None], np.ndarray]
    # None | "learned" | "learned_no_importance" (S4 learned estimator)
    #      | "full" | "no_confidence" | "no_importance" (mean-regret plug-in table, the *_plugin ablations)
    estimator: str | None = None
    uses_truth: bool = False       # receives true utilities (Exact only)
    budget_exempt: bool = False    # may exceed C_FB_max (full_feedback and fitting policies)


def periodic_modes(t: int, K: int, m_fixed: int, costs: np.ndarray, budget: float) -> np.ndarray:
    """Periodic round-robin baseline (spec 7.4).

    Period P = ceil(K C_mfixed / C_FB_max); user k feeds back with mode m_fixed in slot t iff
    (t + k) mod P == 0. If that exceeds the budget, users are dropped in ascending index order.
    A budget below C_mfixed admits no feedback at all.
    """
    modes = np.zeros(K, dtype=np.int64)
    c = costs[m_fixed]
    if budget < c:
        return modes
    period = math.ceil(K * c / budget - 1e-12)
    modes[(t + np.arange(K)) % period == 0] = m_fixed
    for k in np.flatnonzero(modes):
        if costs[modes].sum() <= budget + 1e-9:
            break
        modes[k] = 0
    return modes


def fill_in_order(order: np.ndarray, m: int, costs: np.ndarray, budget: float, K: int) -> np.ndarray:
    """Give mode m to users in the given order until the next one no longer fits."""
    modes = np.zeros(K, dtype=np.int64)
    n = int(np.floor(budget / costs[m] + 1e-9))
    modes[order[:n]] = m
    return modes


def fill_largest(order: np.ndarray, costs: np.ndarray, budget: float, K: int) -> np.ndarray:
    """S5, faculty baseline rule (TaskVoCSI_v2 policy.baseline_modes): walk the users in `order` and
    give each the LARGEST mode that still fits the remaining budget; stop at the first user for whom
    no mode > 0 fits. So the budget is filled instead of leaving bits unused."""
    modes = np.zeros(K, dtype=np.int64)
    left = float(budget)
    for k in order:
        m = int(np.flatnonzero(costs <= left + 1e-9).max())
        if m == 0:
            break
        modes[k] = m
        left -= float(costs[m])
    return modes


def belief_margin(ghat0: np.ndarray) -> np.ndarray:
    """(top1 - top2) / top1 of the predicted beam gains: small = the gNB is unsure which beam is best."""
    g = np.sort(ghat0, axis=1)
    return (g[:, -1] - g[:, -2]) / (g[:, -1] + 1e-6)


# S5: the five faculty baselines (name -> user order). All use only gNB-side information.
FILL_ORDERS = {
    "fill_periodic": lambda inp: np.roll(np.arange(len(inp.v)), -(inp.t % len(inp.v))),   # round robin
    "fill_age": lambda inp: np.argsort(-inp.age, kind="stable"),              # oldest report first
    "fill_uncertainty": lambda inp: np.argsort(belief_margin(inp.ghat0), kind="stable"),
    "fill_importance": lambda inp: np.argsort(-inp.v, kind="stable"),         # most important first
    "fill_channel": lambda inp: np.argsort(inp.ghat0.max(axis=1), kind="stable"),  # weakest believed link first
}


def risk_scheme(name: str, beta: float, eps: float) -> Scheme:
    """S5 'Proposed_risk': Proposed plus a virtual queue (Lyapunov drift-plus-penalty) per critical user.

    Z_k(t+1) = max(Z_k(t) + p_k(t) - max(eps, p_k^best(t)), 0), where p_k(t) is the estimated
    outage probability Pr[Q_k < Q_min] of the mode chosen for critical user k and p_k^best(t) the
    smallest one over all modes (S4 estimator, gNB-side). Z_k grows only while a critical user's
    expected violations run above the target eps AND above what any mode could achieve, i.e. only
    for avoidable violations (cf. the Phase 1 feasibility split). Without the max(eps, p^best) term
    the queue of a user whom no mode can protect grows forever and takes the whole budget (first
    calibration, research/logs/S5/01_calibrate_risk_v1.log). Z_k then raises the value of feedback
    for that user:  value[k][m] = v_k E[Q_k^m] - beta Z_k Pr[Q_k^m < Q_min].
    beta = 0 gives exactly Proposed. The queue lives in the scheme (one per simulate call)."""
    state = {"Z": None}

    def decide(inp, _):
        K = len(inp.v)
        if state["Z"] is None:
            state["Z"] = np.zeros(K)
        Z = state["Z"] * (inp.types == CRITICAL)
        values = inp.U_hat - beta * Z[:, None] * inp.p_out if beta else inp.U_hat
        modes = greedy_tvcsi(values, inp.costs, inp.budget)
        chosen_p = inp.p_out[np.arange(K), modes]
        target = np.maximum(eps, inp.p_out.min(axis=1))
        state["Z"] = np.maximum(Z + (inp.types == CRITICAL) * (chosen_p - target), 0.0)
        return modes
    return Scheme(name, decide, "learned")


def build_scheme(name: str, cfg: dict, rng: np.random.Generator | None = None) -> Scheme:
    """Construct a scheme by name. 'Baseline' resolves to cfg.baseline.kind."""
    m_fixed = cfg["baseline"]["m_fixed"]

    def periodic(inp, _):
        return periodic_modes(inp.t, len(inp.v), m_fixed, inp.costs, inp.budget)

    def snr_threshold(inp, _):
        # weakest believed links first: they gain most from fresh CSI under a channel-only view
        return fill_in_order(np.argsort(inp.gamma_ref, kind="stable"), m_fixed, inp.costs,
                             inp.budget, len(inp.v))

    if name == "Baseline":
        kind = cfg["baseline"]["kind"]
        base = {"periodic": periodic, "snr_threshold": snr_threshold}[kind]
        return Scheme(name, base)
    if name == "Exact":
        return Scheme(name, lambda inp, U_true: dp_mckp(U_true, inp.costs, inp.budget), uses_truth=True)
    dp = lambda inp, _: dp_mckp(inp.U_hat, inp.costs, inp.budget)            # noqa: E731
    greedy = lambda inp, _: greedy_tvcsi(inp.U_hat, inp.costs, inp.budget)   # noqa: E731
    # S4: the main schemes use the learned per-mode quality estimator (tvcsi.estimator).
    if name == "Optimised":
        return Scheme(name, dp, "learned")
    if name == "Proposed":
        return Scheme(name, greedy, "learned")
    if name == "Proposed_no_importance":
        return Scheme(name, greedy, "learned_no_importance")
    # Pre-S4 mean-regret plug-in, kept unchanged as ablations (bit-identical to the old
    # Optimised / Proposed / Proposed_no_confidence / Proposed_no_importance).
    if name == "Optimised_plugin":
        return Scheme(name, dp, "full")
    if name == "Proposed_plugin":
        return Scheme(name, greedy, "full")
    if name == "Proposed_plugin_no_confidence":
        return Scheme(name, greedy, "no_confidence")
    if name == "Proposed_plugin_no_importance":
        return Scheme(name, greedy, "no_importance")
    if name == "Proposed_risk":
        return risk_scheme(name, float(cfg["risk"]["beta"]), float(cfg["risk"]["eps_target"]))
    if name in FILL_ORDERS:
        order = FILL_ORDERS[name]
        return Scheme(name, lambda inp, _: fill_largest(order(inp), inp.costs, inp.budget, len(inp.v)))
    if name == "periodic":
        return Scheme(name, periodic)
    if name == "snr_threshold":
        return Scheme(name, snr_threshold)
    if name == "importance_only":
        m_top = len(cfg["feedback"]["code_len"])
        return Scheme(name, lambda inp, _: fill_in_order(np.argsort(-inp.v, kind="stable"), m_top,
                                                         inp.costs, inp.budget, len(inp.v)))
    if name == "prediction_only":
        return Scheme(name, lambda inp, _: np.zeros(len(inp.v), dtype=np.int64))
    if name == "full_feedback":
        m_top = len(cfg["feedback"]["code_len"])
        return Scheme(name, lambda inp, _: np.full(len(inp.v), m_top, dtype=np.int64),
                      budget_exempt=True)
    if name.startswith("random_p0="):
        # table-fitting policy: mode 0 with probability p0, otherwise uniform over 1..M_f
        p0, n_fb = float(name.split("=")[1]), len(cfg["feedback"]["code_len"])

        def random_policy(inp, _):
            K = len(inp.v)
            return np.where(rng.random(K) < p0, 0, rng.integers(1, n_fb + 1, K))
        return Scheme(name, random_policy, budget_exempt=True)
    raise ValueError(f"unknown scheme {name!r}")
