"""Feedback-mode scheduling schemes (spec sections 2 and 7).

A scheme sees only DecisionInputs, which hold gNB-side information: estimated utilities, the
gNB's SINR belief, importance, costs and budget. Only 'Exact' additionally receives the true
per-mode utilities (its genie role); the simulator passes None to every other scheme.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .solvers import dp_mckp, greedy_tvcsi


@dataclass
class DecisionInputs:
    t: int
    U_hat: np.ndarray | None   # [K, M_f + 1] estimated utilities (None if the scheme needs none)
    gamma_ref: np.ndarray      # [K] predictor-based SINR belief
    v: np.ndarray              # [K] semantic importance
    costs: np.ndarray          # [M_f + 1] bits per mode
    budget: float              # C_FB_max in bits


@dataclass(frozen=True)
class Scheme:
    name: str
    decide: Callable[[DecisionInputs, np.ndarray | None], np.ndarray]
    estimator: str | None = None   # None | "full" | "no_confidence" | "no_importance"
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
    if name == "Optimised":
        return Scheme(name, lambda inp, _: dp_mckp(inp.U_hat, inp.costs, inp.budget), "full")
    if name == "Proposed":
        return Scheme(name, lambda inp, _: greedy_tvcsi(inp.U_hat, inp.costs, inp.budget), "full")
    if name == "Proposed_no_confidence":
        return Scheme(name, lambda inp, _: greedy_tvcsi(inp.U_hat, inp.costs, inp.budget), "no_confidence")
    if name == "Proposed_no_importance":
        return Scheme(name, lambda inp, _: greedy_tvcsi(inp.U_hat, inp.costs, inp.budget), "no_importance")
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
