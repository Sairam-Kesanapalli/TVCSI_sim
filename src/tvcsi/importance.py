"""Semantic importance v_k(t) and message types (spec 3.7)."""
from __future__ import annotations

import numpy as np

CRITICAL, NORMAL, BACKGROUND = 0, 1, 2
TYPE_NAMES = ("critical", "normal", "background")


def n_critical(K: int, crit_share: float) -> int:
    """Number of critical users |K_c| = round-half-up(crit_share * K)."""
    return int(np.floor(crit_share * K + 0.5))


def assign_types(K: int, crit_share: float) -> np.ndarray:
    """First |K_c| users are critical; the rest alternate normal / background."""
    nc = n_critical(K, crit_share)
    types = np.empty(K, dtype=np.int64)
    types[:nc] = CRITICAL
    types[nc:] = np.where(np.arange(K - nc) % 2 == 0, NORMAL, BACKGROUND)
    return types


def gamma0_per_user(cfg: dict, types: np.ndarray) -> np.ndarray:
    """Quality threshold gamma0 [dB] of each user's message type."""
    g0 = cfg["quality"]["gamma0_db"]
    return np.array([g0[TYPE_NAMES[t]] for t in types], dtype=float)


def generate_importance(cfg: dict, types: np.ndarray, rng: np.random.Generator,
                        T: int | None = None) -> np.ndarray:
    """Two-state (low/high) Markov importance, v redrawn uniformly within its state every slot.

    The initial state is drawn from the chain's stationary distribution so there is no
    start-up transient. Returns v of shape [K, T] in [0, 1].
    """
    imp = cfg["importance"]
    T = cfg["sim"]["T"] if T is None else T
    K = len(types)
    crit = types == CRITICAL
    p_lh = np.where(crit, imp["p_low_to_high_crit"], imp["p_low_to_high"])
    p_hl = np.where(crit, imp["p_high_to_low_crit"], imp["p_high_to_low"])

    high = np.empty((K, T), dtype=bool)
    high[:, 0] = rng.random(K) < p_lh / (p_lh + p_hl)
    u = rng.random((K, T - 1))
    for t in range(1, T):
        prev = high[:, t - 1]
        high[:, t] = np.where(prev, u[:, t - 1] >= p_hl, u[:, t - 1] < p_lh)

    lo, hi = imp["low_range"], imp["high_range"]
    v_low = rng.uniform(lo[0], lo[1], (K, T))
    v_high = rng.uniform(hi[0], hi[1], (K, T))
    return np.where(high, v_high, v_low)
