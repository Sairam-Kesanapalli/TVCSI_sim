"""Per-slot feedback-mode allocation: a multiple-choice knapsack over users (Eqs. 7, 11).

Every solver takes values [K, M_f + 1] (utility of each user/mode), integer costs [M_f + 1]
with costs[0] = 0, and a budget in bits, and returns the chosen mode per user (int [K]).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

_TOL = 1e-9  # absorbs float round-off in budget comparisons


@lru_cache(maxsize=8)
def _all_mode_vectors(K: int, n_modes: int) -> np.ndarray:
    """Every mode vector, shape [n_modes**K, K]. Cached: enumeration is fixed for (K, n_modes)."""
    return np.indices((n_modes,) * K, dtype=np.int8).reshape(K, -1).T


def brute_force(values: np.ndarray, costs: np.ndarray, budget: float) -> np.ndarray:
    """Exhaustive search over all (M_f+1)^K mode vectors; feasible only for small K (<= 10)."""
    K, n_modes = values.shape
    X = _all_mode_vectors(K, n_modes)
    total = values[np.arange(K), X].sum(axis=1)
    total[costs[X].sum(axis=1) > budget + _TOL] = -np.inf
    return X[np.argmax(total)].astype(np.int64)


def dp_mckp(values: np.ndarray, costs: np.ndarray, budget: float) -> np.ndarray:
    """Exact multiple-choice knapsack by dynamic programming over (user, budget units).

    The budget is discretised in units of gcd(C_m) (32 bits for the default modes), so the DP is
    exact. best[c] = max total value of the users seen so far using at most c units. Ties go to
    the cheaper mode.
    """
    K, n_modes = values.shape
    unit = int(np.gcd.reduce(costs[costs > 0]))
    w = (costs // unit).astype(np.int64)
    cap = int(np.floor(budget / unit + _TOL))
    best = np.zeros(cap + 1)
    choice = np.zeros((K, cap + 1), dtype=np.int64)
    cand = np.empty((n_modes, cap + 1))
    for k in range(K):
        cand.fill(-np.inf)
        for m in range(n_modes):
            if w[m] <= cap:
                cand[m, w[m]:] = best[: cap + 1 - w[m]] + values[k, m]
        choice[k] = np.argmax(cand, axis=0)
        best = cand[choice[k], np.arange(cap + 1)]
    modes = np.zeros(K, dtype=np.int64)
    c = cap
    for k in range(K - 1, -1, -1):
        modes[k] = choice[k, c]
        c -= w[modes[k]]
    return modes


def upper_hull(costs: np.ndarray, vals: np.ndarray) -> list[int]:
    """Modes on the upper concave hull of the points (C_m, U_m), starting at mode 0 and keeping
    only segments of strictly positive slope. LP-dominated modes (below a chord) are dropped."""
    hull: list[int] = []
    for m in range(len(costs)):
        while len(hull) >= 2:
            o, a = hull[-2], hull[-1]
            # drop a if it lies on or below the chord o -> m
            if (vals[a] - vals[o]) * (costs[m] - costs[o]) <= (vals[m] - vals[o]) * (costs[a] - costs[o]):
                hull.pop()
            else:
                break
        hull.append(m)
    for i in range(1, len(hull)):
        if vals[hull[i]] <= vals[hull[i - 1]]:
            return hull[:i]
    return hull


def greedy_tvcsi(values: np.ndarray, costs: np.ndarray, budget: float) -> np.ndarray:
    """Greedy marginal-TVCSI upgrades (spec 7.3, the 'Proposed' rule).

    All users start in mode 0. Each user's admissible upgrades follow its upper hull, whose
    successive slopes (Uhat_{m'} - Uhat_m) / (C_{m'} - C_m) are decreasing marginal TVCSI values.
    The upgrade with the largest positive marginal TVCSI that still fits in the remaining budget
    is applied until none is left.
    """
    K = values.shape[0]
    hulls = [upper_hull(costs, values[k]) for k in range(K)]
    pos = np.zeros(K, dtype=np.int64)
    slope = np.full(K, -np.inf)
    step_cost = np.zeros(K)

    def refresh(k: int) -> None:
        h, i = hulls[k], pos[k]
        if i + 1 < len(h):
            step_cost[k] = costs[h[i + 1]] - costs[h[i]]
            slope[k] = (values[k, h[i + 1]] - values[k, h[i]]) / step_cost[k]
        else:
            slope[k] = -np.inf

    for k in range(K):
        refresh(k)
    remaining = float(budget)
    while True:
        cand = np.where(step_cost <= remaining + _TOL, slope, -np.inf)
        k = int(np.argmax(cand))
        if not cand[k] > 0:
            break
        remaining -= step_cost[k]
        pos[k] += 1
        refresh(k)
    return np.array([hulls[k][pos[k]] for k in range(K)], dtype=np.int64)
