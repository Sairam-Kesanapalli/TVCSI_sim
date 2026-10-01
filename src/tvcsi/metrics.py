"""Evaluation metrics (spec 3.9) and confidence intervals."""
from __future__ import annotations

import numpy as np
from scipy import stats


def semantic_loss(v: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Importance-weighted semantic loss L_sem(t) = sum_k v_k(t) (1 - Q_k(t)) (Eq. 36).

    v, Q: [..., K] -> [...]."""
    return np.sum(v * (1.0 - Q), axis=-1)


def cvar(L: np.ndarray, alpha: float) -> np.ndarray:
    """CVaR_alpha of the empirical distribution of L, sorted-tail form of Eq. 37.

    With n samples the tail mass is n (1 - alpha) samples: the mean of the largest
    floor(n(1-alpha)) values plus the fractional share of the next one. This equals
    min_eta { eta + E[(L - eta)^+] / (1 - alpha) } exactly (see cvar_min_eta).
    """
    x = np.sort(np.asarray(L, dtype=float))[::-1]
    tail = len(x) * (1.0 - alpha)
    full = int(np.floor(tail + 1e-12))
    frac = tail - full
    total = x[:full].sum() + (frac * x[full] if full < len(x) else 0.0)
    return total / tail


def cvar_min_eta(L: np.ndarray, alpha: float) -> float:
    """CVaR_alpha by direct minimisation over eta of Eq. 37.

    The objective is convex and piecewise linear in eta with breakpoints at the samples, so its
    minimum is attained at one of them; evaluating every sample point is exact.
    """
    x = np.asarray(L, dtype=float)
    eta = np.unique(x)
    obj = eta + np.maximum(x[None, :] - eta[:, None], 0).mean(axis=1) / (1.0 - alpha)
    return float(obj.min())


def switch_indicator(b: np.ndarray, b_prev: np.ndarray) -> np.ndarray:
    """Beam-switch indicator s_k(t) = 1{b_k(t) != b_k(t-1)} (Eq. 38)."""
    return (b != b_prev).astype(float)


def t_interval(x, confidence: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of the Student-t confidence interval over seeds (NaNs ignored)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(x[0]), np.nan
    half = stats.t.ppf(0.5 + confidence / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(x.mean()), float(half)
