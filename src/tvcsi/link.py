"""Link-level mappings: beam regret, SINR, task quality and utility (spec 3.4-3.6)."""
from __future__ import annotations

import numpy as np

from .config import lin2db


def select_beam(ghat: np.ndarray) -> np.ndarray:
    """Beam selection bhat_k = argmax_b ghat_kb (Eq. 14). ghat [..., B] -> [...] int."""
    return np.argmax(ghat, axis=-1)


def gain_at(g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """g[k, b_k] for every leading index k."""
    if g.ndim == 2:   # the common [K, B] case; plain fancy indexing is much cheaper
        return g[np.arange(g.shape[0]), b]
    return np.take_along_axis(g, b[..., None], axis=-1)[..., 0]


def beam_regret(g_true: np.ndarray, bhat: np.ndarray, eps: float) -> np.ndarray:
    """Normalised beam-semantic regret d_k = 1 - g_{k,bhat} / (max_b g_kb + eps) (Eq. 15)."""
    return 1.0 - gain_at(g_true, bhat) / (g_true.max(axis=-1) + eps)


def sinr(g_true: np.ndarray, bhat: np.ndarray, snr_lin: float, M: int) -> np.ndarray:
    """Interference-free SINR gamma_k = snr * g_{k,bhat} / M (spec 3.5; Eq. 26 with one user per RB)."""
    return snr_lin * gain_at(g_true, bhat) / M


def task_quality(gamma_lin: np.ndarray, gamma0_db: np.ndarray, s_db: float) -> np.ndarray:
    """Logistic task quality Q(gamma) = 1 / (1 + exp(-(gamma_dB - gamma0_dB) / s)) (Eq. 20, spec 3.6)."""
    z = (lin2db(gamma_lin) - gamma0_db) / s_db
    return 0.5 * (1.0 + np.tanh(0.5 * z))   # numerically stable form of the logistic


def task_utility(v: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Task utility U_k = v_k * Q_k(gamma_k) (Eq. 21, single semantic representation)."""
    return v * Q
