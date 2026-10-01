"""Synthetic time-correlated multipath ULA channel (spec 3.1).

Every channel source exposes the same output: a complex array h of shape [users, time, M].
"""
from __future__ import annotations

import numpy as np

from .config import ar1_rho


def array_response(theta: np.ndarray, M: int) -> np.ndarray:
    """ULA steering vector a(theta)[n] = exp(j pi n sin(theta)) / sqrt(M), n = 0..M-1.

    theta of any shape -> output of shape theta.shape + (M,).
    """
    n = np.arange(M)
    return np.exp(1j * np.pi * np.sin(theta)[..., None] * n) / np.sqrt(M)


def generate_channels(cfg: dict, rng: np.random.Generator, K: int | None = None,
                      T: int | None = None) -> np.ndarray:
    """Draw h_k(t) = sqrt(M) * sum_p alpha_kp(t) a(theta_kp(t)) / sqrt(P) for all users and slots.

    Path gains follow the Jakes AR(1) model alpha(t+1) = rho alpha(t) + sqrt(1-rho^2) CN(0,1) and
    angles a Gaussian random walk with std c_theta * speed per slot. Returns complex128 [K, T, M]
    with E||h||^2 = M.
    """
    K = cfg["system"]["K"] if K is None else K
    T = cfg["sim"]["T"] if T is None else T
    M, ch = cfg["system"]["M"], cfg["channel"]
    if ch["source"] == "deepmimo":
        return load_deepmimo(cfg)
    P = ch["P"]
    rho = ar1_rho(cfg)
    sigma_theta = ch["c_theta"] * ch["speed_kmh"]

    alpha = np.empty((K, T, P), dtype=complex)
    theta = np.empty((K, T, P))
    alpha[:, 0] = _cn(rng, (K, P))
    theta[:, 0] = rng.uniform(-np.pi / 2, np.pi / 2, (K, P))
    innov = np.sqrt(1 - rho**2) * _cn(rng, (K, T - 1, P))
    dtheta = sigma_theta * rng.standard_normal((K, T - 1, P))
    for t in range(1, T):
        alpha[:, t] = rho * alpha[:, t - 1] + innov[:, t - 1]
    theta[:, 1:] = theta[:, :1] + np.cumsum(dtheta, axis=1)
    # Reflect angles back into [-pi/2, pi/2]. arcsin(sin(.)) leaves sin(theta), and hence the
    # channel, unchanged; it only keeps the stored angle in the physical field of view.
    theta = np.arcsin(np.sin(theta))

    a = array_response(theta, M)                                  # [K, T, P, M]
    return np.sqrt(M / P) * np.einsum("ktp,ktpm->ktm", alpha, a)


def load_deepmimo(cfg: dict) -> np.ndarray:
    """Placeholder for a DeepMIMO ray-tracing loader.

    Expected output (identical to generate_channels): complex array of shape [users, time, M],
    one row per user, consecutive slots of duration Ts along axis 1, M ULA antenna coefficients
    along axis 2, normalised so that E||h||^2 = M.
    """
    raise NotImplementedError("DeepMIMO loading is not implemented; use channel.source: synthetic")


def pilot_estimate(h: np.ndarray, snr_pilot_lin: float, rng: np.random.Generator) -> np.ndarray:
    """Pilot-based estimate hhat = h + CN(0, sigma_e^2 I), sigma_e^2 = M / snr_pilot (spec step 4.1)."""
    M = h.shape[-1]
    return h + np.sqrt(M / snr_pilot_lin) * _cn(rng, h.shape)


def _cn(rng: np.random.Generator, shape) -> np.ndarray:
    """Circularly-symmetric CN(0, 1) samples."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
