"""DFT beam codebook and beam-utility vectors (spec 3.2-3.3)."""
from __future__ import annotations

import numpy as np


def dft_codebook(M: int, B: int) -> np.ndarray:
    """DFT codebook w_b[n] = exp(j pi n (2b/B - 1)) / sqrt(M); returns W of shape [M, B].

    Beam b points at sin(theta) = 2b/B - 1, matching the sign convention of array_response.
    """
    n = np.arange(M)[:, None]
    u = 2 * np.arange(B)[None, :] / B - 1
    return np.exp(1j * np.pi * n * u) / np.sqrt(M)


def beam_gains(h: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Beam-utility vector g_kb = |h_k^H w_b|^2 (Eqs. 3-4). h [..., M] -> g [..., B]."""
    return np.abs(h.conj() @ W) ** 2
