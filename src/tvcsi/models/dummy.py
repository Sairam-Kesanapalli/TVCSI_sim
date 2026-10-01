"""Placeholder models used until trained networks exist. Results produced with them are not
research results; every figure is stamped "DUMMY MODELS".
"""
from __future__ import annotations

import numpy as np

from ..codebook import beam_gains
from .base import BeamPredictor, Decoder, Encoder

_LOG_FLOOR = 1e-12  # keeps log() finite for an (exactly) zero beam gain


class DummyEncoder(Encoder):
    """Returns noisy log beam gains: z^m = log g(hhat) + sigma_m * Z, per beam, Z ~ N(0, 1).

    g(hhat) = |hhat^H w_b|^2 of the latest channel estimate. Z is drawn once per channel estimate
    and shared by all modes, so a higher mode refines the same description instead of drawing an
    independent one (progressive refinement, Eq. 10). The code has B entries rather than d_m;
    the feedback size is accounted for only through C_m.
    """

    def __init__(self, W: np.ndarray, sigma_m, rng: np.random.Generator):
        self.W = W
        self.sigma = np.asarray(sigma_m, dtype=float)
        self.rng = rng
        self._last_input: np.ndarray | None = None
        self._z: np.ndarray | None = None

    def encode(self, hhat_hist: np.ndarray, m: int) -> np.ndarray:
        latest = hhat_hist[:, -1]
        if self._last_input is None or not np.array_equal(latest, self._last_input):
            self._last_input = latest.copy()
            self._z = self.rng.standard_normal((latest.shape[0], self.W.shape[1]))
        g = beam_gains(latest, self.W)
        return np.log(g + _LOG_FLOOR) + self.sigma[m - 1] * self._z


class DummyDecoder(Decoder):
    """Inverse of DummyEncoder's log map: ghat^m = exp(z^m) = g(hhat) * exp(sigma_m N(0,1))."""

    def decode(self, code: np.ndarray, m: int) -> np.ndarray:
        return np.exp(code)


class PersistencePredictor(BeamPredictor):
    """Persistence predictor: ghat0 = last pushed gains, confidence = exp(-age / tau).

    age = slots since the user's last mode > 0 feedback (1 if it fed back in the previous slot).
    Before any feedback the predictor holds the uninformative prior g = M (the full array gain,
    i.e. "nominal SNR is attainable") with confidence 0.
    """

    def __init__(self, K: int, B: int, M: int, tau: float):
        self.last = np.full((K, B), float(M))
        self.age = np.full(K, np.inf)
        self.tau = tau

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        return self.last.copy(), np.exp(-self.age / self.tau)

    def update(self, ghat: np.ndarray, fresh: np.ndarray) -> None:
        self.last = np.array(ghat, dtype=float, copy=True)
        self.age = np.where(fresh, 1.0, self.age + 1.0)
