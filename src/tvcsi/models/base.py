"""Model interfaces. All methods are batched over users: arrays carry a leading K axis.

Trained models (a learned semantic encoder/decoder and a GRU beam predictor) plug in by
subclassing these and being named in the config (models: trained).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Encoder(ABC):
    """Semantic CSI encoder E_theta^m (Eq. 9)."""

    @abstractmethod
    def encode(self, hhat_hist: np.ndarray, m: int) -> np.ndarray:
        """hhat_hist: complex [K, L_h, M] channel estimates, most recent last; m in 1..M_f.

        Returns the feedback code z^m, an array with leading axis K.
        """


class Decoder(ABC):
    """Semantic CSI decoder D_omega (Eq. 12)."""

    @abstractmethod
    def decode(self, code: np.ndarray, m: int) -> np.ndarray:
        """Returns the estimated beam-utility vectors ghat^m of shape [K, B]."""


class BeamPredictor(ABC):
    """gNB-side beam predictor P_psi (Eq. 13), one instance per scheme.

    The predictor owns its history buffer (a recurrent model needs its own state anyway), so
    predict() reads that buffer instead of taking it as an argument.
    """

    @abstractmethod
    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (ghat0 [K, B], confidence [K] in [0, 1]) from the history pushed so far."""

    @abstractmethod
    def update(self, ghat: np.ndarray, fresh: np.ndarray) -> None:
        """Push the gains the gNB actually holds after the slot.

        ghat: [K, B] decoded gains for users that fed back, predicted gains otherwise.
        fresh: [K] bool, True where the user sent feedback (mode > 0) this slot.
        """
