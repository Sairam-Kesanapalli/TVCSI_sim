"""Stage S6: trained models, numpy inference (no torch needed to simulate).

Adapted from the faculty TaskVoCSI_v2 `taskv2/nn.py` (ProgressiveCodec, BeamGRU) to our system:

* TrainedEncoder / TrainedDecoder: a progressive codec. The encoder maps the user's latest pilot
  estimate hhat (real and imaginary parts, plus its pilot SNR) to 32 latent values in [-1, 1],
  quantised to 4 bits each (16 levels). Mode m sends the first d_m = 8 / 16 / 32 values, so the
  report costs exactly C_m = 32 / 64 / 128 bits (Eqs. 8-10, progressive refinement). The decoder
  zero-fills the values not sent and outputs log(1 + g) for all B beams.
* GRUPredictor: the gNB-side beam predictor (Eq. 13). Its input per slot is the last DECODED beam
  gains the gNB holds for the user, the feedback age and a fresh-feedback flag (as the faculty
  `History`), plus the configured speed and SNR. A GRU runs over the last `window` slots and a head
  outputs log(1 + g) for every beam and a confidence = Pr[the predicted beam is within 10 % of the
  best beam's gain] (a learned confidence, which the faculty GRU did not have).

Weights are trained by `scripts/train_models.py` (torch) and stored as plain arrays in an .npz file
(`trained.kwargs.weights`), so the simulator runs them with numpy. Inference mirrors the torch
modules exactly (tests/test_s6_models.py checks this when torch is installed).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..config import resolve_path
from .base import BeamPredictor, Decoder, Encoder


@lru_cache(maxsize=4)
def load_weights(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k].astype(np.float64) for k in z.files}


def _weights(cfg: dict, weights: str) -> dict[str, np.ndarray]:
    return load_weights(str(resolve_path(cfg, weights)))


def _relu(x):
    return np.maximum(x, 0.0)


def _softplus(x):
    return np.logaddexp(0.0, x)


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _lin(w, name, x):
    return x @ w[f"{name}.weight"].T + w[f"{name}.bias"]


def quantize(z: np.ndarray, bits: int) -> np.ndarray:
    """Uniform scalar quantiser of values in [-1, 1] to 2**bits levels (the codec's STE forward)."""
    L = 2 ** bits - 1
    return np.round((z + 1.0) * (L / 2.0)) / (L / 2.0) - 1.0


def codec_input(hhat: np.ndarray, snr_db: float) -> np.ndarray:
    """[..., M] complex pilot estimate -> [..., 2M + 1] real input (Re, Im, pilot SNR / 30)."""
    return np.concatenate((hhat.real, hhat.imag, np.full(hhat.shape[:-1] + (1,), snr_db / 30.0)), axis=-1)


class TrainedEncoder(Encoder):
    def __init__(self, cfg: dict, W=None, rng=None, weights: str = "", **_):
        self.w = _weights(cfg, weights)
        self.dims = [int(d) for d in cfg["feedback"]["code_len"]]
        self.bits = int(cfg["feedback"]["bits_per_value"])
        self.snr_db = float(cfg["link"]["snr_db"]) + float(cfg["link"]["pilot_snr_offset_db"])
        self._cache_key = None
        self._cache_z = None

    def latent(self, hhat: np.ndarray) -> np.ndarray:
        """[K, M] -> quantised latent [K, d_max] (shared by all modes: progressive refinement)."""
        x = codec_input(hhat, self.snr_db)
        z = np.tanh(_lin(self.w, "enc.2", _relu(_lin(self.w, "enc.0", x))))
        return quantize(z, self.bits)

    def encode(self, hhat_hist: np.ndarray, m: int) -> np.ndarray:
        latest = hhat_hist[:, -1]
        if self._cache_key is None or not np.array_equal(latest, self._cache_key):   # once per slot
            self._cache_key, self._cache_z = latest.copy(), self.latent(latest)
        return self._cache_z[:, : self.dims[m - 1]].copy()


class TrainedDecoder(Decoder):
    def __init__(self, cfg: dict, weights: str = "", **_):
        self.w = _weights(cfg, weights)
        self.d_max = int(cfg["feedback"]["code_len"][-1])

    def decode_log(self, code: np.ndarray) -> np.ndarray:
        z = np.zeros(code.shape[:-1] + (self.d_max,))
        z[..., : code.shape[-1]] = code
        return _softplus(_lin(self.w, "dec.2", _relu(_lin(self.w, "dec.0", z))))

    def decode(self, code: np.ndarray, m: int) -> np.ndarray:
        return np.clip(np.expm1(self.decode_log(code)), 0.0, None)


# ----------------------------------------------------------------------------- GRU predictor

AGE_CAP, AGE_SCALE = 100.0, 25.0


def history_row(last: np.ndarray, age: np.ndarray, fresh: np.ndarray, speed_kmh: float, snr_db: float) -> np.ndarray:
    """One slot of predictor input [K, B + 4]: log(1 + last decoded gains), capped age, fresh flag,
    speed / 60, SNR / 30."""
    K = last.shape[0]
    a = np.minimum(np.where(np.isfinite(age), age, AGE_CAP), AGE_CAP) / AGE_SCALE
    return np.column_stack((np.log1p(last), a, fresh.astype(float), np.full(K, speed_kmh / 60.0),
                            np.full(K, snr_db / 30.0)))


def gru_forward(w: dict, X: np.ndarray) -> np.ndarray:
    """PyTorch nn.GRU (one layer, batch_first, h0 = 0) over X [N, L, F]; returns the last hidden state."""
    Wi, Wh, bi, bh = w["gru.weight_ih_l0"], w["gru.weight_hh_l0"], w["gru.bias_ih_l0"], w["gru.bias_hh_l0"]
    H = Wh.shape[1]
    h = np.zeros((X.shape[0], H))
    gi_all = X @ Wi.T + bi                                      # [N, L, 3H]
    for t in range(X.shape[1]):
        gi, gh = gi_all[:, t], h @ Wh.T + bh
        r = _sigmoid(gi[:, :H] + gh[:, :H])
        z = _sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
        n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        h = (1.0 - z) * n + z * h
    return h


def gru_head(w: dict, h: np.ndarray, B: int) -> tuple[np.ndarray, np.ndarray]:
    """-> (log(1 + g) [N, B], confidence logit [N])."""
    out = _lin(w, "head.2", _relu(_lin(w, "head.0", h)))
    return _softplus(out[:, :B]), out[:, B]


class GRUPredictor(BeamPredictor):
    """Keeps, per user, the last decoded gains and the age (slots since the last mode > 0
    feedback; inf before any) and a window of the last `window` input rows."""

    def __init__(self, cfg: dict, K: int, weights: str = "", **_):
        self.w = _weights(cfg, weights)
        self.B = int(cfg["system"]["B"])
        self.window = int(self.w["window"])
        self.speed = float(cfg["channel"]["speed_kmh"])
        self.snr = float(cfg["link"]["snr_db"])
        self.last = np.zeros((K, self.B))
        self.age = np.full(K, np.inf)
        self.hist = np.zeros((K, self.window, self.B + 4))

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        lg, c = gru_head(self.w, gru_forward(self.w, self.hist), self.B)
        return np.clip(np.expm1(lg), 0.0, None), _sigmoid(c)

    def update(self, ghat: np.ndarray, fresh: np.ndarray) -> None:
        # Only DECODED feedback enters the history; on other slots the gNB's own prediction (ghat
        # for non-fresh users) is not fed back into the network, as in the faculty History.
        self.last = np.where(fresh[:, None], ghat, self.last)
        self.age = np.where(fresh, 1.0, self.age + 1.0)
        row = history_row(self.last, self.age, fresh, self.speed, self.snr)
        self.hist = np.concatenate((self.hist[:, 1:], row[:, None, :]), axis=1)


class Persistence(BeamPredictor):
    """The dummy PersistencePredictor with the trained-model constructor signature, so that
    `models: trained` can combine the trained codec with persistence (if the GRU fails the S6 gate)."""

    def __init__(self, cfg: dict, K: int, **_):
        from . import predictor_tau
        from .dummy import PersistencePredictor
        self._p = PersistencePredictor(K, int(cfg["system"]["B"]), int(cfg["system"]["M"]), predictor_tau(cfg))

    def predict(self):
        return self._p.predict()

    def update(self, ghat, fresh):
        self._p.update(ghat, fresh)
