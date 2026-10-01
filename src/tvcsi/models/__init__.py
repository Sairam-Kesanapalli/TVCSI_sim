"""Model factory: dummy placeholders or trained classes loaded by dotted path from the config."""
from __future__ import annotations

import importlib
from typing import Callable

import numpy as np

from ..config import ar1_rho
from .base import BeamPredictor, Decoder, Encoder
from .dummy import DummyDecoder, DummyEncoder, PersistencePredictor


def predictor_tau(cfg: dict) -> float:
    """Confidence time constant: tau = 1 / (1 - rho) for the configured speed, or a fixed number."""
    tau = cfg["dummy"]["tau"]
    return 1.0 / (1.0 - ar1_rho(cfg)) if tau == "auto" else float(tau)


def build_models(cfg: dict, W: np.ndarray, rng: np.random.Generator
                 ) -> tuple[Encoder, Decoder, Callable[[int], BeamPredictor]]:
    """Returns (encoder, decoder, predictor_factory). predictor_factory(K) makes one fresh
    predictor; every scheme gets its own because its history depends on its own decisions."""
    M, B = cfg["system"]["M"], cfg["system"]["B"]
    if cfg["models"] == "dummy":
        tau = predictor_tau(cfg)
        return (DummyEncoder(W, cfg["dummy"]["sigma_m"], rng), DummyDecoder(),
                lambda K: PersistencePredictor(K, B, M, tau))
    if cfg["models"] == "trained":
        tr = cfg["trained"]
        kw = dict(tr.get("kwargs") or {})
        enc_cls, dec_cls, pred_cls = (_load(tr[k]) for k in ("encoder", "decoder", "predictor"))
        return (enc_cls(cfg=cfg, W=W, rng=rng, **kw), dec_cls(cfg=cfg, **kw),
                lambda K: pred_cls(cfg=cfg, K=K, **kw))
    raise ValueError(f"unknown models setting {cfg['models']!r}")


def _load(dotted: str):
    """Import 'package.module:ClassName'."""
    module, _, name = dotted.partition(":")
    return getattr(importlib.import_module(module), name)
