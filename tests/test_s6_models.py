"""Stage S6: trained codec and GRU predictor (numpy inference of the torch-trained weights)."""
import importlib.util

import numpy as np
import pytest

from tvcsi.config import ROOT, apply_overrides, load_config
from tvcsi.models.trained import (GRUPredictor, TrainedDecoder, TrainedEncoder, codec_input, gru_forward,
                                  gru_head, load_weights)
from tvcsi.sim import simulate

WEIGHTS = "data/models/s6_models.npz"


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_codec_sizes_and_quantisation(cfg):
    enc = TrainedEncoder(cfg, weights=WEIGHTS)
    hhat = np.random.default_rng(0).standard_normal((5, 3, cfg["system"]["M"])) * (1 + 1j)
    levels = 2 ** cfg["feedback"]["bits_per_value"]
    full = enc.encode(hhat, 3)
    for m, d in enumerate(cfg["feedback"]["code_len"], start=1):
        code = enc.encode(hhat, m)
        assert code.shape == (5, d)                                  # C_m = d_m x 4 bits
        assert np.array_equal(code, full[:, :d])                     # progressive: a prefix of mode 3
    q = (full + 1) * (levels - 1) / 2
    assert np.allclose(q, np.round(q)) and len(np.unique(np.round(q))) <= levels
    g = TrainedDecoder(cfg, weights=WEIGHTS).decode(full, 3)
    assert g.shape == (5, cfg["system"]["B"]) and (g >= 0).all()


def test_gru_predictor_interface(cfg):
    p = GRUPredictor(cfg, K=4, weights=WEIGHTS)
    g0, c0 = p.predict()
    assert g0.shape == (4, cfg["system"]["B"]) and ((c0 >= 0) & (c0 <= 1)).all()
    fresh = np.array([True, False, True, False])
    p.update(np.full((4, cfg["system"]["B"]), 3.0), fresh)
    assert np.array_equal(p.age, np.array([1.0, np.inf, 1.0, np.inf]))
    assert (p.last[~fresh] == 0).all() and (p.last[fresh] == 3.0).all()   # predictions are not stored


def test_numpy_inference_matches_torch():
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("train_models", ROOT / "scripts" / "train_models.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    w = load_weights(str(ROOT / WEIGHTS))
    cfg = load_config()
    B, M = cfg["system"]["B"], cfg["system"]["M"]
    codec = tm.Codec(2 * M + 1, B, cfg["feedback"]["code_len"], cfg["feedback"]["bits_per_value"])
    gru = tm.Predictor(B + 4, B, int(w["gru.weight_hh_l0"].shape[1]))
    sd = {k: torch.tensor(v, dtype=torch.float64) for k, v in w.items() if k != "window"}
    codec.double().load_state_dict({k: v for k, v in sd.items() if k.startswith(("enc", "dec"))})
    gru.double().load_state_dict({k: v for k, v in sd.items() if k.startswith(("gru", "head"))})
    rng = np.random.default_rng(1)
    hhat = (rng.standard_normal((50, M)) + 1j * rng.standard_normal((50, M)))
    x = codec_input(hhat, 20.0)
    enc = TrainedEncoder(apply_overrides(cfg, {"link.snr_db": 15.0, "link.pilot_snr_offset_db": 5.0}), weights=WEIGHTS)
    dec = TrainedDecoder(cfg, weights=WEIGHTS)
    z = enc.latent(hhat)
    with torch.no_grad():
        for m, d in enumerate(cfg["feedback"]["code_len"]):
            ref = codec(torch.tensor(x), m).numpy()
            assert np.allclose(dec.decode_log(z[:, :d]), ref, atol=1e-9), m
        X = rng.standard_normal((20, int(w["window"]), B + 4))
        ref_lg, ref_c = gru(torch.tensor(X))
    lg, c = gru_head(w, gru_forward(w, X), B)
    assert np.allclose(lg, ref_lg.numpy(), atol=1e-9) and np.allclose(c, ref_c.numpy(), atol=1e-9)


@pytest.mark.parametrize("predictor", ["Persistence", "GRUPredictor"])
def test_trained_models_in_the_simulator(cfg, predictor):
    """Short run with the trained models: budgets hold, and no true-channel leak (tamper test) for the
    schemes that do not need an estimator."""
    c = apply_overrides(cfg, {"models": "trained", "trained.predictor": f"tvcsi.models.trained:{predictor}",
                              "sim.T": 200, "sim.warmup": 10})
    names = ["Baseline", "fill_age", "fill_uncertainty", "prediction_only", "full_feedback"]
    garbage = np.random.default_rng(3)

    def tamper(shared):
        shared.g_true = garbage.exponential(50.0, shared.g_true.shape)
    a = simulate(c, 1000, names, record_decisions=True)
    b = simulate(c, 1000, names, record_decisions=True, tamper=tamper)
    for n in names:
        assert np.array_equal(a.decisions[n], b.decisions[n]), n
    assert a.metrics["full_feedback"]["regret"] < a.metrics["prediction_only"]["regret"]
