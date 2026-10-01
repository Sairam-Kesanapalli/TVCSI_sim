"""Spec section 9, tests 1-3, plus sanity checks of the channel and importance generators."""
import numpy as np
import pytest

from tvcsi.channel import array_response, generate_channels
from tvcsi.codebook import beam_gains, dft_codebook
from tvcsi.config import ar1_rho
from tvcsi.importance import CRITICAL, assign_types, generate_importance
from tvcsi.link import beam_regret, select_beam, task_quality, task_utility


# Test 1: a single-path channel aligned with beam b has best beam b and zero regret.
@pytest.mark.parametrize("M,B", [(32, 32), (16, 16), (8, 8)])
def test_codebook_alignment(M, B):
    W = dft_codebook(M, B)
    for b in range(B):
        theta = np.arcsin(2 * b / B - 1)
        h = np.sqrt(M) * array_response(np.array([theta]), M)      # one path, alpha = 1
        g = beam_gains(h, W)
        assert select_beam(g)[0] == b
        assert g[0, b] == pytest.approx(M)                        # full array gain
        assert beam_regret(g, np.array([b]), 1e-9)[0] == pytest.approx(0.0, abs=1e-9)


# Test 2: regret lies in [0, 1] and is 0 exactly when the true best beam is chosen.
def test_regret_range(cfg):
    rng = np.random.default_rng(0)
    h = generate_channels(cfg, rng, K=8, T=200).reshape(-1, cfg["system"]["M"])
    g = beam_gains(h, dft_codebook(cfg["system"]["M"], cfg["system"]["B"]))
    for _ in range(5):
        b = rng.integers(0, g.shape[1], g.shape[0])
        d = beam_regret(g, b, 1e-9)
        assert np.all(d >= 0) and np.all(d <= 1)
    # residual is the eps / max_b g offset of Eq. 15, tiny unless the channel is in a deep fade
    assert np.allclose(beam_regret(g, select_beam(g), 1e-9), 0.0, atol=1e-6)


# Test 3: Q is monotone in gamma and bounded; utility never exceeds importance.
def test_quality_and_utility():
    gamma = np.logspace(-3, 4, 2000)
    for g0 in (6.0, 10.0, 14.0):
        Q = task_quality(gamma, g0, 1.5)
        assert np.all(np.diff(Q) >= 0)
        assert np.all((Q >= 0) & (Q <= 1))
        assert task_quality(np.array([10 ** (g0 / 10)]), g0, 1.5)[0] == pytest.approx(0.5)
    v = np.random.default_rng(1).random(2000)
    assert np.all(task_utility(v, task_quality(gamma, 10.0, 1.5)) <= v)


def test_channel_statistics(cfg):
    h = generate_channels(cfg, np.random.default_rng(2), K=64, T=500)
    M = cfg["system"]["M"]
    assert h.shape == (64, 500, M)
    assert np.mean(np.sum(np.abs(h) ** 2, axis=-1)) == pytest.approx(M, rel=0.1)
    assert 0 < ar1_rho(cfg) < 1


def test_importance(cfg):
    types = assign_types(8, 0.25)
    assert list(types) == [0, 0, 1, 2, 1, 2, 1, 2]
    v = generate_importance(cfg, types, np.random.default_rng(3), T=5000)
    assert v.shape == (8, 5000) and v.min() >= 0.05 and v.max() <= 1.0
    # critical users spend a larger fraction of time in the high state (0.15/0.35 vs 0.1/0.4)
    high = v > 0.5
    assert high[types == CRITICAL].mean() > high[types != CRITICAL].mean()
