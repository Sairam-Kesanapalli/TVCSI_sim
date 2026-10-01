"""Spec section 9, tests 5-7, plus unit tests of the dummy models and the TVCSI table."""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from tvcsi.metrics import cvar, cvar_min_eta
from tvcsi.models.dummy import DummyDecoder, DummyEncoder, PersistencePredictor
from tvcsi.codebook import dft_codebook, beam_gains
from tvcsi.solvers import brute_force, dp_mckp, greedy_tvcsi, upper_hull
from tvcsi.tvcsi import expected_utility, fit_table, tvcsi_scores

COSTS = np.array([0, 32, 64, 128])


def _instance(rng, K, concave=False):
    """Random utilities, increasing in mode on average but not necessarily concave in cost."""
    V = np.sort(rng.random((K, 4)), axis=1) if concave else rng.random((K, 4))
    budget = rng.uniform(0, K * 128)
    return V, budget


def _value(V, modes):
    return V[np.arange(len(modes)), modes].sum()


def _feasible(modes, budget):
    return COSTS[modes].sum() <= budget + 1e-9


# Test 5: DP equals brute force on 200 random instances with K <= 6.
def test_dp_equals_brute_force():
    rng = np.random.default_rng(5)
    for i in range(200):
        K = rng.integers(1, 7)
        V, budget = _instance(rng, K, concave=bool(i % 2))
        m_dp, m_bf = dp_mckp(V, COSTS, budget), brute_force(V, COSTS, budget)
        assert _feasible(m_dp, budget) and _feasible(m_bf, budget)
        assert _value(V, m_dp) == pytest.approx(_value(V, m_bf), abs=1e-12)


# Test 6: greedy never beats the DP optimum, stays feasible, and handles non-concave utilities.
def test_greedy_bounded_by_dp():
    rng = np.random.default_rng(6)
    ratios = []
    for i in range(500):
        K = rng.integers(1, 9)
        V, budget = _instance(rng, K, concave=bool(i % 2))
        m_g, m_dp = greedy_tvcsi(V, COSTS, budget), dp_mckp(V, COSTS, budget)
        assert _feasible(m_g, budget)
        vg, vdp = _value(V, m_g), _value(V, m_dp)
        assert vg <= vdp + 1e-12
        assert vg >= _value(V, np.zeros(K, dtype=int)) - 1e-12   # never worse than all-mode-0
        ratios.append(vg / vdp)
    print(f"mean greedy/DP ratio on random instances: {np.mean(ratios):.4f}")


def test_upper_hull_prunes_dominated_modes():
    # mode 1 lies below the chord from mode 0 to mode 2 -> pruned; mode 3 adds nothing -> cut
    vals = np.array([0.0, 0.1, 1.0, 0.9])
    assert upper_hull(COSTS, vals) == [0, 2]
    # concave increasing: everything kept
    assert upper_hull(COSTS, np.array([0.0, 0.5, 0.8, 1.0])) == [0, 1, 2, 3]
    # nothing beats mode 0
    assert upper_hull(COSTS, np.array([1.0, 0.5, 0.9, 1.0])) == [0]


def test_greedy_skips_dominated_mode():
    # user 0: mode 1 is LP-dominated, so greedy jumps 0 -> 2 directly
    V = np.array([[0.0, 0.1, 1.0, 1.05]])
    assert greedy_tvcsi(V, COSTS, 64)[0] == 2
    # with 63 bits the hull step 0 -> 2 is unaffordable and pruned mode 1 is never revisited:
    # this is the greedy gap the DP closes (DP picks mode 1 here)
    assert greedy_tvcsi(V, COSTS, 63)[0] == 0 and dp_mckp(V, COSTS, 63)[0] == 1


# Test 7: the sorted-tail CVaR matches the min-over-eta form of Eq. 37.
@pytest.mark.parametrize("alpha", [0.5, 0.9, 0.95, 0.99])
def test_cvar_forms_agree(alpha):
    rng = np.random.default_rng(7)
    for n in (20, 101, 1900):
        L = rng.gamma(2.0, 1.0, n)
        assert cvar(L, alpha) == pytest.approx(cvar_min_eta(L, alpha), rel=1e-10)
        f = lambda eta: eta + np.maximum(L - eta, 0).mean() / (1 - alpha)
        res = minimize_scalar(f, bounds=(L.min(), L.max()), method="bounded",
                              options={"xatol": 1e-10})
        assert cvar(L, alpha) == pytest.approx(res.fun, rel=1e-6)


def test_dummy_models_and_predictor():
    rng = np.random.default_rng(8)
    W = dft_codebook(32, 32)
    h = (rng.standard_normal((4, 3, 32)) + 1j * rng.standard_normal((4, 3, 32)))
    enc = DummyEncoder(W, [0.0, 0.0, 0.0], rng)
    ghat = DummyDecoder().decode(enc.encode(h, 2), 2)
    assert np.allclose(ghat, beam_gains(h[:, -1], W), rtol=1e-9, atol=1e-9)

    pred = PersistencePredictor(K=2, B=32, M=32, tau=5.0)
    g0, c = pred.predict()
    assert np.all(g0 == 32) and np.all(c == 0)
    pred.update(ghat[:2], np.array([True, False]))
    g0, c = pred.predict()
    assert np.allclose(g0, ghat[:2]) and c[0] == pytest.approx(np.exp(-1 / 5)) and c[1] == 0
    pred.update(g0, np.array([False, False]))
    assert pred.predict()[1][0] == pytest.approx(np.exp(-2 / 5))


def test_tvcsi_table_and_utility():
    rng = np.random.default_rng(9)
    conf = rng.random(10_000)
    regret = np.column_stack([1 - conf, np.full_like(conf, 0.3), np.full_like(conf, 0.2),
                              np.full_like(conf, 0.1)]) * rng.uniform(0.9, 1.1, (10_000, 4))
    tab = fit_table(conf, regret, 10)
    assert tab.r_hat.shape == (4, 10)
    assert np.all(np.diff(tab.r_hat[0]) < 0)          # staler prediction -> more regret
    U = expected_utility(tab, np.array([30.0, 30.0]), np.array([0.05, 0.95]), np.array([1.0, 0.2]),
                         np.array([10.0, 10.0]), 1.5)
    assert U.shape == (2, 4) and np.all(U <= np.array([[1.0], [0.2]]) + 1e-12)
    S = tvcsi_scores(U, COSTS, 1e-9)
    assert S[0].max() > S[1].max()                    # important + stale user is worth more bits
    no_conf = expected_utility(tab, np.array([30.0]), np.array([0.05]), np.array([1.0]),
                               np.array([10.0]), 1.5, use_confidence=False)
    assert no_conf[0, 0] > U[0, 0]                    # pooled regret hides the staleness
