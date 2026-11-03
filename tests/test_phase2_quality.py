"""Phase 2: measured task-quality curves (Q = 1 - BLER) and the quality-model interface."""
import numpy as np
import pandas as pd
import pytest

from tvcsi.config import load_config, resolve_path
from tvcsi.link import task_quality
from tvcsi.quality import (GRID_LO_DB, LogisticQuality, MeasuredQuality, TYPE_ORDER, build_quality,
                           curve_key, curve_on_grid, grid, payload_bits_per_type, prbs_per_user)
from tvcsi.sim import fit_tvcsi_table


def test_logistic_model_is_the_legacy_formula():
    """quality.model: logistic reproduces the old link.task_quality bit for bit, on either user axis."""
    rng = np.random.default_rng(0)
    g0 = np.array([14.0, 10.0, 6.0])
    m = LogisticQuality(g0, 1.5)
    gamma_km = rng.exponential(20.0, (3, 4))            # [K, modes], users first
    gamma_tk = rng.exponential(20.0, (50, 3))           # [T, K], users last
    np.testing.assert_array_equal(m(gamma_km, user_axis=0), task_quality(gamma_km, g0[:, None], 1.5))
    np.testing.assert_array_equal(m(gamma_tk, user_axis=-1), task_quality(gamma_tk, g0[None, :], 1.5))


def _linear_curve(lo_db, hi_db):
    """Measured rows of a curve whose Q rises linearly from 0 at lo_db to 1 at hi_db."""
    s = np.arange(lo_db, hi_db + 1e-9, 0.25)
    return pd.DataFrame(dict(snr_db=s, bler=1.0 - (s - lo_db) / (hi_db - lo_db)))


def test_measured_model_interpolates_each_users_curve():
    ramp = curve_on_grid(_linear_curve(0.0, 10.0))
    dead = curve_on_grid(pd.DataFrame(dict(snr_db=[np.nan], bler=[1.0])))   # infeasible: Q = 0
    m = MeasuredQuality(np.stack([ramp, dead]), np.array([0, 1, 0]), ["ramp", "dead"])
    db = np.array([[-20.0, -20.0, 2.5],
                   [5.0, 30.0, 7.3],
                   [60.0, 60.0, 9.99]])                   # [T, K]
    q = m(10 ** (db / 10), user_axis=-1)
    expected = np.array([[0.0, 0.0, 0.25],
                         [0.5, 0.0, 0.73],
                         [1.0, 0.0, 0.999]])
    np.testing.assert_allclose(q, expected, atol=1e-6)
    # the same numbers with users on the first axis
    np.testing.assert_allclose(m((10 ** (db / 10)).T, user_axis=0), expected.T, atol=1e-6)
    assert m(np.array([0.0, 0.0, 0.0])).max() == 0.0     # zero SINR (-inf dB) -> Q = 0


def test_curve_on_grid_is_monotone_and_bounded():
    rows = pd.DataFrame(dict(snr_db=[1.0, 1.25, 1.5, 1.75, 2.0], bler=[1.0, 0.6, 0.65, 0.1, 0.0]))  # MC wiggle
    q = curve_on_grid(rows)
    assert np.all(np.diff(q) >= 0) and q.min() == 0.0 and q.max() == 1.0
    g = grid()
    assert np.all(q[g < 0.75] == 0.0) and np.all(q[g > 2.25] == 1.0)
    assert q[np.argmin(abs(g - 1.5))] == pytest.approx(0.4)   # the wiggle (0.35) is lifted to 0.4
    assert len(q) == len(g) and g[0] == GRID_LO_DB


def test_missing_curve_is_an_error(tmp_path):
    path = tmp_path / "curves.csv"
    cfg = load_config(overrides={"quality.curves": str(path)})
    key = curve_key(cfg, "critical", prbs_per_user(cfg))
    _linear_curve(0, 5).assign(curve=key).to_csv(path, index=False)
    with pytest.raises(KeyError, match="normal"):
        build_quality(cfg, np.array([0, 1, 2]))          # normal / background curves are missing
    q = build_quality(cfg, np.array([0, 0]))             # critical only: fine
    assert isinstance(q, MeasuredQuality) and q.keys == [key]


def test_table_does_not_depend_on_quality_model():
    """The TVCSI table holds beam regret only, so its cache key may ignore quality / radio."""
    base = {"sim.T": 300, "sim.warmup": 20, "seeds.val": [2000], "quality.model": "logistic"}
    a = fit_tvcsi_table(load_config(overrides=base))
    b = fit_tvcsi_table(load_config(overrides={**base, "quality.gamma0_db.critical": 2.0, "quality.s_db": 0.3,
                                               "radio.carrier_prbs": 33}))
    np.testing.assert_array_equal(a.r_hat, b.r_hat)
    np.testing.assert_array_equal(a.edges, b.edges)


# ----------------------------------------------------------------------------- the measured file

@pytest.fixture(scope="module")
def curves():
    cfg = load_config()
    path = resolve_path(cfg, cfg["quality"]["curves"])
    if not path.exists():
        pytest.fail(f"{path} missing: run `python scripts/measure_q.py curves`")
    return cfg, pd.read_csv(path)


def test_measured_file_covers_every_configuration(curves):
    """Every (type, PRBs per user) the default point and the K sweep need has a curve."""
    cfg, df = curves
    for K in sorted({cfg["system"]["K"], *cfg["sweeps"]["users"]["values"]}):
        c = load_config(overrides={"system.K": K})
        for t in TYPE_ORDER:
            assert curve_key(c, t, prbs_per_user(c)) in set(df["curve"]), (K, t)


def test_measured_curves_span_the_waterfall(curves):
    """Each feasible curve starts where (almost) every block fails and ends where (almost) none does,
    and its transport block holds the payload."""
    cfg, df = curves
    ll = cfg["link_level"]
    pay = payload_bits_per_type(cfg)
    for key, g in df.groupby("curve"):
        g = g.sort_values("snr_db")
        if g["mcs"].iloc[0] < 0:
            continue
        assert g["bler"].iloc[0] >= 0.99, key
        assert g["bler"].iloc[-1] < ll["stop_bler"], key
        assert np.all(np.diff(g["snr_db"]) > 0), key
        assert (g["tbs"] >= g["payload_bits"]).all() and (g["payload_bits"] == pay[g["msg_type"].iloc[0]]).all()
        assert ((g["n_errors"] >= ll["min_errors"]) | (g["n_blocks"] >= ll["max_blocks"])).all(), key


def test_selected_mcs_is_the_lowest_that_fits(curves):
    """Re-derive the MCS choice from TS 38.214 (needs Sionna): the chosen MCS fits the payload, the
    next lower one does not."""
    pytest.importorskip("sionna")
    from sionna.phy.nr.utils import calculate_tb_size, decode_mcs_index
    cfg, df = curves
    r = cfg["radio"]
    for _, row in df.drop_duplicates("curve").iterrows():
        def tbs(mcs):
            qm, rate = decode_mcs_index(mcs, table_index=r["mcs_table"], is_pusch=False)
            return int(calculate_tb_size(int(qm), float(rate), num_prbs=int(row["n_prb"]),
                                         num_ofdm_symbols=r["pdsch_symbols"],
                                         num_dmrs_per_prb=r["dmrs_re_per_prb"], return_cw_length=False)[0])
        if row["mcs"] < 0:
            assert tbs(28) < row["payload_bits"]
            continue
        assert tbs(int(row["mcs"])) == row["tbs"] >= row["payload_bits"]
        if row["mcs"] > 0:
            assert tbs(int(row["mcs"]) - 1) < row["payload_bits"]
