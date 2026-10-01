"""Spec section 9, tests 4 and 8-10."""
import io

import numpy as np
import pandas as pd
import pytest

from tvcsi.config import load_config
from tvcsi.sim import fit_tvcsi_table, metrics_rows, simulate
from tvcsi.tvcsi import TVCSITable

ALL = ["Exact", "Optimised", "Proposed", "Baseline", "prediction_only", "importance_only",
       "snr_threshold", "full_feedback", "Proposed_no_confidence", "Proposed_no_importance"]


@pytest.fixture(scope="module")
def default_cfg():
    return load_config()


@pytest.fixture(scope="module")
def table(default_cfg):
    return fit_tvcsi_table(default_cfg)


# Test 4: perfect CSI => zero regret with feedback, and Exact == Optimised == full feedback
# when the budget admits mode 3 for everyone.
def test_perfect_csi():
    cfg = load_config(overrides={"dummy.sigma_m": [0.0, 0.0, 0.0], "link.pilot_snr_offset_db": 300.0,
                                 "feedback.budget_frac": 1.0, "sim.T": 400, "sim.warmup": 20})
    # any table that says "fresh feedback has no regret, prediction has some"
    tab = TVCSITable(edges=np.array([0.5]), r_hat=np.array([[0.6, 0.3], [0, 0], [0, 0], [0, 0]]),
                     r_pooled=np.array([0.45, 0, 0, 0]), counts=np.ones((4, 2), dtype=int))
    m = simulate(cfg, 11, ["Exact", "Optimised", "full_feedback"], table=tab).metrics
    assert m["full_feedback"]["regret"] < 1e-6
    assert m["Optimised"]["regret"] < 1e-6
    assert m["Exact"]["regret"] < 1e-6
    assert m["Exact"]["utility"] == pytest.approx(m["full_feedback"]["utility"], abs=1e-9)
    assert m["Optimised"]["utility"] == pytest.approx(m["full_feedback"]["utility"], abs=1e-9)


# Test 8: over a full default run no scheme except full_feedback exceeds C_FB_max
# (simulate() hard-asserts the budget every slot for non-exempt schemes).
def test_budget_never_exceeded(default_cfg, table):
    res = simulate(default_cfg, default_cfg["seeds"]["test"][0], ALL, table=table)
    budget = 0.25 * 8 * 128
    for name in ALL:
        if name != "full_feedback":
            assert res.metrics[name]["bits"] <= budget + 1e-9
    assert res.metrics["full_feedback"]["bits"] == 8 * 128


# Test 9: identical seeds give byte-identical CSVs; different seeds give different numbers.
def test_reproducibility(table):
    cfg = load_config(overrides={"sim.T": 300, "sim.warmup": 20})

    def csv(seed):
        rows = metrics_rows(simulate(cfg, seed, ALL, table=table), seed=seed)
        buf = io.StringIO()
        pd.DataFrame(rows).drop(columns="seed").to_csv(buf, index=False)
        return buf.getvalue()

    assert csv(5) == csv(5)
    a, b = (pd.read_csv(io.StringIO(csv(s))) for s in (5, 6))
    assert not np.allclose(a["utility"], b["utility"])


# Test 10: replacing the true channel with garbage after the shared outcome tables are built
# changes nothing in the decisions of non-genie schemes (only Exact may react).
def test_no_true_channel_leak(table):
    cfg = load_config(overrides={"sim.T": 400})
    garbage_rng = np.random.default_rng(123)

    def tamper(shared):
        shared.g_true = garbage_rng.exponential(50.0, shared.g_true.shape)

    clean = simulate(cfg, 7, ALL, table=table, record_decisions=True)
    dirty = simulate(cfg, 7, ALL, table=table, record_decisions=True, tamper=tamper)
    for name in ALL:
        if name != "Exact":
            np.testing.assert_array_equal(clean.decisions[name], dirty.decisions[name], err_msg=name)
    # sanity: the tampering took effect
    assert not np.array_equal(clean.decisions["Exact"], dirty.decisions["Exact"])
    assert clean.metrics["prediction_only"]["utility"] != dirty.metrics["prediction_only"]["utility"]
