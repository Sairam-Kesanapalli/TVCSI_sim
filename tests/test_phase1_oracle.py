"""Phase 1: true best-beam feasibility oracle and the missing constraints of P1 (Eq. 39).

These tests check properties that must hold exactly, whatever the models or the schemes do.
"""
import numpy as np
import pytest

from tvcsi.config import load_config
from tvcsi.link import deliverable_bits
from tvcsi.sim import ORACLE, fit_tvcsi_table, simulate

ALL = ["Exact", "Optimised", "Proposed", "Baseline", "prediction_only", "importance_only",
       "snr_threshold", "full_feedback", "Proposed_no_confidence", "Proposed_no_importance"]


@pytest.fixture(scope="module")
def short_cfg():
    return load_config(overrides={"sim.T": 600, "sim.warmup": 50})


@pytest.fixture(scope="module")
def table(short_cfg):
    return fit_tvcsi_table(load_config(overrides={"sim.T": 600, "seeds.val": [2000, 2001]}))


@pytest.fixture(scope="module")
def run(short_cfg, table):
    return simulate(short_cfg, 1000, ALL, table=table).metrics


def test_oracle_bounds_every_scheme(run):
    """Perfect CSI for everyone: no scheme (not even the genie or full feedback) can beat it."""
    o = run[ORACLE]
    for name in ALL:
        assert run[name]["utility"] <= o["utility"] + 1e-12, name
        assert run[name]["violation_rate"] >= o["violation_rate"] - 1e-12, name
        assert run[name]["payload_miss_rate"] >= o["payload_miss_rate"] - 1e-12, name
    assert o["regret"] == 0.0
    assert np.isnan(o["bits"])   # the oracle uses no feedback channel; its cost is undefined


def test_violation_decomposition(run):
    """Every infeasible critical slot is a violation for every scheme, so
    violation_rate = infeasible_share + (1 - infeasible_share) * violation_rate_feasible."""
    for name in ALL + [ORACLE]:
        m = run[name]
        p_inf = m["crit_infeasible_share"]
        assert m["violation_rate"] == pytest.approx(p_inf + (1 - p_inf) * m["violation_rate_feasible"], abs=1e-12), name
        # the same identity for the payload constraint (39i)
        q_inf = m["payload_infeasible_share"]
        assert m["payload_miss_rate"] == pytest.approx(q_inf + (1 - q_inf) * m["payload_miss_rate_feasible"], abs=1e-12), name
    # The oracle fails exactly in the infeasible slots and never in a feasible one.
    assert run[ORACLE]["violation_rate_feasible"] == 0.0
    assert run[ORACLE]["payload_miss_rate_feasible"] == 0.0


def test_feasibility_is_channel_only(short_cfg, table, run):
    """Feasibility depends on the channel alone, so it is the same for every scheme and does not
    change with the set of schemes simulated."""
    shares = {run[n]["crit_infeasible_share"] for n in ALL + [ORACLE]}
    assert len(shares) == 1
    alone = simulate(short_cfg, 1000, ["Baseline"], table=table).metrics
    for name in (ORACLE, "Baseline"):
        assert alone[name].keys() == run[name].keys()
        for k in run[name]:   # NaN-aware: the oracle's bits are NaN, and NaN != NaN
            np.testing.assert_equal(alone[name][k], run[name][k], err_msg=f"{name}.{k}")


def test_per_user_reliability(run):
    """The worst critical user's feasible violation rate bounds the pooled one, and
    reliability_met is exactly 'worst <= eps'."""
    eps = load_config()["constraints"]["eps_violation"]
    for name in ALL:
        m = run[name]
        assert m["crit_worst_violation_feasible"] >= m["violation_rate_feasible"] - 1e-12, name
        assert m["reliability_met"] == float(m["crit_worst_violation_feasible"] <= eps), name


def test_payload_extremes(table):
    """Huge bandwidth: the best beam always fits the payload. Tiny bandwidth: nothing ever does.

    A scheme can still miss at huge bandwidth: a stale beam can point almost exactly into a null
    of the true channel (SINR ~ -36 dB was observed at seed 3), so for Baseline we only require
    that every miss is a feasible-slot miss.
    """
    base = {"sim.T": 200, "sim.warmup": 20}
    big = simulate(load_config(overrides={**base, "radio.prb_hz": 1e9}), 3, ["Baseline"], table=table).metrics
    tiny = simulate(load_config(overrides={**base, "radio.prb_hz": 1e-3}), 3, ["Baseline"], table=table).metrics
    assert big[ORACLE]["payload_infeasible_share"] == 0.0 and big[ORACLE]["payload_miss_rate"] == 0.0
    assert big["Baseline"]["payload_miss_rate"] == big["Baseline"]["payload_miss_rate_feasible"] < 0.01
    assert tiny["Baseline"]["payload_miss_rate"] == 1.0 and tiny[ORACLE]["payload_infeasible_share"] == 1.0


def test_user_bandwidth():
    from tvcsi.config import user_bandwidth_hz
    cfg = load_config()
    # the carrier's PRBs shared equally by the K users, 180 kHz each
    assert user_bandwidth_hz(cfg) == (cfg["radio"]["carrier_prbs"] // cfg["system"]["K"]) * 180e3
    assert user_bandwidth_hz(load_config(overrides={"system.K": 200})) == 180e3   # at least one PRB


def test_deliverable_bits():
    # 1 MHz x 1 ms x log2(1 + 3) = 2000 bits
    assert deliverable_bits(np.array(3.0), 1e6, 1e-3) == pytest.approx(2000.0)


def test_objective_reduces_to_utility(table):
    """With all lambdas zero, the P1 objective (39a) is the plain utility."""
    cfg = load_config(overrides={"sim.T": 200, "sim.warmup": 20, "constraints.lambda_F": 0.0,
                                 "constraints.lambda_B": 0.0, "constraints.lambda_R": 0.0})
    m = simulate(cfg, 3, ["Proposed", "Baseline"], table=table).metrics
    for name in ("Proposed", "Baseline"):
        assert m[name]["objective_p1"] == pytest.approx(m[name]["utility"], abs=1e-12)


def test_constraints_do_not_change_decisions(table):
    """The constraints are evaluation-only: changing them must not move a single decision."""
    base = {"sim.T": 300, "sim.warmup": 20}
    a = simulate(load_config(overrides=base), 4, ALL, table=table, record_decisions=True)
    b = simulate(load_config(overrides={**base, "constraints.eps_violation": 0.5, "constraints.lambda_F": 1.0,
                                        "constraints.payload_check_scale": 3.0, "radio.prb_hz": 1e3}),
                 4, ALL, table=table, record_decisions=True)
    for name in ALL:
        np.testing.assert_array_equal(a.decisions[name], b.decisions[name], err_msg=name)
        assert a.metrics[name]["utility"] == b.metrics[name]["utility"]
