"""Stage S5: the five faculty baselines, the Proposed_risk virtual queue, and the paired statistics."""
import numpy as np
import pandas as pd
import pytest

from tvcsi.config import apply_overrides, load_config
from tvcsi.importance import CRITICAL
from tvcsi.schemes import FILL_ORDERS, DecisionInputs, Scheme, belief_margin, fill_largest, risk_scheme
from tvcsi.sim import simulate
from tvcsi.stats import paired_holm

FILL = list(FILL_ORDERS)
COSTS = np.array([0, 32, 64, 128])


# ---------------------------------------------------------------- fill rule (faculty baseline_modes)

def _faculty_fill(order, costs, budget):
    """Literal re-statement of taskv2/policy.py::baseline_modes' filling loop (faculty code needs torch)."""
    modes = np.zeros(len(order), dtype=int)
    left = int(budget)
    for k in order:
        feasible = np.where(costs <= left)[0]
        mode = int(feasible[-1])
        if mode == 0:
            break
        modes[k] = mode
        left -= int(costs[mode])
    return modes


@pytest.mark.parametrize("budget", [0, 31, 32, 96, 160, 256, 1024])
def test_fill_largest_matches_faculty_rule(budget):
    rng = np.random.default_rng(budget)
    for _ in range(20):
        order = rng.permutation(8)
        assert np.array_equal(fill_largest(order, COSTS, budget, 8), _faculty_fill(order, COSTS, budget))


def test_fill_orders():
    v = np.array([0.1, 0.9, 0.5, 0.2])
    age = np.array([3.0, np.inf, 1.0, 7.0])
    ghat0 = np.array([[1.0, 0.9, 0.1], [5.0, 1.0, 0.0], [2.0, 0.1, 0.1], [0.5, 0.49, 0.0]])
    inp = DecisionInputs(t=5, U_hat=None, gamma_ref=np.zeros(4), v=v, costs=COSTS, budget=256.0,
                         ghat0=ghat0, age=age)
    assert list(FILL_ORDERS["fill_periodic"](inp)) == [1, 2, 3, 0]       # roll by t mod K
    assert list(FILL_ORDERS["fill_age"](inp)) == [1, 3, 0, 2]            # never fed back first
    assert list(FILL_ORDERS["fill_importance"](inp)) == [1, 2, 3, 0]
    assert list(FILL_ORDERS["fill_channel"](inp)) == [3, 0, 2, 1]        # weakest believed link first
    assert list(FILL_ORDERS["fill_uncertainty"](inp)) == list(np.argsort(belief_margin(ghat0), kind="stable"))
    assert FILL_ORDERS["fill_uncertainty"](inp)[0] == 3                  # 0.5 vs 0.49: least sure


@pytest.fixture(scope="module")
def short_cfg():
    return load_config(overrides={"sim.T": 300, "sim.warmup": 20})


@pytest.mark.parametrize("frac", [0.05, 0.25, 1.0])
def test_fill_baselines_respect_and_fill_budget(short_cfg, frac):
    cfg = apply_overrides(short_cfg, {"feedback.budget_frac": frac})
    res = simulate(cfg, 1000, FILL, record_decisions=True)          # simulate() asserts the budget
    budget = 8 * 128 * frac
    for n in FILL:
        bits = COSTS[res.decisions[n].astype(int)].sum(axis=1)
        assert (bits <= budget + 1e-9).all(), n
        assert (budget - bits < 32).all() or (bits == 8 * 128).all(), n      # nothing affordable left


# ---------------------------------------------------------------- Proposed_risk

def test_risk_with_zero_weight_is_proposed(short_cfg, estimator):
    res = simulate(short_cfg, 1001, ["Proposed", risk_scheme("risk0", 0.0, 0.1)], estimator=estimator,
                   record_decisions=True)
    assert np.array_equal(res.decisions["Proposed"], res.decisions["risk0"])


def test_risk_queue_behaviour(short_cfg, estimator):
    """The queue stays >= 0, is zero for non-critical users, and a positive weight changes decisions
    (it only acts on critical users)."""
    seen = []
    inner = risk_scheme("risk", 1.0, 0.0)          # eps 0: every avoidable expected outage counts

    def decide(inp, _):
        seen.append(inp.types.copy())
        return inner.decide(inp, _)
    res = simulate(short_cfg, 1002, ["Proposed", Scheme("risk", decide, "learned")], estimator=estimator,
                   record_decisions=True)
    assert not np.array_equal(res.decisions["Proposed"], res.decisions["risk"])
    assert (seen[0] == CRITICAL).any()


def test_risk_no_true_channel_leak(short_cfg, estimator):
    garbage = np.random.default_rng(5)

    def tamper(shared):
        shared.g_true = garbage.exponential(50.0, shared.g_true.shape)
    a = simulate(short_cfg, 7, ["Proposed_risk"] + FILL, estimator=estimator, record_decisions=True)
    b = simulate(short_cfg, 7, ["Proposed_risk"] + FILL, estimator=estimator, record_decisions=True, tamper=tamper)
    for n in ["Proposed_risk"] + FILL:
        assert np.array_equal(a.decisions[n], b.decisions[n]), n


# ---------------------------------------------------------------- statistics

def test_paired_holm():
    rng = np.random.default_rng(0)
    seeds = np.arange(10)
    rows = []
    for s in seeds:
        base = rng.normal()
        rows += [dict(seed=s, scheme="A", utility=base + 1.0), dict(seed=s, scheme="B", utility=base + rng.normal(0, 0.01)),
                 dict(seed=s, scheme="C", utility=base + 1.0 + rng.normal(0, 0.01))]
    out = paired_holm(pd.DataFrame(rows), "A")
    b = out.set_index("other").loc["B"]
    assert b["mean_diff"] == pytest.approx(1.0, abs=0.02) and b["wins"] == 10 and b["p_holm"] <= b["p"] * 2 + 1e-12
    c = out.set_index("other").loc["C"]
    assert c["p_holm"] > 0.05                                             # no real difference
    assert (out["p_holm"] >= out["p"] - 1e-15).all() and (out["p_holm"] <= 1).all()
