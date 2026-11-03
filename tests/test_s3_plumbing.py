"""Stage S3 gate: the schemes receive more gNB-side state, and no decision changes.

1. Golden regression: every scheme's decisions and metrics are bit-identical to the pre-S3 code
   (research/logs/S3/golden_pre_S3.npz, written by research/logs/S3/make_golden_pre_S3.py).
2. age / last_mode bookkeeping agrees with the persistence predictor and with the scheme's own
   past decisions.
3. Shapes and dtypes of the new fields; the arrays are copies.
4. No true-channel information reaches the new fields.
"""
import json

import numpy as np
import pytest

from tvcsi.config import ROOT, load_config, resolve_path
from tvcsi.importance import assign_types
from tvcsi.models import predictor_tau
from tvcsi.schemes import Scheme
from tvcsi.sim import ORACLE, simulate
from tvcsi.tvcsi import TVCSITable

GOLDEN = ROOT / "research" / "logs" / "S3" / "golden_pre_S3.npz"
FIELDS = ("ghat0", "conf", "age", "last_mode", "types")


# ---------------------------------------------------------------- 1. golden regression

def _golden():
    with np.load(GOLDEN) as z:
        data = {k: z[k] for k in z.files}
    return data, json.loads(str(data.pop("meta")))


GOLDEN_DATA, GOLDEN_META = _golden()


# S4 renamed the pre-S4 plug-in schemes (the names Optimised / Proposed / Proposed_no_importance now
# use the learned estimator, deliberately different). The renamed plug-in schemes must still reproduce
# the pre-S3 decisions and metrics exactly.
RENAMED_S4 = {"Optimised": "Optimised_plugin", "Proposed": "Proposed_plugin",
              "Proposed_no_confidence": "Proposed_plugin_no_confidence",
              "Proposed_no_importance": "Proposed_plugin_no_importance"}


@pytest.mark.parametrize("case", list(GOLDEN_META["cases"]))
@pytest.mark.parametrize("seed", GOLDEN_META["seeds"])
def test_golden_regression_bit_identical(case, seed):
    cfg = load_config(overrides=GOLDEN_META["cases"][case])
    table = TVCSITable.from_json(resolve_path(cfg, cfg["tvcsi"]["table_path"]))
    now = {n: RENAMED_S4.get(n, n) for n in GOLDEN_META["schemes"]}   # golden name -> current name
    res = simulate(cfg, seed, list(now.values()), table=table, record_decisions=True)
    names = GOLDEN_META["metric_names"]
    for name in GOLDEN_META["schemes"]:
        old = GOLDEN_DATA[f"{case}/{seed}/{name}/decisions"]
        new = res.decisions[now[name]]
        assert new.dtype == old.dtype, name
        assert np.array_equal(new, old), f"{case} seed {seed}: decisions of {name} ({now[name]}) changed"
    # Decisions (above) must be exactly equal on every computer. Metrics: equal up to floating-point
    # rounding. The golden file was written on one computer; another numpy version / CPU can sum the
    # same numbers in a different order, which changes the last digit (observed with the UNCHANGED
    # pre-S3 code: <= 4.4e-16 in 9 of 138 arrays, numpy 2.4.6 vs 2.5.4; research/logs/review_S3_S7/).
    # A real change in behaviour moves a decision or a metric by far more than 1e-12. On the computer
    # that wrote the golden file the metrics are bit-identical.
    for name in GOLDEN_META["schemes"] + [ORACLE]:     # all schemes + the BestBeam_oracle row
        m = res.metrics[now.get(name, name)]
        assert sorted(m) == names, name
        new = np.array([m[k] for k in names], dtype=float)
        old = GOLDEN_DATA[f"{case}/{seed}/{name}/metrics"]
        np.testing.assert_allclose(new, old, rtol=0, atol=1e-12, equal_nan=True,
                                   err_msg=f"{case} seed {seed}: metrics of {name} changed")


# ---------------------------------------------------------------- probe schemes

def _pattern(t: int, K: int) -> np.ndarray:
    """Deterministic mode pattern that mixes 'no feedback' gaps of different lengths with all modes."""
    k = np.arange(K)
    return np.where((t * (k + 1)) % 7 < 2, (t + k) % 3 + 1, 0).astype(np.int64)


def _probe(name: str, log: list, mutate: bool = False) -> Scheme:
    """Records (copies of) what it receives, decides by _pattern, and optionally scribbles over
    every input array afterwards to check that this cannot affect the simulator."""
    def decide(inp, _):
        log.append({f: np.array(getattr(inp, f), copy=True) for f in FIELDS} | {"t": inp.t})
        modes = _pattern(inp.t, len(inp.v))
        if mutate:
            inp.ghat0[...] = -1.0
            inp.conf[...] = 0.123
            inp.age[...] = -7.0
            inp.last_mode[...] = 3
            inp.types[...] = 99
        return modes
    return Scheme(name, decide, budget_exempt=True)


@pytest.fixture(scope="module")
def probe_cfg():
    return load_config(overrides={"sim.T": 300, "sim.warmup": 10})


# ---------------------------------------------------------------- 2. bookkeeping consistency

def test_age_and_last_mode_consistent(probe_cfg):
    cfg, K, T = probe_cfg, probe_cfg["system"]["K"], probe_cfg["sim"]["T"]
    log: list = []
    simulate(cfg, 1000, [_probe("probe", log)])
    assert [r["t"] for r in log] == list(range(T))
    tau = predictor_tau(cfg)

    age = np.full(K, np.inf)
    for r in log:
        t = r["t"]
        # recomputed independently from the probe's own decisions
        assert np.array_equal(r["age"], age), f"age at t={t}"
        prev = _pattern(t - 1, K) if t > 0 else np.zeros(K, dtype=np.int64)
        assert np.array_equal(r["last_mode"], prev), f"last_mode at t={t}"
        # the same age as the persistence predictor that produced conf
        assert np.array_equal(r["conf"], np.exp(-r["age"] / tau)), f"conf vs age at t={t}"
        age = np.where(_pattern(t, K) > 0, 1.0, age + 1.0)
    assert np.isinf(log[0]["age"]).all() and (log[0]["conf"] == 0).all()
    assert (log[-1]["age"] < np.inf).all()   # every user fed back at least once


# ---------------------------------------------------------------- 3. shapes, dtypes, copies

def test_field_shapes_and_dtypes(probe_cfg):
    cfg = probe_cfg
    K, B = cfg["system"]["K"], cfg["system"]["B"]
    log: list = []
    simulate(cfg, 1001, [_probe("probe", log)])
    for r in log:
        assert r["ghat0"].shape == (K, B) and r["ghat0"].dtype == np.float64
        assert r["conf"].shape == (K,) and r["conf"].dtype == np.float64
        assert ((r["conf"] >= 0) & (r["conf"] <= 1)).all()
        assert r["age"].shape == (K,) and r["age"].dtype == np.float64
        assert r["last_mode"].shape == (K,) and r["last_mode"].dtype == np.int64
        assert r["types"].shape == (K,) and np.issubdtype(r["types"].dtype, np.integer)
        assert np.array_equal(r["types"], assign_types(K, cfg["importance"]["crit_share"]))


def test_inputs_are_copies(probe_cfg):
    """A scheme that overwrites its inputs in place sees exactly what a well-behaved one sees."""
    clean_log, dirty_log = [], []
    simulate(probe_cfg, 1002, [_probe("clean", clean_log)])
    simulate(probe_cfg, 1002, [_probe("dirty", dirty_log, mutate=True)])
    assert len(clean_log) == len(dirty_log)
    for a, b in zip(clean_log, dirty_log):
        for f in FIELDS:
            assert np.array_equal(a[f], b[f]), f"{f} at t={a['t']}"


def test_existing_scheme_unaffected_by_mutating_neighbour(probe_cfg, estimator):
    """Running a mutating probe next to real schemes does not change their decisions."""
    table = TVCSITable.from_json(resolve_path(probe_cfg, probe_cfg["tvcsi"]["table_path"]))
    names = ["Proposed", "Proposed_plugin", "Baseline", "Exact"]
    alone = simulate(probe_cfg, 1003, names, table=table, estimator=estimator, record_decisions=True)
    with_probe = simulate(probe_cfg, 1003, names + [_probe("dirty", [], mutate=True)],
                          table=table, estimator=estimator, record_decisions=True)
    for n in names:
        assert np.array_equal(alone.decisions[n], with_probe.decisions[n]), n


# ---------------------------------------------------------------- 4. no true-channel leak

def test_new_fields_carry_no_true_channel(probe_cfg):
    """Replacing the true channel with garbage after the shared tables are built changes none of
    the new fields (they come only from the scheme's own predictor and decisions)."""
    garbage_rng = np.random.default_rng(123)

    def tamper(shared):
        shared.g_true = garbage_rng.exponential(50.0, shared.g_true.shape)

    clean_log, dirty_log = [], []
    simulate(probe_cfg, 7, [_probe("probe", clean_log)])
    simulate(probe_cfg, 7, [_probe("probe", dirty_log)], tamper=tamper)
    for a, b in zip(clean_log, dirty_log):
        for f in FIELDS:
            assert np.array_equal(a[f], b[f]), f"{f} at t={a['t']}"
