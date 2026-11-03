"""Task-quality models Q_k(gamma) (Eq. 20).

Two models share one interface, `quality(gamma_lin, user_axis)`:

* MeasuredQuality (default, Phase 2): Q = 1 - BLER of the user's semantic payload, measured by the
  5G NR link-level simulation of scripts/measure_q.py and stored in data/q_curves.csv.
* LogisticQuality (legacy, spec 3.6): Q = 1 / (1 + exp(-(gamma_dB - gamma0_dB) / s)).

The radio parameters that fix which measured curve a user gets (carrier, PRBs per user, payload,
PDSCH layout) live in the `radio` section of the config.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import lin2db, resolve_path
from .importance import TYPE_NAMES
from .link import task_quality

TYPE_ORDER = TYPE_NAMES   # ("critical", "normal", "background")

# Uniform SINR grid the measured curves are resampled onto, for a vectorised lookup.
GRID_LO_DB, GRID_HI_DB, GRID_STEP_DB = -40.0, 60.0, 0.01


# ----------------------------------------------------------------------------- radio helpers

def prbs_per_user(cfg: dict, K: int | None = None) -> int:
    """PRBs of one user: the carrier shared equally by the K users, floor(N_RB / K), at least 1."""
    K = cfg["system"]["K"] if K is None else K
    return max(1, cfg["radio"]["carrier_prbs"] // K)


def re_per_prb(cfg: dict) -> int:
    """Data resource elements per PRB in one slot: 12 subcarriers x PDSCH symbols - DMRS REs."""
    r = cfg["radio"]
    return 12 * r["pdsch_symbols"] - r["dmrs_re_per_prb"]


def payload_bits_per_type(cfg: dict) -> dict[str, int]:
    return {t: 8 * int(cfg["radio"]["payload_bytes"][t]) for t in TYPE_ORDER}


def link_plan(cfg: dict) -> dict:
    """Every radio parameter a measured curve depends on (besides type and PRBs)."""
    r = cfg["radio"]
    return {k: r[k] for k in ("pdsch_symbols", "dmrs_re_per_prb", "mcs_table")} | {
        "payload_bytes": dict(r["payload_bytes"])}


def curve_key(cfg: dict, msg_type: str, n_prb: int) -> str:
    """Identifier of one measured curve; changes whenever anything the curve depends on changes."""
    r = cfg["radio"]
    return (f"{msg_type}|{payload_bits_per_type(cfg)[msg_type]}b|{n_prb}prb|{r['pdsch_symbols']}sym|"
            f"{r['dmrs_re_per_prb']}dmrs|mcs_t{r['mcs_table']}")


# ----------------------------------------------------------------------------- models

class QualityModel:
    """Q_k(gamma) for the K users of a run. `user_axis` names the axis of gamma that indexes users."""

    def __call__(self, gamma_lin: np.ndarray, user_axis: int = -1) -> np.ndarray:
        raise NotImplementedError


class LogisticQuality(QualityModel):
    """Legacy logistic Q (Eq. 20 as specified in spec 3.6), bit-identical to link.task_quality."""

    def __init__(self, gamma0_db: np.ndarray, s_db: float):
        self.gamma0_db = np.asarray(gamma0_db, dtype=float)
        self.s_db = s_db

    def __call__(self, gamma_lin, user_axis=-1):
        g = np.asarray(gamma_lin)
        shape = [1] * g.ndim
        shape[user_axis] = len(self.gamma0_db)
        return task_quality(g, self.gamma0_db.reshape(shape), self.s_db)


class MeasuredQuality(QualityModel):
    """Measured Q = 1 - BLER, one curve per user, linearly interpolated in dB.

    table: [C, G] Q of each distinct curve on the uniform grid; row_of_user: [K] curve of each user.
    Below the grid Q = table[:, 0] (0), above it table[:, -1] (1).
    """

    def __init__(self, table: np.ndarray, row_of_user: np.ndarray, keys: list[str]):
        self.table = np.ascontiguousarray(table, dtype=float)
        self.row_of_user = np.asarray(row_of_user, dtype=np.intp)
        self.keys = keys
        self._flat = self.table.ravel()
        self._n = self.table.shape[1]

    def __call__(self, gamma_lin, user_axis=-1):
        g = np.moveaxis(np.asarray(gamma_lin, dtype=float), user_axis, -1)   # users last
        x = (lin2db(g) - GRID_LO_DB) / GRID_STEP_DB
        x = np.clip(x, 0.0, self._n - 1.000001)
        i = x.astype(np.intp)
        w = x - i
        flat = self.row_of_user * self._n + i
        q = self._flat[flat] * (1.0 - w) + self._flat[flat + 1] * w
        return np.moveaxis(q, -1, user_axis)


def grid() -> np.ndarray:
    n = int(round((GRID_HI_DB - GRID_LO_DB) / GRID_STEP_DB)) + 1
    return GRID_LO_DB + GRID_STEP_DB * np.arange(n)


def curve_on_grid(points: pd.DataFrame) -> np.ndarray:
    """Q on the uniform grid from one curve's measured rows (snr_db, bler).

    Q = 1 - BLER, made non-decreasing in SNR (removes Monte-Carlo wiggles; the true curve is
    monotone). One step below the lowest measured point (where BLER >= 0.99) Q is set to 0, and one
    step above the highest (where BLER < 1e-3) to 1; outside that range Q stays at 0 / 1.
    An infeasible curve (no MCS fits the payload, snr_db NaN) is 0 everywhere.
    """
    p = points.dropna(subset=["snr_db"]).sort_values("snr_db")
    if p.empty:
        return np.zeros(len(grid()))
    s = p["snr_db"].to_numpy()
    q = np.maximum.accumulate(1.0 - p["bler"].to_numpy())
    step = np.min(np.diff(s)) if len(s) > 1 else 0.25
    s = np.concatenate([[s[0] - step], s, [s[-1] + step]])
    q = np.concatenate([[0.0], q, [1.0]])
    return np.interp(grid(), s, q, left=0.0, right=1.0)


@lru_cache(maxsize=8)
def _load_curves(path: str, mtime: float) -> dict[str, np.ndarray]:
    df = pd.read_csv(path)
    return {key: curve_on_grid(g) for key, g in df.groupby("curve", sort=False)}


def load_curves(cfg: dict) -> dict[str, np.ndarray]:
    path = Path(resolve_path(cfg, cfg["quality"]["curves"]))
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python scripts/measure_q.py curves` first")
    return _load_curves(str(path), path.stat().st_mtime)


def build_quality(cfg: dict, types: np.ndarray) -> QualityModel:
    """The quality model of one run, from cfg["quality"]["model"]."""
    model = cfg["quality"]["model"]
    if model == "logistic":
        from .importance import gamma0_per_user
        return LogisticQuality(gamma0_per_user(cfg, types), cfg["quality"]["s_db"])
    if model != "measured":
        raise ValueError(f"unknown quality.model {model!r}")
    curves = load_curves(cfg)
    n_prb = prbs_per_user(cfg)
    keys = [curve_key(cfg, TYPE_ORDER[t], n_prb) for t in types]
    missing = sorted(set(keys) - set(curves))
    if missing:
        raise KeyError(f"no measured curve for {missing}: run `python scripts/measure_q.py curves` "
                       "for this configuration")
    uniq = list(dict.fromkeys(keys))
    table = np.stack([curves[k] for k in uniq])
    return MeasuredQuality(table, np.array([uniq.index(k) for k in keys]), uniq)
