"""Task-Value of CSI estimator (spec section 6, Eq. 32).

The gNB cannot observe the regret a mode will produce in the current slot. It instead uses a
lookup table of the mean realised regret r_hat[m][bin] per feedback mode and predictor-confidence
bin, fitted offline on validation seeds. Expected utilities are built only from the predictor
output, the importance v_k and that table; no true-channel quantity enters.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .link import task_quality


@dataclass
class TVCSITable:
    edges: np.ndarray     # interior confidence quantile edges, ascending; n_bins = len(edges) + 1
    r_hat: np.ndarray     # [M_f + 1, n_bins] mean regret per mode and confidence bin
    r_pooled: np.ndarray  # [M_f + 1] mean regret per mode over all bins (no-confidence ablation)
    counts: np.ndarray    # [M_f + 1, n_bins] samples behind each cell

    def bin_of(self, conf: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.edges, conf, side="right")

    def regret(self, conf: np.ndarray, use_confidence: bool = True) -> np.ndarray:
        """Expected regret for every mode, shape [K, M_f + 1]."""
        if not use_confidence:
            return np.broadcast_to(self.r_pooled, (len(conf), len(self.r_pooled)))
        return self.r_hat[:, self.bin_of(conf)].T

    def to_json(self, path: str | Path, meta: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"edges": self.edges.tolist(), "r_hat": self.r_hat.tolist(),
                       "r_pooled": self.r_pooled.tolist(), "counts": self.counts.tolist(),
                       "meta": meta or {}}, f, indent=1)

    @classmethod
    def from_json(cls, path: str | Path) -> "TVCSITable":
        with open(path) as f:
            d = json.load(f)
        return cls(np.array(d["edges"]), np.array(d["r_hat"]), np.array(d["r_pooled"]),
                   np.array(d["counts"]))


def fit_table(conf: np.ndarray, regret: np.ndarray, n_bins: int) -> TVCSITable:
    """Fit r_hat[m][bin] = mean regret of mode m among samples whose confidence falls in bin.

    conf: [N] predictor confidence at decision time. regret: [N, M_f + 1] realised regret had
    each mode been used in that slot. Bin edges are confidence quantiles; duplicate edges (many
    identical confidences, e.g. exactly 0 before the first feedback) are merged, so fewer than
    n_bins bins can result. An edge equal to the smallest confidence would only open an empty
    bin, so it is dropped as well. Empty cells fall back to the mode's pooled mean.
    """
    q = np.quantile(conf, np.arange(1, n_bins) / n_bins)
    edges = np.unique(q)
    edges = edges[edges > conf.min()]
    bins = np.searchsorted(edges, conf, side="right")
    nb, nm = len(edges) + 1, regret.shape[1]
    counts = np.zeros((nm, nb), dtype=np.int64)
    sums = np.zeros((nm, nb))
    for m in range(nm):
        counts[m] = np.bincount(bins, minlength=nb)
        sums[m] = np.bincount(bins, weights=regret[:, m], minlength=nb)
    pooled = regret.mean(axis=0)
    r_hat = np.where(counts > 0, sums / np.maximum(counts, 1), pooled[:, None])
    return TVCSITable(edges, r_hat, pooled, counts)


def gamma_ref(ghat0: np.ndarray, snr_lin: float, M: int) -> np.ndarray:
    """gNB's belief about the attainable SINR, gamma_ref_k = snr * max_b ghat0_kb / M (spec 6)."""
    return snr_lin * ghat0.max(axis=-1) / M


def expected_utility(table: TVCSITable, gref: np.ndarray, conf: np.ndarray, v: np.ndarray,
                     gamma0_db: np.ndarray, s_db: float, use_confidence: bool = True,
                     use_importance: bool = True) -> np.ndarray:
    """Expected utility Uhat[k][m] = v_k Q_k(gamma_ref_k (1 - r_hat[m][bin(c_k)])) (spec 6).

    Returns [K, M_f + 1]. The ablations drop the confidence (pooled regret per mode) or the
    importance (v_k = 1).
    """
    r = table.regret(conf, use_confidence)
    gamma = gref[:, None] * (1.0 - r)
    Q = task_quality(gamma, gamma0_db[:, None], s_db)
    w = v if use_importance else np.ones_like(v)
    return w[:, None] * Q


def tvcsi_scores(U_hat: np.ndarray, costs: np.ndarray, eps: float) -> np.ndarray:
    """TVCSI_{k,m} = (Uhat_k^m - Uhat_k^0) / (C_m + eps) for m > 0 (Eq. 32). Returns [K, M_f]."""
    return (U_hat[:, 1:] - U_hat[:, :1]) / (costs[1:] + eps)
