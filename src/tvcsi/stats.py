"""S5: seed-paired comparisons with Holm correction (same procedure as faculty taskv2/eval.py).

Each test seed is one independent replicate (slots within a run are correlated and are never
treated as independent observations).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def paired_holm(df: pd.DataFrame, ref: str, metric: str = "utility", others: list[str] | None = None,
                index: str = "seed") -> pd.DataFrame:
    """ref - other for every other scheme: mean paired difference, 95 % t half-width, wins,
    two-sided one-sample t-test p-value and its Holm-adjusted value (family = the listed others)."""
    piv = df.pivot_table(index=index, columns="scheme", values=metric)
    others = [c for c in (others or piv.columns) if c != ref and c in piv.columns]
    rows = []
    for o in others:
        d = (piv[ref] - piv[o]).dropna().to_numpy()
        n = len(d)
        sem = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        p = float(stats.ttest_1samp(d, 0.0).pvalue) if n > 1 and d.std(ddof=1) > 0 else (0.0 if n > 1 and d.mean() != 0 else 1.0)
        rows.append(dict(ref=ref, other=o, n=n, mean_diff=float(d.mean()),
                         ci95=float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else np.nan,
                         wins=int((d > 0).sum()), p=p))
    out = pd.DataFrame(rows).sort_values("p", kind="stable").reset_index(drop=True)
    m = len(out)
    running = 0.0
    holm = []
    for i, p in enumerate(out["p"]):
        running = max(running, (m - i) * p)
        holm.append(min(1.0, running))
    out["p_holm"] = holm
    return out
