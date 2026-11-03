"""Distinct coded bits per code block: our measurement (Sionna limit K'/R_mother) vs TS 38.212 5.4.2.1
(circular buffer N = 66 Zc / 50 Zc minus filler bits; LBRM inactive for these TB sizes)."""
import math, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from measure_q import base_graph
Z = sorted({a * 2**j for a in (2, 3, 5, 7, 9, 11, 13, 15) for j in range(8) if a * 2**j <= 384})
df = pd.read_csv("data/q_curves.csv").drop_duplicates("curve")
rows = []
for _, r in df.iterrows():
    tbs, C, G, base, qm, R = int(r.tbs), int(r.num_cbs), int(r.coded_bits), int(r.base_coded_bits), int(r.qm), float(r.coderate)
    bg = base_graph(tbs, R)
    B = tbs + 24
    Kp = B // C + (24 if C > 1 else 0)                     # bits per code block incl. CRC(s)
    Kb = 22 if bg == "bg1" else (10 if B > 640 else 9 if B > 560 else 8 if B > 192 else 6)
    Zc = min(z for z in Z if Kb * z >= Kp)
    K, N = (22 * Zc, 66 * Zc) if bg == "bg1" else (10 * Zc, 50 * Zc)
    usable = N - (K - Kp)                                  # filler bits are not sent
    ours = base // C
    rep = G > base
    e_cb = G // C
    real_distinct = min(e_cb, usable)
    ours_distinct = min(e_cb, ours)
    # Gaussian-capacity estimate of the SNR penalty of repeating instead of sending fresh parity:
    # threshold per transmitted symbol ~ (2^(2 r_code) - 1) / repetition, QPSK (2 coded bits / symbol)
    def thr(distinct):
        rc = Kp / distinct
        return (2 ** (2 * rc) - 1) / (e_cb / distinct)
    loss_db = 10 * math.log10(thr(ours_distinct) / thr(real_distinct)) if rep else 0.0
    rows.append(dict(curve=r.curve.split("|12sym")[0], bg=bg, C=C, Kp=Kp, Zc=Zc, G_per_cb=e_cb, ours=ours,
                     nr_buffer=usable, repetition=rep, missing_parity=(real_distinct - ours_distinct) if rep else 0,
                     est_loss_db=round(loss_db, 3)))
out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.to_string(index=False))
