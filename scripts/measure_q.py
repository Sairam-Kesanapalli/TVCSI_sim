"""Measure the task-quality curves Q_type(SINR) with a 5G NR link-level simulation (Phase 2).

PHY abstraction, as in 3GPP system-level evaluation: the task quality of a user in a slot is the
probability that its semantic message arrives intact,

    Q_type(gamma) = 1 - BLER_type(gamma),

where BLER is the transport-block error rate of sending that message type's payload over the
user's PRBs at SINR gamma. The curve is measured, not assumed, with the standard-compliant chain
of Sionna (TS 38.212 / 38.214):

    TB CRC -> code-block segmentation + CB CRC -> LDPC (base graph 1 or 2) -> rate matching
    -> scrambling -> QAM (PDSCH MCS table) -> AWGN at SNR = Es/N0 -> APP demapping
    -> LDPC belief-propagation decoding -> block error if any information bit is wrong.

AWGN at the per-slot SINR is exact for this simulator: its channel is flat across the band, so a
slot has one SINR and no effective-SINR mapping is needed.

MCS: the lowest index of the configured PDSCH MCS table whose transport-block size (TS 38.214
5.1.3.2) holds the payload. A lower MCS is more robust, so this is the best fixed choice for a
message that must fit in one slot. If even the highest MCS is too small, the message cannot be
sent in one slot and Q = 0 at every SINR.

Usage:
    python scripts/measure_q.py select   # step 1: pick the carrier (rule in ASSUMPTIONS.md)
    python scripts/measure_q.py curves   # step 2: measure every curve the sweeps need
    python scripts/measure_q.py probe    # time one SNR point (sizing only)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from tvcsi.config import apply_overrides, load_config, resolve_path
from tvcsi.quality import (TYPE_ORDER, curve_key, link_plan, payload_bits_per_type,
                           prbs_per_user, re_per_prb)


@dataclass
class Plan:
    """One curve to measure: a message type sent over n_prb PRBs."""
    msg_type: str
    payload_bits: int
    n_prb: int
    mcs: int            # -1: no MCS of the table fits the payload in one slot
    qm: int             # bits per QAM symbol
    coderate: float     # target code rate of the MCS
    tbs: int            # transport-block size [bits] (>= payload; the rest is padding)
    coded_bits: int     # G, coded bits that fit in the allocation
    num_cbs: int        # code blocks the TB is split into
    base_coded_bits: int  # bits out of the LDPC encoder; < G => the rest repeats them (see Chain)
    bg: str = ""        # LDPC base graph chosen by TS 38.212 7.2.2 ("bg1" / "bg2")


def base_graph(tbs: int, rate: float) -> str:
    """LDPC base graph of TS 38.212 clause 7.2.2, from the TB size A and the MCS target rate R."""
    if tbs <= 292 or (tbs <= 3824 and rate <= 0.67) or rate <= 0.25:
        return "bg2"
    return "bg1"


# Lowest code rate of each mother code (BG1: 22 Z info bits in 66 Z coded bits; BG2: 10 Z in 50 Z).
# Below it, 5G NR rate matching repeats bits of its circular buffer (TS 38.212 5.4.2.1).
MOTHER_RATE = {"bg1": 1 / 3, "bg2": 1 / 5}


def select_mcs(cfg: dict, payload_bits: int, n_prb: int, msg_type: str) -> Plan:
    """Lowest MCS of the configured PDSCH table whose TBS holds the payload (TS 38.214 5.1.3)."""
    import torch  # noqa: F401  (sionna needs it loaded)
    from sionna.phy.nr.utils import calculate_codeword_bits, calculate_tb_size, decode_mcs_index
    r = cfg["radio"]
    for mcs in range(29):
        try:
            qm, rate = decode_mcs_index(mcs, table_index=r["mcs_table"], is_pusch=False)
        except ValueError:
            continue   # index reserved in this table
        qm, rate = int(qm), float(rate)
        tbs, cbs, ncb, *_ = calculate_tb_size(qm, rate, num_prbs=n_prb, num_ofdm_symbols=r["pdsch_symbols"],
                                              num_dmrs_per_prb=r["dmrs_re_per_prb"], return_cw_length=False)
        if int(tbs) >= payload_bits:
            G = calculate_codeword_bits(qm, n_prb, r["pdsch_symbols"], r["dmrs_re_per_prb"])
            # Sionna's LDPC encoder stops at the mother-code rate (1/3 for BG1, 1/5 for BG2). 5G NR goes
            # lower by repeating bits of its circular buffer; Chain does the same with what is left over.
            bg = base_graph(int(tbs), rate)
            n_cb_max = (int(int(cbs) / MOTHER_RATE[bg]) // qm) * qm
            base = min(int(G), n_cb_max * int(ncb))
            return Plan(msg_type, payload_bits, n_prb, mcs, qm, rate, int(tbs), int(G), int(ncb), base, bg)
    return Plan(msg_type, payload_bits, n_prb, -1, 0, 0.0, 0, 0, 0, 0, "")


class Chain:
    """Sionna 5G NR transport-block chain over AWGN for one Plan.

    Code rates below the mother-code rate (1/3 for BG1, 1/5 for BG2): the TB is encoded to
    base_coded_bits (the mother-code rate per code block, Sionna's lower limit), and the G - base
    remaining coded bits repeat those bits cyclically, as 5G NR rate matching repeats its circular
    buffer. The receiver adds the LLRs of all copies of a bit before decoding. Repetition only occurs
    at QPSK rates here, and QPSK over AWGN acts on every bit independently, so the order (and
    scrambling) of the copies does not change the result. The base graph Sionna picks is checked
    against TS 38.212 7.2.2.
    """

    def __init__(self, plan: Plan, bp_iter: int):
        import torch
        from sionna.phy.channel import AWGN
        from sionna.phy.mapping import BinarySource, Constellation, Demapper, Mapper
        from sionna.phy import config as sn_config
        from sionna.phy.nr import TBDecoder, TBEncoder
        self.torch = torch
        self.plan = plan
        self.device = sn_config.device          # Sionna's device: a GPU if one is available, else the CPU
        self.enc = TBEncoder(target_tb_size=plan.tbs, num_coded_bits=plan.base_coded_bits,
                             target_coderate=plan.coderate, num_bits_per_symbol=plan.qm, channel_type="PDSCH")
        self.dec = TBDecoder(self.enc, num_bp_iter=bp_iter)
        got = getattr(self.enc.ldpc_encoder, "_bg", None)
        if plan.bg and got != plan.bg:
            raise ValueError(f"Sionna chose base graph {got}, TS 38.212 gives {plan.bg} for {plan}")
        const = Constellation("qam", plan.qm)
        self.mapper, self.demapper = Mapper(constellation=const), Demapper("app", constellation=const)
        self.awgn, self.src = AWGN(), BinarySource()
        self.rep = None
        if plan.base_coded_bits < plan.coded_bits:
            if plan.qm != 2:
                raise ValueError("repetition is only exact for QPSK")
            self.rep = torch.arange(plan.coded_bits, device=self.device) % plan.base_coded_bits  # coded bit per slot bit

    def errors(self, snr_db: float, batch: int) -> int:
        """Number of transport blocks with at least one wrong information bit, out of `batch`."""
        torch = self.torch
        no = torch.tensor(10.0 ** (-snr_db / 10.0), device=self.device)   # unit-energy QAM: SNR = Es/N0 = 1/no
        with torch.inference_mode():
            b = self.src([batch, self.plan.tbs])
            c = self.enc(b)
            if self.rep is not None:
                c = c[..., self.rep]                                       # cyclic repetition
            llr = self.demapper(self.awgn(self.mapper(c), no), no)
            if self.rep is not None:                                       # combine the copies
                llr = torch.zeros(batch, self.plan.base_coded_bits, dtype=llr.dtype,
                                  device=llr.device).index_add_(1, self.rep, llr)
            b_hat, _ = self.dec(llr)
            return int((b_hat != b).any(dim=-1).sum())


def measure_curve(plan: Plan, ll: dict, seed: int) -> pd.DataFrame:
    """BLER on an SNR grid that covers the whole waterfall, from BLER >= 0.99 down to < stop_bler.

    1. Locate: from the Shannon SNR of the code (no code can work below it), one batch per 1 dB step
       until a batch has fewer than half of its blocks in error.
    2. Refine: snr_step_db steps from 1.5 dB below that point, walking further down until BLER >=
       0.99 and up until BLER < stop_bler. Each point runs until min_errors block errors or
       max_blocks blocks. Outside the refined range the curve is 0 (below) or 1 (above).
    """
    rows = []
    if plan.mcs < 0:
        return pd.DataFrame(rows)
    import torch
    import sionna
    sionna.phy.config.seed = seed
    torch.manual_seed(seed)
    chain = Chain(plan, ll["bp_iterations"])
    step, batch = ll["snr_step_db"], ll["batch"]
    eff = plan.tbs / (plan.coded_bits / plan.qm)              # information bits per QAM symbol
    shannon = 10 * np.log10(2.0 ** eff - 1.0)

    s = float(np.floor(shannon / step) * step)                # 1. locate
    while chain.errors(s, batch) > batch // 2 and s < ll["snr_max_db"]:
        s += 1.0

    done: dict[float, float] = {}

    tag = f"{plan.msg_type}/{plan.n_prb}prb"
    t_start = time.time()

    def point(snr):
        snr = round(snr, 4)
        if snr not in done:
            n = e = 0
            while e < ll["min_errors"] and n < ll["max_blocks"]:
                e += chain.errors(snr, batch)
                n += batch
            rows.append(dict(snr_db=snr, n_blocks=n, n_errors=e, bler=e / n))
            done[snr] = e / n
            print(f"    . {tag}: SNR {snr:6.2f} dB  BLER {e / n:.4f}  ({n} msgs, {time.time() - t_start:.0f}s)",
                  flush=True)
        return done[snr]

    lo = s - 1.5                                               # 2. refine
    while point(lo) < 0.99 and lo > ll["snr_min_db"]:
        lo -= step
    x = lo + step
    while x < ll["snr_max_db"] and point(x) >= ll["stop_bler"]:
        x += step
    df = pd.DataFrame(rows).sort_values("snr_db").reset_index(drop=True)
    for k, v in asdict(plan).items():
        df[k] = v
    return df


def _curve_job(args):
    plan, ll, seed, threads = args
    import torch
    torch.set_num_threads(threads)
    t0 = time.time()
    df = measure_curve(plan, ll, seed)
    return plan, df, time.time() - t0


def needed_plans(cfg: dict, carrier_prbs: int, K_values: list[int], types=TYPE_ORDER) -> list[Plan]:
    c = apply_overrides(cfg, {"radio.carrier_prbs": carrier_prbs})
    pay = payload_bits_per_type(c)
    plans, seen = [], set()
    for K in K_values:
        n_prb = prbs_per_user(c, K)
        for t in types:
            key = (t, n_prb)
            if key not in seen:
                seen.add(key)
                plans.append(select_mcs(c, pay[t], n_prb, t))
    return plans


def report(plan: Plan, df: pd.DataFrame, dt: float, prefix: str = "") -> None:
    tag = f"{prefix}{plan.msg_type:10s} n_prb={plan.n_prb:3d}"
    if plan.mcs < 0:
        print(f"  {tag}: no MCS fits {plan.payload_bits} bits in one slot -> Q = 0", flush=True)
    else:
        b10 = np.interp(0.1, df["bler"][::-1], df["snr_db"][::-1])
        print(f"  {tag}: MCS {plan.mcs:2d} (Qm={plan.qm}, R={plan.coderate:.3f}), TBS {plan.tbs}, "
              f"{plan.num_cbs} CB, {len(df)} points, BLER=0.1 at {b10:.2f} dB  [{dt:.0f}s]", flush=True)


def run_plans(plans: list[Plan], ll: dict, n_jobs: int, seeds: list[int] | None = None,
              on_done=None) -> list[tuple[Plan, pd.DataFrame, float]]:
    """Measure the plans in parallel. Each result is reported (and passed to on_done, e.g. to save
    it) as soon as ITS curve finishes, not in submission order. Returns results in plan order."""
    from concurrent.futures import as_completed
    threads = max(1, (os.cpu_count() or 2) // n_jobs)
    seeds = seeds if seeds is not None else [ll["seed"] + i for i in range(len(plans))]
    out: list = [None] * len(plans)
    with ProcessPoolExecutor(n_jobs) as pool:
        futs = {pool.submit(_curve_job, (p, ll, s, threads)): i for i, (p, s) in enumerate(zip(plans, seeds))}
        for n, fut in enumerate(as_completed(futs), start=1):
            i = futs[fut]
            try:
                plan, df, dt = fut.result()
            except Exception as exc:   # one broken curve must not stop the others from being saved
                print(f"  [{n}/{len(plans)}] FAILED {plans[i].msg_type} n_prb={plans[i].n_prb}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            report(plan, df, dt, prefix=f"[{n}/{len(plans)}] ")
            if on_done is not None:
                on_done(plan, df)
            out[i] = (plan, df, dt)
    return out


def preflight(plans: list[Plan], ll: dict) -> None:
    """Build every curve's chain and send a few messages at very high and very low SNR, BEFORE the long
    run: any setup error (base graph, rate, repetition) shows up in seconds, not after an hour."""
    for p in plans:
        if p.mcs < 0:
            continue
        chain = Chain(p, ll["bp_iterations"])
        hi, lo = chain.errors(40.0, 8), chain.errors(-20.0, 8)
        rep = f", repetition x{p.coded_bits / p.base_coded_bits:.2f}" if p.base_coded_bits < p.coded_bits else ""
        print(f"  preflight {p.msg_type:10s} n_prb={p.n_prb:3d}: MCS {p.mcs:2d} {p.bg}{rep} | "
              f"errors 40 dB: {hi}/8, -20 dB: {lo}/8", flush=True)
        if hi != 0 or lo != 8:
            raise RuntimeError(f"preflight failed for {p}: expected 0/8 and 8/8 errors")


def q_from_bler(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Monotone Q = 1 - BLER on the measured SNR points (Monte-Carlo wiggles removed)."""
    snr = df["snr_db"].to_numpy()
    q = np.maximum.accumulate(1.0 - df["bler"].to_numpy())
    return snr, q


def cmd_select(cfg: dict, n_jobs: int) -> None:
    """Pick the smallest candidate carrier on which a critical message is deliverable, with the TRUE
    best beam, in at least `feasibility_target` of the critical slots at the default point.

    Uses a coarse measurement and the VALIDATION seeds, so the test seeds play no part in the choice.
    """
    from tvcsi.codebook import beam_gains, dft_codebook
    from tvcsi.config import db2lin, lin2db
    from tvcsi.importance import CRITICAL
    from tvcsi.sim import make_realization, rng_streams

    ll = {**cfg["link_level"], **cfg["link_level"]["select"]}
    K, M = cfg["system"]["K"], cfg["system"]["M"]
    snr = db2lin(cfg["link"]["snr_db"])
    best = []
    for seed in cfg["seeds"]["val"]:
        r = make_realization(cfg, rng_streams(seed))
        g = lin2db(snr * beam_gains(r.h, dft_codebook(M, cfg["system"]["B"])).max(-1) / M)
        best.append(g[r.types == CRITICAL, cfg["sim"]["warmup"]:].ravel())
    best = np.concatenate(best)
    print(f"critical best-beam SINR on validation seeds: {len(best)} samples, "
          f"p10 = {np.percentile(best, 10):.2f} dB, median = {np.median(best):.2f} dB")

    plans = [needed_plans(cfg, c, [K], types=("critical",))[0] for c in ll["carrier_candidates"]]
    res = run_plans(plans, ll, n_jobs)
    q_min, target = cfg["quality"]["q_min"], ll["feasibility_target"]
    rows = []
    for carrier, (plan, df, _) in zip(ll["carrier_candidates"], res):
        if plan.mcs < 0:
            feas = 0.0
        else:
            s, q = q_from_bler(df)
            feas = float(np.mean(np.interp(best, s, q, left=0.0, right=q[-1]) >= q_min))
        rows.append(dict(carrier_prbs=carrier, n_prb=plan.n_prb, mcs=plan.mcs, tbs=plan.tbs, feasible_share=feas))
    sel = pd.DataFrame(rows)
    ok = sel[sel["feasible_share"] >= target]
    chosen = int(ok["carrier_prbs"].min()) if len(ok) else int(sel.loc[sel["feasible_share"].idxmax(), "carrier_prbs"])
    out = resolve_path(cfg, "data")
    out.mkdir(exist_ok=True)
    sel.assign(chosen=sel["carrier_prbs"] == chosen).to_csv(out / "carrier_selection.csv", index=False)
    print(sel.to_string(index=False))
    print(f"\nrule: smallest carrier with feasible_share >= {target}: carrier_prbs = {chosen}"
          + ("" if len(ok) else "  (NO candidate meets the target; the best one is reported)"))


CURVE_COLS = ["curve", "msg_type", "payload_bits", "n_prb", "mcs", "qm", "coderate", "tbs", "coded_bits",
              "base_coded_bits", "num_cbs", "snr_db", "n_blocks", "n_errors", "bler"]
# Measurement settings a saved curve depends on (besides the curve key itself).
_SETTING_KEYS = ("bp_iterations", "batch", "min_errors", "max_blocks", "stop_bler", "snr_step_db",
                 "snr_min_db", "snr_max_db", "seed")


def curve_seed(ll: dict, key: str) -> int:
    """Fixed seed per curve, from its key: a resumed run gives the same numbers as an uninterrupted one."""
    return int(ll["seed"]) + zlib.crc32(key.encode()) % 100_000


def part_path(cfg: dict, key: str) -> "Path":
    """File of one finished curve. The name holds a fingerprint of the measurement settings, so a curve
    measured with other settings is never reused."""
    ll = cfg["link_level"]
    fp = hashlib.sha1(json.dumps({k: ll[k] for k in _SETTING_KEYS}, sort_keys=True).encode()).hexdigest()[:8]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    return resolve_path(cfg, cfg["link_level"]["parts_dir"]) / f"{safe}__{fp}.csv"


def save_part(cfg: dict, plan: Plan, df: pd.DataFrame) -> None:
    """Write one finished curve atomically (temp file + rename), so a crash never leaves half a file."""
    key = curve_key(cfg, plan.msg_type, plan.n_prb)
    if plan.mcs < 0:   # record the infeasible curve explicitly: Q = 0 everywhere
        df = pd.DataFrame([dict(snr_db=np.nan, n_blocks=0, n_errors=0, bler=1.0, **asdict(plan))])
    path = part_path(cfg, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.assign(curve=key)[CURVE_COLS].to_csv(tmp, index=False)
    os.replace(tmp, path)


def cmd_curves(cfg: dict, n_jobs: int) -> None:
    """Measure every (type, PRBs per user) curve needed by the default point and the K sweep.

    Each curve is saved to link_level.parts_dir the moment it finishes; curves already saved with the
    same settings are skipped, so an interrupted run resumes where it stopped. When all curves exist
    they are combined into quality.curves (data/q_curves.csv).
    """
    import sionna
    import torch
    ll = cfg["link_level"]
    K_values = sorted({cfg["system"]["K"], *cfg["sweeps"]["users"]["values"]})
    plans = needed_plans(cfg, cfg["radio"]["carrier_prbs"], K_values)
    plans.sort(key=lambda p: -p.coded_bits * max(p.num_cbs, 1))   # heaviest first: balances the workers
    keys = [curve_key(cfg, p.msg_type, p.n_prb) for p in plans]
    todo = [i for i, k in enumerate(keys) if not part_path(cfg, k).exists()]
    print(f"{len(plans)} curves for carrier_prbs = {cfg['radio']['carrier_prbs']}, K in {K_values}: "
          f"{len(plans) - len(todo)} already saved, {len(todo)} to measure", flush=True)
    t0 = time.time()
    if todo:
        preflight([plans[i] for i in todo], ll)
        print("preflight OK", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()   # leave the GPU memory to the worker processes
        run_plans([plans[i] for i in todo], ll, n_jobs, seeds=[curve_seed(ll, keys[i]) for i in todo],
                  on_done=lambda plan, df: save_part(cfg, plan, df))
    missing = [k for k in keys if not part_path(cfg, k).exists()]
    if missing:
        sys.exit(f"{len(missing)} curves missing after the run: {missing}")
    out = resolve_path(cfg, cfg["quality"]["curves"])
    out.parent.mkdir(exist_ok=True)
    pd.concat([pd.read_csv(part_path(cfg, k)) for k in keys])[CURVE_COLS].to_csv(out, index=False)
    meta = dict(link_plan=link_plan(cfg), carrier_prbs=cfg["radio"]["carrier_prbs"], link_level=ll,
                sionna=sionna.__version__, torch=torch.__version__, python=sys.version.split()[0],
                device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                seconds_this_run=round(time.time() - t0), curves_measured_this_run=len(todo),
                seeds={k: curve_seed(ll, k) for k in keys},
                chain="TB CRC, CB segmentation + CRC, LDPC, rate matching, scrambling, QAM, AWGN, APP demapper, "
                      "BP decoder; block error = any information bit wrong")
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out} from {len(keys)} curves ({time.time() - t0:.0f}s this run)")


def snr_at_bler(snr_db: np.ndarray, bler: np.ndarray, target: float = 0.1) -> float:
    """SNR [dB] where a (decreasing) BLER curve crosses `target`, interpolated in log10(BLER)."""
    b = np.log10(np.clip(np.minimum.accumulate(bler), 1e-6, 1.0))
    return float(np.interp(-np.log10(1 / target), b[::-1], snr_db[::-1]))


def fbl_bound_db(k: int, n_real: int, eps: float = 0.1) -> float:
    """Least SNR = Es/N0 [dB] at which ANY code with k information bits over n_real real channel uses
    reaches block error eps on AWGN, by the normal approximation of Polyanskiy, Poor and Verdu (2010):
    k = n C - sqrt(n V) Qinv(eps) + log2(n) / 2. Practical codes sit some tenths of a dB above it."""
    from scipy.optimize import brentq
    from scipy.stats import norm

    def gap(sdb):
        s = 10 ** (sdb / 10)
        C = 0.5 * np.log2(1 + s)
        V = s * (s + 2) / (2 * (s + 1) ** 2) * np.log2(np.e) ** 2
        return n_real * C - np.sqrt(n_real * V) * norm.isf(eps) + 0.5 * np.log2(n_real) - k
    return float(brentq(gap, -30, 60))


def cmd_validate(cfg: dict) -> None:
    """Cross-check every measured curve against NVIDIA's pre-computed BLER tables (Sionna SYS).

    PHYAbstraction maps (MCS, allocated REs, SINR) to a transport-block error rate using its
    bundled PDSCH tables (per code block; TB error = some code block fails). The tables stop at a
    code-block size of 2000 bits and clamp above it, so for larger code blocks the comparison is
    approximate. Reports the SNR at BLER = 0.1 of both and their difference, writes
    data/q_validation.csv and figures/q_curves_validation.png.
    """
    import matplotlib.pyplot as plt
    import torch
    from sionna.sys import PHYAbstraction
    from tvcsi.plots import CATEGORICAL, apply_style, footnote, save
    from tvcsi.quality import re_per_prb

    df = pd.read_csv(resolve_path(cfg, cfg["quality"]["curves"]))
    phy = PHYAbstraction()
    rows, curves = [], []
    for key, g in df.groupby("curve", sort=False):
        g = g.sort_values("snr_db")
        p = g.iloc[0]
        if p["mcs"] < 0:
            rows.append(dict(curve=key, mcs=-1, note="infeasible: no MCS fits"))
            continue
        if p["mcs"] < 3:   # the bundled tables start at MCS 3 (rates < 1/5 need repetition)
            rows.append(dict(curve=key, msg_type=p["msg_type"], n_prb=int(p["n_prb"]), mcs=int(p["mcs"]),
                             note="no NVIDIA table for MCS 0-2"))
            continue
        n_re = int(p["n_prb"]) * re_per_prb(cfg)
        s = np.arange(g["snr_db"].min() - 1.0, g["snr_db"].max() + 1.0, 0.05)
        sinr = torch.tensor(10 ** (s / 10), dtype=torch.float32)
        out = phy(mcs_index=torch.full_like(sinr, int(p["mcs"]), dtype=torch.int32), sinr_eff=sinr,
                  num_allocated_re=torch.full_like(sinr, n_re, dtype=torch.int32),
                  mcs_table_index=cfg["radio"]["mcs_table"], mcs_category=1)
        tbler = out[3].cpu().numpy()
        s_meas = snr_at_bler(g["snr_db"].to_numpy(), g["bler"].to_numpy())
        s_tab = snr_at_bler(s, tbler)
        cb_bits = (int(p["tbs"]) + (24 if p["tbs"] > 3824 else 16)) / max(int(p["num_cbs"]), 1)
        fbl = fbl_bound_db(int(p["tbs"]) + (24 if p["tbs"] > 3824 else 16), 2 * int(p["coded_bits"]) // int(p["qm"]))
        rows.append(dict(curve=key, msg_type=p["msg_type"], n_prb=int(p["n_prb"]), mcs=int(p["mcs"]),
                         num_cbs=int(p["num_cbs"]), cb_bits=round(cb_bits),
                         snr_bler01_measured=round(s_meas, 3), snr_bler01_tables=round(s_tab, 3),
                         difference_db=round(s_meas - s_tab, 3),
                         tables_extrapolated=cb_bits > 2000, fbl_bound_db=round(fbl, 3),
                         measured_above_fbl_db=round(s_meas - fbl, 3), tables_above_fbl_db=round(s_tab - fbl, 3)))
        curves.append((key, p, g, s, tbler))
    val = pd.DataFrame(rows)
    out_csv = resolve_path(cfg, "data") / "q_validation.csv"
    val.to_csv(out_csv, index=False)
    print(val.to_string(index=False))
    ok = val.dropna(subset=["difference_db"])
    print(f"\n|difference| at BLER 0.1: mean {ok['difference_db'].abs().mean():.2f} dB, "
          f"max {ok['difference_db'].abs().max():.2f} dB over {len(ok)} curves; wrote {out_csv}")

    apply_style()
    K = cfg["system"]["K"]
    from tvcsi.quality import prbs_per_user
    n_def = prbs_per_user(cfg, K)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, (key, p, g, s, tbler) in enumerate(c for c in curves if int(c[1]["n_prb"]) == n_def):
        col = CATEGORICAL[i]
        ax.semilogy(g["snr_db"], np.clip(g["bler"], 1e-4, 1), "o", color=col, ms=5,
                    label=f"{p['msg_type']}: measured (MCS {int(p['mcs'])}, {int(p['tbs'])}-bit TB)")
        ax.semilogy(s, np.clip(tbler, 1e-4, 1), "-", color=col, lw=1.4, label=f"{p['msg_type']}: NVIDIA tables")
    ax.axhline(0.1, color="#8a8985", lw=1, ls=":")
    ax.set(xlabel="SNR = Es/N0 [dB]", ylabel="transport-block error rate", ylim=(1e-4, 1.5),
           title=f"Measured BLER vs Sionna's pre-computed tables (K = {K}, {n_def} PRBs per user)")
    ax.legend(fontsize=7.2, loc="lower left")
    footnote(fig, "points: this project's 5G NR PDSCH simulation; lines: sionna.sys.PHYAbstraction "
                  "(tables simulated up to 2000-bit code blocks, clamped above)")
    save(fig, resolve_path(cfg, cfg["paths"]["figures"]), "q_curves_validation", cfg["plots"]["dpi"])


def cmd_probe(cfg: dict) -> None:
    """Time one point of the heaviest default curve, to size the measurement."""
    import torch
    ll = cfg["link_level"]
    plan = needed_plans(cfg, cfg["radio"]["carrier_prbs"], [cfg["system"]["K"]], types=("critical",))[0]
    print(plan)
    if plan.mcs < 0:
        print("critical payload does not fit in one slot on this carrier")
        return
    chain = Chain(plan, ll["bp_iterations"])
    for threads in (torch.get_num_threads(), 1):
        torch.set_num_threads(threads)
        chain.errors(30.0, ll["batch"])   # warm-up
        t0 = time.time()
        e = chain.errors(30.0, ll["batch"])
        print(f"threads={threads}: {ll['batch']} blocks in {time.time() - t0:.2f}s ({e} errors at 30 dB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["select", "curves", "validate", "probe"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--jobs", type=int, default=0, help="parallel processes (0 = auto)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    n_jobs = args.jobs or max(1, (os.cpu_count() or 2) // 2)
    if args.command in ("select", "curves") and not args.jobs:
        import torch
        if torch.cuda.is_available():   # ~2 GB of GPU memory per process; 3 share one GPU well
            n_jobs = 3
    {"select": lambda: cmd_select(cfg, n_jobs), "curves": lambda: cmd_curves(cfg, n_jobs),
     "validate": lambda: cmd_validate(cfg), "probe": lambda: cmd_probe(cfg)}[args.command]()


if __name__ == "__main__":
    main()
