"""Stage S6: train the progressive CSI codec and the GRU beam predictor (needs torch, CPU is enough).

python scripts/train_models.py [--out data/models/s6_models] [--epochs-codec 15] [--epochs-gru 12]

* Data: TRAIN seeds 3000-3019 over a speed / SNR grid (`models_train.grid`), the realisations the
  simulator itself uses (same seeds, same random streams). Epochs are selected on the VALIDATION
  seeds 2000-2004 (lowest mean beam regret). Diagnostics are reported on the TEST seeds 1000-1009.
* Codec (faculty ProgressiveCodec adapted): 2M + 1 inputs -> 128 -> 32 latent values, 4-bit STE
  quantisation, modes send the first 8 / 16 / 32 values; decoder 32 -> 128 -> B, output log(1 + g).
  Loss: MSE on log(1 + g) for every mode + 0.08 x cross-entropy on the best beam (faculty).
* GRU (faculty BeamGRU adapted): input rows from `tvcsi.models.trained.history_row` over a window,
  hidden 64, head 64 -> B + 1 (log(1 + g) and a confidence logit). Training histories come from
  random-feedback policies (P(mode 0) in `estimator.fit_p0`) with the TRAINED codec's decoded gains.
  Loss: MSE + 0.1 x BCE(confidence, 1[predicted beam within 10 % of the best]).
* Writes <out>.npz (numpy weights for tvcsi.models.trained) and <out>.json (settings, metrics).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch  # import torch before scipy / sklearn (Windows DLL order)
from torch import nn

import numpy as np

from tvcsi.codebook import beam_gains, dft_codebook
from tvcsi.config import apply_overrides, db2lin, load_config, resolve_path
from tvcsi.models.dummy import DummyDecoder, DummyEncoder
from tvcsi.models.trained import (TrainedDecoder, TrainedEncoder, codec_input, gru_forward, gru_head,
                                  history_row, load_weights)
from tvcsi.sim import make_realization, rng_streams

torch.set_num_threads(4)
LOG = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


# ----------------------------------------------------------------------------- configs / data

def grid_configs(cfg: dict) -> list[dict]:
    out = [cfg]
    for param, values in cfg["models_train"]["grid"].items():
        for v in values:
            c = apply_overrides(cfg, {param: v})
            if c != cfg:
                out.append(c)
    return out


def assignments(cfg: dict, split: str) -> list[tuple[dict, int]]:
    """(config, seed) pairs: the configurations are cycled over the seeds of the split; every
    configuration gets at least one seed (seeds are reused cyclically if there are fewer)."""
    cfgs, seeds = grid_configs(cfg), cfg["seeds"][split]
    n = max(len(cfgs), len(seeds))
    return [(cfgs[i % len(cfgs)], seeds[i % len(seeds)]) for i in range(n)]


def realization(c: dict, seed: int):
    real = make_realization(c, rng_streams(seed))
    W = dft_codebook(c["system"]["M"], c["system"]["B"])
    snr_pilot = c["link"]["snr_db"] + c["link"]["pilot_snr_offset_db"]
    return real, beam_gains(real.h, W), snr_pilot


def codec_data(pairs):
    X, Y = [], []
    for c, s in pairs:
        real, g, snr_pilot = realization(c, s)
        X.append(codec_input(real.hhat, snr_pilot).reshape(-1, 2 * c["system"]["M"] + 1))
        Y.append(np.log1p(g).reshape(-1, g.shape[-1]))
    return torch.tensor(np.vstack(X), dtype=torch.float32), torch.tensor(np.vstack(Y), dtype=torch.float32)


# ----------------------------------------------------------------------------- codec

class Codec(nn.Module):
    def __init__(self, n_in: int, B: int, dims: list[int], bits: int):
        super().__init__()
        self.dims, self.L = dims, 2 ** bits - 1
        self.enc = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Linear(128, dims[-1]), nn.Tanh())
        self.dec = nn.Sequential(nn.Linear(dims[-1], 128), nn.ReLU(), nn.Linear(128, B), nn.Softplus())

    def forward(self, x, mode_idx: int):
        z = self.enc(x)
        zq = torch.round((z + 1) * (self.L / 2)) / (self.L / 2) - 1
        z = z + (zq - z).detach()                                  # straight-through estimator
        mask = torch.zeros_like(z)
        mask[:, : self.dims[mode_idx]] = 1
        return self.dec(z * mask)


def regret_of(pred_log: np.ndarray, g_log: np.ndarray) -> tuple[float, float]:
    """Mean beam regret and top-1 accuracy of argmax(pred) against the true gains."""
    g = np.expm1(g_log)
    b = pred_log.argmax(-1)
    gb = np.take_along_axis(g, b[..., None], -1)[..., 0]
    return float((1 - gb / (g.max(-1) + 1e-9)).mean()), float((b == g.argmax(-1)).mean())


def train_codec(cfg, Xtr, Ytr, Xva, Yva, epochs: int, seed: int):
    torch.manual_seed(seed)
    dims = [int(d) for d in cfg["feedback"]["code_len"]]
    model = Codec(Xtr.shape[1], Ytr.shape[1], dims, int(cfg["feedback"]["bits_per_value"]))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    gen = torch.Generator().manual_seed(seed)
    best = (np.inf, None, -1)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(Xtr), 512):
            idx = perm[i:i + 512]
            x, y = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = 0.0
            target = y.argmax(-1)
            for m in range(len(dims)):
                p = model(x, m)
                loss = loss + ((p - y) ** 2).mean() + 0.08 * nn.functional.cross_entropy(p, target)
            (loss / len(dims)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            regs = [regret_of(model(Xva, m).numpy(), Yva.numpy())[0] for m in range(len(dims))]
        log(f"  codec epoch {ep + 1}: val regret per mode {np.round(regs, 4).tolist()}")
        if np.mean(regs) < best[0]:
            best = (float(np.mean(regs)), {k: v.clone() for k, v in model.state_dict().items()}, ep + 1)
    model.load_state_dict(best[1])
    log(f"  codec: chosen epoch {best[2]} (val mean regret {best[0]:.4f})")
    return model, best[2]


# ----------------------------------------------------------------------------- GRU

class Predictor(nn.Module):
    def __init__(self, n_in: int, B: int, hidden: int):
        super().__init__()
        self.B = B
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, B + 1))

    def forward(self, X):
        h, _ = self.gru(X)
        out = self.head(h[:, -1])
        return nn.functional.softplus(out[:, : self.B]), out[:, self.B]


def decoded_all(enc: TrainedEncoder, dec: TrainedDecoder, hhat: np.ndarray, dims: list[int]) -> np.ndarray:
    """Decoded gains for every user, slot and mode: [M_f, K, T, B]."""
    K, T, M = hhat.shape
    z = enc.latent(hhat.reshape(-1, M))
    return np.stack([np.clip(np.expm1(dec.decode_log(z[:, :d])), 0, None).reshape(K, T, -1) for d in dims])


def gru_sequences(cfg, pairs, weights_path: str, window: int, stride: int, p0s, policy_seed: int):
    """Windows of predictor input rows under random-feedback policies, with targets.

    Returns X [N, window, B+4], Y = log(1+g_true) [N, B], plus per-window persistence predictions
    (last decoded gains, M before any feedback) for the diagnostics."""
    X, Y, P = [], [], []
    for c, s in pairs:
        real, g, _ = realization(c, s)
        K, T, B = g.shape
        enc = TrainedEncoder(c, weights=weights_path)
        dec = TrainedDecoder(c, weights=weights_path)
        ghat = decoded_all(enc, dec, real.hhat, [int(d) for d in c["feedback"]["code_len"]])
        speed, snr = float(c["channel"]["speed_kmh"]), float(c["link"]["snr_db"])
        for j, p0 in enumerate(p0s):
            rng = np.random.default_rng([policy_seed, s, j])
            last = np.zeros((K, B))
            pers = np.full((K, B), float(c["system"]["M"]))   # PersistencePredictor prior
            age = np.full(K, np.inf)
            hist = np.zeros((K, window, B + 4))
            for t in range(T):
                if t % stride == 0:
                    X.append(hist.copy())
                    Y.append(np.log1p(g[:, t]))
                    P.append(np.log1p(pers))
                m = np.where(rng.random(K) < p0, 0, rng.integers(1, len(ghat) + 1, K))
                fresh = m > 0
                dec_t = ghat[np.maximum(m - 1, 0), np.arange(K), t]
                last = np.where(fresh[:, None], dec_t, last)
                pers = np.where(fresh[:, None], dec_t, pers)
                age = np.where(fresh, 1.0, age + 1.0)
                hist = np.concatenate((hist[:, 1:], history_row(last, age, fresh, speed, snr)[:, None]), axis=1)
    return (torch.tensor(np.vstack(X), dtype=torch.float32), torch.tensor(np.vstack(Y), dtype=torch.float32),
            np.vstack(P))


def train_gru(Xtr, Ytr, Xva, Yva, hidden: int, epochs: int, seed: int):
    torch.manual_seed(seed)
    B = Ytr.shape[1]
    model = Predictor(Xtr.shape[2], B, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    gen = torch.Generator().manual_seed(seed)
    best = (np.inf, None, -1)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(Xtr), 512):
            idx = perm[i:i + 512]
            x, y = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            lg, cl = model(x)
            with torch.no_grad():
                g = torch.expm1(y)
                gb = g.gather(1, lg.argmax(1, keepdim=True))[:, 0]
                good = (gb >= 0.9 * g.max(1).values).float()
            loss = ((lg - y) ** 2).mean() + 0.1 * nn.functional.binary_cross_entropy_with_logits(cl, good)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            reg, acc = regret_of(model(Xva)[0].numpy(), Yva.numpy())
        log(f"  gru epoch {ep + 1}: val regret {reg:.4f} acc {acc:.4f}")
        if reg < best[0]:
            best = (reg, {k: v.clone() for k, v in model.state_dict().items()}, ep + 1)
    model.load_state_dict(best[1])
    log(f"  gru: chosen epoch {best[2]} (val regret {best[0]:.4f})")
    return model, best[2]


# ----------------------------------------------------------------------------- diagnostics (test)

def paired_ci(d: np.ndarray) -> tuple[float, float]:
    from scipy import stats
    return float(d.mean()), float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))


def diagnostics(cfg, weights_path: str, window: int, stride: int) -> dict:
    """Test seeds: codec vs the dummy codec per mode; GRU vs persistence per configuration."""
    w = load_weights(str(Path(weights_path).resolve()))
    out = {"codec": {}, "predictor": {}}
    dims = [int(d) for d in cfg["feedback"]["code_len"]]
    # codec, default configuration
    rows = {"trained": [], "dummy": []}
    for s in cfg["seeds"]["test"]:
        real, g, _ = realization(cfg, s)
        K, T, M = real.hhat.shape
        enc, dec = TrainedEncoder(cfg, weights=weights_path), TrainedDecoder(cfg, weights=weights_path)
        tr = decoded_all(enc, dec, real.hhat, dims)
        W = dft_codebook(M, cfg["system"]["B"])
        de, dd = DummyEncoder(W, cfg["dummy"]["sigma_m"], rng_streams(s)["encoder"]), DummyDecoder()
        du = np.stack([np.stack([dd.decode(de.encode(real.hhat[:, t:t + 1], m), m) for t in range(T)], axis=1)
                       for m in range(1, len(dims) + 1)])
        gl = np.log1p(g)
        rows["trained"].append([regret_of(np.log1p(tr[m]), gl) for m in range(len(dims))])
        rows["dummy"].append([regret_of(np.log1p(du[m]), gl) for m in range(len(dims))])
    for k, v in rows.items():
        a = np.array(v)                                   # [seeds, modes, (regret, acc)]
        out["codec"][k] = {"regret_per_mode": a[:, :, 0].mean(0).round(4).tolist(),
                           "acc_per_mode": a[:, :, 1].mean(0).round(4).tolist()}
    # predictor: default and the speed extremes, random feedback with P(mode 0) = 0.75
    for name, ov in [("default", {}), ("speed3", {"channel.speed_kmh": 3}), ("speed60", {"channel.speed_kmh": 60}),
                     ("snr5", {"link.snr_db": 5})]:
        c = apply_overrides(cfg, ov)
        per = []
        for s in cfg["seeds"]["test"]:
            X, Y, P = gru_sequences(c, [(c, s)], weights_path, window, stride, [0.75], policy_seed=777)
            lg, cl = gru_head(w, gru_forward(w, X.numpy().astype(np.float64)), Y.shape[1])
            rg, ag = regret_of(lg, Y.numpy())
            rp, ap = regret_of(P, Y.numpy())
            g = np.expm1(Y.numpy())
            good = np.take_along_axis(g, lg.argmax(1)[:, None], 1)[:, 0] >= 0.9 * g.max(1)
            conf = 0.5 * (1 + np.tanh(0.5 * cl))
            per.append((rg, ag, rp, ap, float(((conf - good) ** 2).mean()), float(conf.mean()), float(good.mean())))
        a = np.array(per)
        dreg = a[:, 2] - a[:, 0]                          # persistence regret - GRU regret (> 0: GRU better)
        out["predictor"][name] = {
            "gru_regret": float(a[:, 0].mean()), "gru_acc": float(a[:, 1].mean()),
            "persistence_regret": float(a[:, 2].mean()), "persistence_acc": float(a[:, 3].mean()),
            "regret_gain_mean_ci95": paired_ci(dreg), "seeds_gru_better": int((dreg > 0).sum()),
            "conf_brier": float(a[:, 4].mean()), "conf_mean": float(a[:, 5].mean()),
            "good_rate": float(a[:, 6].mean())}
        log(f"  test {name}: GRU regret {a[:, 0].mean():.4f} acc {a[:, 1].mean():.4f} | persistence "
            f"regret {a[:, 2].mean():.4f} acc {a[:, 3].mean():.4f} | gain {paired_ci(dreg)} "
            f"({int((dreg > 0).sum())}/10) | conf Brier {a[:, 4].mean():.4f}")
    log(f"  test codec trained: {out['codec']['trained']}")
    log(f"  test codec dummy:   {out['codec']['dummy']}")
    return out


# ----------------------------------------------------------------------------- main

def export(codec: Codec, gru: Predictor, window: int) -> dict[str, np.ndarray]:
    w = {k: v.detach().numpy() for k, v in codec.state_dict().items()}
    w |= {k: v.detach().numpy() for k, v in gru.state_dict().items()}
    w["window"] = np.array(window)
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs-codec", type=int, default=None)
    ap.add_argument("--epochs-gru", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config()
    mt = cfg["models_train"]
    out = resolve_path(cfg, a.out or mt["out"])
    ec, eg = a.epochs_codec or mt["epochs_codec"], a.epochs_gru or mt["epochs_gru"]
    t0 = time.time()
    tr_pairs, va_pairs = assignments(cfg, "train"), assignments(cfg, "val")
    log(f"configs {len(grid_configs(cfg))}, train pairs {len(tr_pairs)}, val pairs {len(va_pairs)}")

    Xtr, Ytr = codec_data(tr_pairs)
    Xva, Yva = codec_data(va_pairs)
    log(f"codec data: train {len(Xtr)}, val {len(Xva)} ({time.time() - t0:.0f}s)")
    codec, ep_c = train_codec(cfg, Xtr, Ytr, Xva, Yva, ec, int(mt["seed"]))

    # export the codec first: the GRU training histories use the trained codec's numpy decoder
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + "_codec_only.npz")
    window, stride, hidden = int(mt["window"]), int(mt["slot_stride"]), int(mt["gru_hidden"])
    dummy_gru = Predictor(cfg["system"]["B"] + 4, cfg["system"]["B"], hidden)   # placeholder weights
    np.savez(tmp, **export(codec, dummy_gru, window))
    load_weights.cache_clear()
    p0s = cfg["estimator"]["fit_p0"]
    Gtr, GYtr, _ = gru_sequences(cfg, tr_pairs, str(tmp), window, stride, p0s, policy_seed=int(mt["seed"]))
    Gva, GYva, _ = gru_sequences(cfg, va_pairs, str(tmp), window, stride, p0s, policy_seed=int(mt["seed"]) + 1)
    log(f"gru data: train {len(Gtr)}, val {len(Gva)} windows ({time.time() - t0:.0f}s)")
    gru, ep_g = train_gru(Gtr, GYtr, Gva, GYva, hidden, eg, int(mt["seed"]))

    np.savez(out.with_suffix(".npz"), **export(codec, gru, window))
    tmp.unlink()
    load_weights.cache_clear()
    diag = diagnostics(cfg, str(out.with_suffix(".npz")), window, stride)
    meta = {"chosen_epoch_codec": ep_c, "chosen_epoch_gru": ep_g, "window": window, "slot_stride": stride,
            "gru_hidden": hidden, "seed": int(mt["seed"]), "grid": mt["grid"], "train_seeds": cfg["seeds"]["train"],
            "val_seeds": cfg["seeds"]["val"], "test_seeds": cfg["seeds"]["test"], "n_codec_train": len(Xtr),
            "n_gru_train": len(Gtr), "torch": torch.__version__, "seconds": time.time() - t0,
            "test_diagnostics": diag, "log": LOG}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    log(f"wrote {out.with_suffix('.npz')} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
