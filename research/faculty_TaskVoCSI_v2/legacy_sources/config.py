"""Configuration loading and the few quantities derived from it (costs, budget, AR(1) coefficient)."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml
from scipy.special import j0

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
C_LIGHT = 299_792_458.0  # speed of light [m/s]; a physical constant, not a tunable


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load the YAML config and apply dotted-path overrides such as {"system.K": 4}."""
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides or {})


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return cfg


def resolve_path(cfg: dict, rel: str) -> Path:
    """Paths in the config are relative to the repository root, not the working directory."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def feedback_costs(cfg: dict) -> np.ndarray:
    """C_m = d_m * bits_per_value with C_0 = 0 (Eq. 8)."""
    fb = cfg["feedback"]
    return np.array([0] + [d * fb["bits_per_value"] for d in fb["code_len"]], dtype=np.int64)


def feedback_budget(cfg: dict) -> float:
    """C_FB_max = budget_frac * K * C_{M_f} (spec 3.8), the right-hand side of Eq. 11."""
    return float(cfg["feedback"]["budget_frac"] * cfg["system"]["K"] * feedback_costs(cfg)[-1])


def doppler_hz(speed_kmh: float, fc_hz: float) -> float:
    """Maximum Doppler shift f_d = (v / 3.6) * f_c / c."""
    return speed_kmh / 3.6 * fc_hz / C_LIGHT


def ar1_rho(cfg: dict) -> float:
    """Jakes AR(1) coefficient rho = J0(2 pi f_d Ts) for the path gains (spec 3.1)."""
    ch = cfg["channel"]
    return float(j0(2 * np.pi * doppler_hz(ch["speed_kmh"], ch["fc_hz"]) * ch["Ts_s"]))


def db2lin(x):
    return 10.0 ** (np.asarray(x, dtype=float) / 10.0)


def lin2db(x, floor: float = 1e-12):
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), floor))
