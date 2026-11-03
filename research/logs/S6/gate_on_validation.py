"""S6 gate re-check on the VALIDATION seeds (review patch, 2026-11-03).

Why: the S6 decisions "GRU or persistence?" and "trained or dummy codec?" were first taken from the
diagnostics on the TEST seeds (scripts/train_models.py:diagnostics). Model choices must be made on
validation seeds; test seeds are for reporting only. This script repeats exactly the same
diagnostics on the validation seeds 2000-2004 with the committed weights (no training) so the
decisions can be checked on the proper split.

Run from the repository root (needs torch, like scripts/train_models.py):
    <venv>/Scripts/python research/logs/S6/gate_on_validation.py
Writes research/logs/S6/gate_on_validation.json. Note: the per-line output of
train_models.diagnostics() says "test" and ".../10"; here the rows are the 5 validation seeds.
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import train_models as tm  # noqa: E402  (imports torch first, as that script requires)


def main() -> None:
    cfg = tm.load_config()
    mt = cfg["models_train"]
    val = copy.deepcopy(cfg)
    val["seeds"]["test"] = list(cfg["seeds"]["val"])          # diagnostics() iterates seeds["test"]
    n = len(val["seeds"]["test"])
    print(f"validation seeds: {val['seeds']['test']}", flush=True)
    diag = tm.diagnostics(val, str(ROOT / "data" / "models" / "s6_models.npz"), int(mt["window"]), int(mt["slot_stride"]))

    print("\n=== S6 gate on VALIDATION seeds ===")
    for name, d in diag["predictor"].items():
        mean, ci = d["regret_gain_mean_ci95"]
        sig = "GRU significantly better" if mean - ci > 0 else "not significant"
        print(f"{name:8s} persistence - GRU regret = {mean:+.4f} +/- {ci:.4f}  "
              f"(GRU better in {d['seeds_gru_better']}/{n} seeds): {sig}")
    gate = diag["predictor"]["default"]
    gru_passes = gate["regret_gain_mean_ci95"][0] - gate["regret_gain_mean_ci95"][1] > 0
    print(f"gate (default point): {'GRU passes' if gru_passes else 'GRU fails -> keep persistence'}")
    tr, du = diag["codec"]["trained"]["regret_per_mode"], diag["codec"]["dummy"]["regret_per_mode"]
    print(f"codec regret per mode (32/64/128 bit): trained {tr}  dummy {du}")
    Path(__file__).with_suffix(".json").write_text(json.dumps(
        {"seeds": val["seeds"]["test"], "gru_passes_gate": bool(gru_passes), "diagnostics": diag}, indent=1))


if __name__ == "__main__":
    main()
