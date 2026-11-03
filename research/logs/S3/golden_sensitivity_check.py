"""Sanity check: the golden comparison must detect a deliberately broken scheme (in memory only)."""
import json, numpy as np
import tvcsi.schemes as S
from tvcsi.config import ROOT, load_config, resolve_path
from tvcsi.sim import simulate
from tvcsi.tvcsi import TVCSITable
z = np.load(ROOT / "research/logs/S3/golden_pre_S3.npz"); meta = json.loads(str(z["meta"]))
orig = S.periodic_modes
S.periodic_modes = lambda t, K, m, c, b: orig(t + 1, K, m, c, b)   # shift round-robin by one slot
cfg = load_config(); tab = TVCSITable.from_json(resolve_path(cfg, cfg["tvcsi"]["table_path"]))
res = simulate(cfg, 1000, meta["schemes"], table=tab, record_decisions=True)
for n in meta["schemes"]:
    same = np.array_equal(res.decisions[n], z[f"default/1000/{n}/decisions"])
    print(f"{n:24s} identical={same}")
