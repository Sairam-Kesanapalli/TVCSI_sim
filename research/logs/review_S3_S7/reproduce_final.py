import pandas as pd
from tvcsi.config import load_config
from tvcsi.estimator import get_estimator
from tvcsi.sim import simulate
from tvcsi.sweep import get_table
def main():
    cfg = load_config("configs/final.yaml")
    est = get_estimator(cfg)
    schemes = ["Exact", "Proposed", "Proposed_risk", "Baseline", "fill_importance", "fill_age"]
    ref = pd.read_csv("results/final_20261103/default_runs.csv").set_index(["seed", "scheme"])
    for seed in (1000, 1005):
        m = simulate(cfg, seed, schemes, table=get_table(cfg), estimator=est).metrics
        for s in schemes:
            a, b = m[s]["utility"], ref.loc[(seed, s), "utility"]
            print(f"seed {seed} {s:16s} now {a:.12f}  committed {b:.12f}  diff {a-b:+.1e}  bits {m[s]['bits']:.1f}/{ref.loc[(seed, s), 'bits']:.1f}", flush=True)
    

if __name__ == '__main__':
    main()
