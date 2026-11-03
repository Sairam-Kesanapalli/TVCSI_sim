"""Review: refit the default and final estimators under the new (content-hashed) cache names and
compare with the reports of the old cache names (must be identical apart from the fit time)."""
import json
from pathlib import Path
from tvcsi.config import load_config
from tvcsi.estimator import estimator_path, get_estimator

OLD = {"configs/default.yaml": "8d2408872161", "configs/final.yaml": "266dd335e39c"}


def main():
    for c, old in OLD.items():
        cfg = load_config(c)
        est = get_estimator(cfg, log=lambda *a: None)
        new = json.loads(estimator_path(cfg).with_suffix(".json").read_text())
        ref = json.loads((estimator_path(cfg).parent / f"estimator_{old}.json").read_text())
        same = {k: new[k] == ref[k] for k in ref if k != "fit_seconds"}
        print(f"{c}: {estimator_path(cfg).name} vs estimator_{old}.json -> identical: {all(same.values())}",
              {k: v for k, v in same.items() if not v} or "")


if __name__ == "__main__":
    main()
