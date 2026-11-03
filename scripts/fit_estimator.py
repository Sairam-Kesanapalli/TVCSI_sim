"""Fit (or load from the cache) the frozen S4 quality estimator at the default configuration.

python scripts/fit_estimator.py [--config path]

Training data: exploration policies on the TRAIN seeds over estimator.train_grid; the leaf size is
chosen on the VALIDATION seeds. Writes results/estimators/estimator_<hash>.joblib and a .json
report (validation metrics, chosen hyper-parameter). Test seeds are not used.
"""
import argparse
import json

from tvcsi.config import load_config
from tvcsi.estimator import estimator_path, get_estimator


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    est = get_estimator(cfg, log=lambda m: print(m, flush=True))
    print("estimator:", estimator_path(cfg))
    print(json.dumps({k: v for k, v in est.meta.items() if k != "validation"}, indent=1))


if __name__ == "__main__":
    main()
