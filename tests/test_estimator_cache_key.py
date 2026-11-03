"""Review fix (Codex finding on PR #3): the estimator cache key must change when a file the estimator
was fitted against changes in CONTENT (trained weights, measured quality curves), not only when its
path changes; and it must not depend on the line endings of a text file (Windows checkouts)."""
import shutil

from tvcsi.config import load_config, resolve_path
from tvcsi.estimator import estimator_key


def _final_like(weights: str, curves: str) -> dict:
    return load_config(overrides={"models": "trained", "trained.kwargs": {"weights": weights},
                                  "quality.curves": curves})


def test_key_follows_weight_contents(tmp_path):
    cfg = load_config()
    src_w = resolve_path(cfg, "data/models/s6_models.npz")
    curves = str(resolve_path(cfg, cfg["quality"]["curves"]))
    w = tmp_path / "w.npz"
    shutil.copyfile(src_w, w)
    k1 = estimator_key(_final_like(str(w), curves))
    assert estimator_key(_final_like(str(w), curves)) == k1          # deterministic
    w.write_bytes(w.read_bytes() + b"\0")                             # same path, new contents
    assert estimator_key(_final_like(str(w), curves)) != k1


def test_key_follows_curve_contents_not_line_endings(tmp_path):
    cfg = load_config()
    src = resolve_path(cfg, cfg["quality"]["curves"]).read_bytes().replace(b"\r\n", b"\n")
    c = tmp_path / "q.csv"
    c.write_bytes(src)
    key = lambda: estimator_key(load_config(overrides={"quality.curves": str(c)}))  # noqa: E731
    k_lf = key()
    c.write_bytes(src.replace(b"\n", b"\r\n"))                       # Windows line endings only
    assert key() == k_lf
    c.write_bytes(src + b"x\n")                                       # real change
    assert key() != k_lf


def test_dummy_key_ignores_trained_section():
    """With models: dummy the trained weights are not used, so they must not enter the key."""
    a = load_config()
    b = load_config(overrides={"trained.kwargs": {"weights": "does/not/exist.npz"}})
    assert estimator_key(a) == estimator_key(b)
