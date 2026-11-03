"""SHA-256 manifest of a directory tree, and verification against a saved manifest.

    python make_manifest.py write  <root> <manifest.csv>   # record every file's hash
    python make_manifest.py verify <root> <manifest.csv>   # exit 1 if anything changed/appeared/vanished

Python bytecode caches (__pycache__, *.pyc) are skipped: Python regenerates them by itself.
"""
import csv
import hashlib
import sys
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and not (set(p.relative_to(root).parts) & SKIP_DIRS) and p.suffix != ".pyc":
            yield p


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def table(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): (digest(p), p.stat().st_size) for p in files(root)}


def main():
    mode, root, manifest = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    if mode == "write":
        t = table(root)
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "sha256", "bytes"])
            for k, (h, n) in t.items():
                w.writerow([k, h, n])
        print(f"{len(t)} files, {sum(n for _, n in t.values()) / 1e6:.1f} MB -> {manifest}")
    elif mode == "verify":
        with open(manifest, encoding="utf-8") as f:
            saved = {r["path"]: (r["sha256"], int(r["bytes"])) for r in csv.DictReader(f)}
        now = table(root)
        changed = [k for k in saved if k in now and now[k] != saved[k]]
        missing = [k for k in saved if k not in now]
        added = [k for k in now if k not in saved]
        for name, lst in (("CHANGED", changed), ("MISSING", missing), ("ADDED", added)):
            for k in lst[:20]:
                print(f"{name}: {k}")
        ok = not (changed or missing or added)
        print(f"verify {root}: {len(saved)} files in manifest -> " + ("UNCHANGED" if ok else
              f"{len(changed)} changed, {len(missing)} missing, {len(added)} added"))
        sys.exit(0 if ok else 1)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
