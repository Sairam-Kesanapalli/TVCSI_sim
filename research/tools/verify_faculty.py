"""Check that the faculty code in research/faculty_TaskVoCSI_v2/ is identical to the S0 manifest.

Run from anywhere inside the repository:  python research/tools/verify_faculty.py
Expected:
    committed (git HEAD): files=23 match=23 mismatch=0 missing=0
    working tree:         files=23 match=23 mismatch=0 missing=0 (k need line-ending normalisation)
Exit code 1 if any original file is changed or missing.

Two checks:
1. **Committed** (authoritative): the files exactly as stored in git at HEAD, i.e. what is pushed
   and what everyone else checks out. Needs git.
2. **Working tree**: the files on this disk. On Windows with core.autocrlf=true, git writes the
   files with CRLF line endings although the stored content has LF, so a raw hash differs for every
   file although nothing was edited. Such a file counts as a match if it equals the manifest after
   converting CRLF to LF, and it is reported as "line-ending normalisation" (harmless).
Run outputs under results/ are not in the manifest and are ignored.
"""
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
REPO = RESEARCH.parent
FACULTY = "research/faculty_TaskVoCSI_v2"


def manifest() -> dict[str, tuple[str, int]]:
    with open(RESEARCH / "manifests" / "TaskVoCSI_faculty_original.csv", newline="") as f:
        return {r["path"].replace("TaskVoCSI_v2/", "", 1): (r["sha256"], int(r["bytes"]))
                for r in csv.DictReader(f)}


def matches(data: bytes, ref: tuple[str, int]) -> bool:
    return len(data) == ref[1] and hashlib.sha256(data).hexdigest() == ref[0]


def committed(man: dict) -> tuple[list, list, list] | None:
    match, mismatch, missing = [], [], []
    for rel, ref in man.items():
        p = subprocess.run(["git", "show", f"HEAD:{FACULTY}/{rel}"], cwd=REPO, capture_output=True)
        if p.returncode:
            if b"not a git repository" in p.stderr.lower():
                return None
            missing.append(rel)
        else:
            (match if matches(p.stdout, ref) else mismatch).append(rel)
    return match, mismatch, missing


def working_tree(man: dict) -> tuple[list, list, list, list]:
    match, normalised, mismatch, missing = [], [], [], []
    for rel, ref in man.items():
        f = REPO / FACULTY / rel
        if not f.is_file():
            missing.append(rel)
            continue
        data = f.read_bytes()
        if matches(data, ref):
            match.append(rel)
        elif matches(data.replace(b"\r\n", b"\n"), ref):
            normalised.append(rel)
        else:
            mismatch.append(rel)
    return match, normalised, mismatch, missing


def main() -> int:
    man = manifest()
    bad = False
    try:
        c = committed(man)
    except FileNotFoundError:          # git not installed
        c = None
    if c is None:
        print("committed (git HEAD): skipped (no git repository / git not found)")
    else:
        match, mismatch, missing = c
        print(f"committed (git HEAD): files={len(man)} match={len(match)} mismatch={len(mismatch)} missing={len(missing)}")
        for k in mismatch:
            print("  CHANGED in git:", k)
        for k in missing:
            print("  MISSING in git:", k)
        bad |= bool(mismatch or missing)
    match, normalised, mismatch, missing = working_tree(man)
    print(f"working tree:         files={len(man)} match={len(match) + len(normalised)} mismatch={len(mismatch)} "
          f"missing={len(missing)} ({len(normalised)} need line-ending normalisation)")
    for k in mismatch:
        print("  CHANGED on disk:", k)
    for k in missing:
        print("  MISSING on disk:", k)
    bad |= bool(mismatch or missing)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
