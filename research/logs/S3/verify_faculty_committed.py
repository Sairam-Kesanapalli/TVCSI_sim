"""S3 helper: verify the faculty code as COMMITTED in git (HEAD) against the S0 manifest.

On Windows with core.autocrlf=true, git converts LF to CRLF in the working tree, so
research/tools/verify_faculty.py (which hashes the working tree) reports every file as changed
although nothing was edited. This script hashes the committed blobs instead, i.e. exactly what
is pushed. Run from the repository root.
"""
import csv, hashlib, subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
man = root / "research" / "manifests" / "TaskVoCSI_faculty_original.csv"
rows = list(csv.DictReader(open(man, newline="")))
match = mismatch = missing = 0
for r in rows:
    rel = "research/faculty_TaskVoCSI_v2/" + r["path"].replace("TaskVoCSI_v2/", "", 1)
    p = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=root, capture_output=True)
    if p.returncode:
        missing += 1; print("MISSING:", rel); continue
    ok = hashlib.sha256(p.stdout).hexdigest() == r["sha256"] and len(p.stdout) == int(r["bytes"])
    match += ok; mismatch += (not ok)
    if not ok: print("CHANGED:", rel)
print(f"committed blobs: files={len(rows)} match={match} mismatch={mismatch} missing={missing}")
sys.exit(1 if mismatch or missing else 0)
