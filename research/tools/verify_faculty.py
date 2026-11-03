"""Check that the faculty code in research/faculty_TaskVoCSI_v2/ is byte-identical to the S0 manifest.

Run from the repository root:  python research/tools/verify_faculty.py
Expected: files=23 match=23 mismatch=0 missing=0 (extra = run outputs under results/, which is fine).
Exit code 1 if any original file is changed or missing.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_manifest import table  # noqa: E402

research = Path(__file__).resolve().parents[1]
current = {k: (v if isinstance(v, tuple) else tuple(v)) for k, v in table(research / "faculty_TaskVoCSI_v2").items()}
with open(research / "manifests" / "TaskVoCSI_faculty_original.csv", newline="") as f:
    original = {r["path"].replace("TaskVoCSI_v2/", "", 1): (r["sha256"], int(r["bytes"])) for r in csv.DictReader(f)}

match = [k for k in original if current.get(k) == original[k]]
mismatch = [k for k in original if k in current and current[k] != original[k]]
missing = [k for k in original if k not in current]
extra = [k for k in current if k not in original]
print(f"files={len(original)} match={len(match)} mismatch={len(mismatch)} missing={len(missing)} extra={len(extra)}")
for k in mismatch:
    print("CHANGED:", k)
for k in missing:
    print("MISSING:", k)
sys.exit(1 if mismatch or missing else 0)
