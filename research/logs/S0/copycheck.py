import csv, sys
sys.path.insert(0, sys.argv[1])
from make_manifest import table
from pathlib import Path
def load(p): return {r["path"]: (r["sha256"], int(r["bytes"])) for r in csv.DictReader(open(p, encoding="utf-8"))}
fac = load(sys.argv[2]); orig = load(sys.argv[3])
mt = table(Path(sys.argv[4])); st = table(Path(sys.argv[5]))
fac_mt = {k[len("TaskVoCSI_v2/"):]: v for k, v in fac.items() if k.startswith("TaskVoCSI_v2/")}
orig_st = {k: v for k, v in orig.items() if not k.startswith("Group5_code_faculty_given/") and "egg-info" not in k}
for name, want, got in (("main track copy", fac_mt, mt), ("side track copy", orig_st, st)):
    bad = [k for k in want if got.get(k) != want[k]]; extra = [k for k in got if k not in want]
    print(f"{name}: {len(want)} expected files, {len(bad)} mismatched/missing, {len(extra)} extra -> " + ("IDENTICAL" if not bad and not extra else f"PROBLEM {bad[:5]} {extra[:5]}"))
