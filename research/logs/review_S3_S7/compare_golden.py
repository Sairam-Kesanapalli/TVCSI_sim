import sys, json, numpy as np
a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
keys = [k for k in a.files if k != "meta"]
diff = [k for k in keys if not np.array_equal(a[k], b[k], equal_nan=True)]
print(f"{len(keys)} arrays, {len(diff)} differ")
for k in diff:
    x, y = a[k], b[k]
    if k.endswith("decisions"):
        print(" ", k, "slots differing:", int((x != y).any(axis=1).sum()), "of", len(x))
    else:
        print(" ", k, "max |diff|:", float(np.nanmax(np.abs(x - y))))
