"""Per-activation detail table for the MMFlood final test (IoU averaged over the three models).

Usage: python mm_detail.py runs_mm_act data_mm/index.csv out_act/detail.md
"""
import csv
import glob
import os
import sys

import numpy as np

from tta_px1 import compute

METHODS = ["none", "tent", "adabn", "adabn_phys", "adabn_physa"]

runs, index, out = sys.argv[1:4]
country, folders, chips = {}, {}, {}
for r in csv.DictReader(open(index)):
    a = r["event"].split("-")[0]
    country[a] = r["country"]
    folders.setdefault(a, set()).add(r["event"])
    chips[a] = chips.get(a, 0) + 1
res = []
for p in sorted(glob.glob(os.path.join(runs, "mm_hist_m*_p0.npz"))):
    d = dict(np.load(p))
    res.append(compute(d, sorted({k.split("|")[-1] for k in d if k.count("|") == 2}))["V0|all"])
evs = sorted(set.intersection(*[set(r[m]["events"]) for r in res for m in METHODS]))
L = [f"Activations analysed: {len(evs)} (of {len(folders)} with chips); models: {len(res)}", "",
     "| Activation | Country | Folders | Chips | " + " | ".join(f"{m} 0.5 / 20%" for m in METHODS) + " |",
     "| --- " * (4 + len(METHODS)) + "|"]
for e in evs:
    cells = []
    for m in METHODS:
        d0 = np.mean([r[m]["events"][e]["default"]["iou"] for r in res])
        d2 = np.mean([r[m]["events"][e]["crc_0.2"]["iou"] for r in res])
        cells.append(f"{d0:.2f} / {d2:.2f}")
    L.append(f"| {e} | {country.get(e)} | {len(folders.get(e, []))} | {chips.get(e, 0)} | " + " | ".join(cells) + " |")
L.append("")
for m in METHODS:
    f = [np.array([r[m]["events"][e]["crc_0.2"]["fnr"] for e in evs]) for r in res]
    L.append(f"{m}: 20% guarantee, per-activation miss rate mean {np.mean([x.mean() for x in f]):.3f}, "
             f"activations meeting it {np.mean([(x <= 0.2).sum() for x in f]):.1f}/{len(evs)}, worst {np.mean([x.max() for x in f]):.3f}")
txt = "\n".join(L)
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
open(out, "w").write(txt)
print(txt)
print("MM_DETAIL_DONE")
