"""Aggregates runs/*/results.json into mean and spread per variant (runs/summary.json, runs/summary.md)."""
import glob
import json
import os
from collections import defaultdict

import numpy as np

runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join("runs", "*", "results.json")))]
by_var = defaultdict(list)
for r in runs:
    by_var[r["variant"]].append(r)

def stat(vals):
    vals = [v for v in vals if v is not None]
    return {"mean": round(float(np.mean(vals)), 4), "sd": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0,
            "n": len(vals)} if vals else None

summary = {}
for var, rs in sorted(by_var.items()):
    s = {"seeds": sorted(r["seed"] for r in rs), "train_chips": rs[0]["train_chips"],
         "valid": stat([r["best_valid_iou"] for r in rs])}
    for split in ["test", "bolivia"]:
        for sub in ["all", "clean"]:
            s[f"{split}_{sub}"] = stat([r[f"{split}_iou"][sub] for r in rs])
    events = sorted({e for r in rs for e in r["test_per_event"]})
    s["test_events_all"] = {e: stat([r["test_per_event"][e]["all"] for r in rs if e in r["test_per_event"]]) for e in events}
    s["test_events_clean"] = {e: stat([r["test_per_event"][e]["clean"] for r in rs if e in r["test_per_event"]]) for e in events}
    summary[var] = s

with open(os.path.join("runs", "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

fmt = lambda x: "-" if not x else f"{x['mean']:.3f} ± {x['sd']:.3f}"
lines = ["| Variant | Test, all | Test, clean | Bolivia, all | Bolivia, clean |", "| --- | --- | --- | --- | --- |"]
for var, s in summary.items():
    lines.append(f"| {var} | {fmt(s['test_all'])} | {fmt(s['test_clean'])} | {fmt(s['bolivia_all'])} | {fmt(s['bolivia_clean'])} |")
with open(os.path.join("runs", "summary.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print("SUMMARY_DONE")
