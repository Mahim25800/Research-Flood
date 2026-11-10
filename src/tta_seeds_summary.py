"""
Pools every flood-adaptation run in runs_loeo/tta_hist*.npz (different model seeds and
adaptation seeds) and asks whether physics-checked self-training beats no adaptation and
plain self-training consistently.

For each setting (variant, chip subset, threshold rule):
  per-run mean IoU over floods  -> mean and SD across runs, and in how many runs phys wins
  per-flood IoU averaged over runs -> paired Wilcoxon test across the 11 floods
Writes runs_loeo/tta_seeds_summary.md and .json.
"""
import glob
import json
import os

import numpy as np

from tta_px1 import METHODS, compute

RULES = ["default", "crc_0.2", "crc_0.3"]
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


def wil(d):
    if wilcoxon is None or len(d) < 5 or not np.any(d):
        return None
    return float(wilcoxon(d).pvalue)


runs = {}
for p in sorted(glob.glob(os.path.join("runs_loeo", "tta_hist*.npz"))):
    if ".tmp" in p:
        continue
    name = os.path.basename(p)[len("tta_hist"):-4] or "_m0_a0"
    runs[name] = compute(dict(np.load(p)))
print("runs:", ", ".join(runs), flush=True)

out, lines = {"runs": list(runs)}, [f"Runs pooled: {', '.join(runs)}", ""]
for key in ["V0|all", "V1|all", "V0|clean", "V1|clean"]:
    lines += [f"### {key}", "",
              "| Rule | " + " | ".join(f"{m} mean IoU (SD over runs)" for m in METHODS) +
              " | phys > none, runs | phys vs none, floods (p) | phys vs st, floods (p) | floods meeting target, none / phys |",
              "| --- |" + " --- |" * (len(METHODS) + 4)]
    out[key] = {}
    for rule in RULES:
        res = {}
        okruns = [r for r in runs.values() if all(m in r.get(key, {}) for m in METHODS)]
        if not okruns:
            continue
        events = list(okruns[0][key]["none"]["events"])
        iou = {m: np.array([[r[key][m]["events"][e][rule]["iou"] for e in events] for r in okruns]) for m in METHODS}
        means = {m: iou[m].mean(1) for m in METHODS}
        flood = {m: iou[m].mean(0) for m in METHODS}
        d_none, d_st = flood["phys"] - flood["none"], flood["phys"] - flood["st"]
        meet = {m: [r[key][m]["summary"][rule]["meeting"] for r in okruns] for m in ["none", "phys"]}
        res = {"n_runs": len(okruns),
               "mean_iou": {m: [round(float(means[m].mean()), 4), round(float(means[m].std()), 4)] for m in METHODS},
               "runs_phys_beats_none": int((means["phys"] > means["none"]).sum()),
               "phys_vs_none": {"mean": round(float(d_none.mean()), 4), "wins": int((d_none > 0).sum()), "p": wil(d_none)},
               "phys_vs_st": {"mean": round(float(d_st.mean()), 4), "wins": int((d_st > 0).sum()), "p": wil(d_st)},
               "meeting": meet,
               "per_flood_iou": {m: dict(zip(events, np.round(flood[m], 4).tolist())) for m in METHODS}}
        out[key][rule] = res
        f = lambda d: f"{d['mean']:+.3f}, {d['wins']}/{len(events)} (p {'-' if d['p'] is None else format(d['p'], '.3f')})"
        mt = "-" if rule == "default" else f"{np.mean(meet['none']):.1f} / {np.mean(meet['phys']):.1f}"
        lines.append(f"| {rule} | " + " | ".join(f"{res['mean_iou'][m][0]:.3f} ({res['mean_iou'][m][1]:.3f})" for m in METHODS) +
                     f" | {res['runs_phys_beats_none']}/{len(okruns)} | {f(res['phys_vs_none'])} | {f(res['phys_vs_st'])} | {mt} |")
    lines.append("")

with open(os.path.join("runs_loeo", "tta_seeds_summary.json"), "w") as fh:
    json.dump(out, fh, indent=1)
with open(os.path.join("runs_loeo", "tta_seeds_summary.md"), "w") as fh:
    fh.write("\n".join(lines))
print("\n".join(lines))
print("SEEDS_SUMMARY_DONE")
