"""
Analysis for the second backbone (exploratory: same tests as the pre-registered MMFlood analysis, not pre-registered).

Pools histogram files per model seed (several parts), optionally merges MMFlood region folders into activations,
then reports IoU per rule and the primary-vs-reference tests (one-sided Wilcoxon, Holm over three tests).

Usage: python bb_analyse.py --hist "runs_ks_r34_p*/ks_hist_m{s}_v2.npz" --title "Kuro Siwo, ResNet-34 U-Net" --out out/ks_r34.md
       python bb_analyse.py --hist "runs_mm_r34/mm_hist_m{s}_p*.npz" --activation --title "MMFlood, ResNet-34 U-Net" --out out/mm_r34.md
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

from tta_px1 import compute

RULES = ["default", "crc_0.1", "crc_0.2", "crc_0.3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", required=True, help="glob with {s} for the model seed")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--methods", default="none,tent,adabn,adabn_physa")
    ap.add_argument("--primary", default="adabn_physa")
    ap.add_argument("--variant", default="V0")
    ap.add_argument("--activation", action="store_true", help="merge MMFlood region folders (EMSRxxx-N) into activations")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    methods = a.methods.split(",")
    res = []
    for s in a.seeds.split(","):
        files = sorted(glob.glob(a.hist.format(s=s)))
        if not files:
            print(f"no files for seed {s}", flush=True)
            continue
        data = {}
        for p in files:
            for k, v in np.load(p).items():
                if a.activation:
                    parts = k.split("|"); parts[-1] = parts[-1].split("-")[0]; k = "|".join(parts)
                data[k] = data[k] + v if k in data else v.copy()
        events = sorted({k.split("|")[-1] for k in data if k.count("|") == 2})
        res.append(compute(data, events)[f"{a.variant}|all"])
    methods = [m for m in methods if all(m in r for r in res)]
    evs = sorted(set.intersection(*[set(r[m]["events"]) for r in res for m in methods]))
    unit = "activations" if a.activation else "events"
    L = [f"## {a.title}", "", f"Models: {len(res)}; {unit} analysed: {len(evs)}. Exploratory (not pre-registered).", "",
         "| Method | " + " | ".join(f"{r} IoU" for r in RULES) + f" | 20%: mean miss rate | 20%: {unit} meeting |",
         "| --- |" + " --- |" * (len(RULES) + 2)]
    per = {}
    for m in methods:
        for r in RULES:
            per[m, r] = np.array([np.mean([x[m]["events"][e][r]["iou"] for x in res]) for e in evs])
        fnr = [np.array([x[m]["events"][e]["crc_0.2"]["fnr"] for e in evs]) for x in res]
        L.append(f"| {m} | " + " | ".join(f"{per[m, r].mean():.3f}" for r in RULES)
                 + f" | {np.mean([f.mean() for f in fnr]):.3f} | {np.mean([(f <= 0.2).sum() for f in fnr]):.1f}/{len(evs)} |")
    L.append("")
    tests = [(ref, r) for ref, r in [("tent", "crc_0.2"), ("adabn", "crc_0.2"), ("none", "default")] if ref in methods]
    out, ps = {}, []
    for ref, r in tests:
        d = per[a.primary, r] - per[ref, r]
        p = float(wilcoxon(d, alternative="greater").pvalue) if np.any(d != 0) else 1.0
        ps.append(p)
        out[f"{ref}|{r}"] = {"mean_gain": float(d.mean()), "wins": int((d > 0).sum()), "n": len(d), "p": p}
    running = 0.0
    for rank, i in enumerate(np.argsort(ps)):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[f"{tests[i][0]}|{tests[i][1]}"]["p_holm"] = running
    for (ref, r) in tests:
        o = out[f"{ref}|{r}"]
        L.append(f"- {a.primary} vs {ref} ({r}): mean gain {o['mean_gain']:+.3f}, better on {o['wins']}/{o['n']}, "
                 f"p {o['p']:.3g}, Holm p {o['p_holm']:.3g}")
    txt = "\n".join(L)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write(txt)
    json.dump({"tests": out, "table": {f"{m}|{r}": float(per[m, r].mean()) for m in methods for r in RULES}},
              open(a.out[:-3] + ".json", "w"), indent=1)
    print(txt)
    print("BB_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
