"""
Pre-registered MMFlood analysis (see the PX1 doc, 9 Nov 2026 01:05 BDT).

Unit: event, IoU averaged over the three models (V0, radar only). One-sided Wilcoxon, alpha 0.05, Holm over H1-H3.
  H1 primary > TENT under the 20% guarantee
  H2 primary > AdaBN under the 20% guarantee
  H3 primary > no adaptation at threshold 0.5
  H4 mean miss rate under the 20% guarantee <= 0.20 (primary)

Usage: python mm_analyse.py --primary adabn_physa --runs runs_mm --mm data_mm --out out
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

from tta_px1 import compute

METHODS = ["none", "tent", "adabn", "adabn_phys", "adabn_physa"]
RULES = ["default", "crc_0.1", "crc_0.2", "crc_0.3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True, choices=["adabn_physa", "adabn_phys"])
    ap.add_argument("--runs", default="runs_mm")
    ap.add_argument("--mm", default="data_mm")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = []
    for s in range(3):
        parts = sorted(glob.glob(os.path.join(a.runs, f"mm_hist_m{s}_p*.npz")))
        if not parts:
            continue
        data = {}
        for p in parts:
            data.update(dict(np.load(p)))
        events = sorted({k.split("|")[2] for k in data if k.count("|") == 2})
        res.append(compute(data, events)["V0|all"])
    secondary = "adabn_phys" if a.primary == "adabn_physa" else "adabn_physa"
    L = ["# MMFlood final test (pre-registered)", "",
         f"Primary method: {a.primary} (chosen by the pre-registered rule from the Kuro Siwo results). Secondary: {secondary}.", ""]
    ex = os.path.join(a.mm, "excluded.csv")
    if os.path.exists(ex):
        rows = list(csv.DictReader(open(ex)))
        L.append(f"Excluded for overlap with Kuro Siwo or Sen1Floods11: {len(rows)} events ({', '.join(r['event'] for r in rows)}).")
    methods = [m for m in METHODS if all(m in r for r in res)]
    evs = sorted(set.intersection(*[set(r[m]["events"]) for r in res for m in methods]))
    L += [f"Models: {len(res)}; events with labelled water analysed: {len(evs)}.", "",
          "| Method | " + " | ".join(f"{r} IoU" for r in RULES) + " | 20%: mean miss rate | 20%: events meeting |",
          "| --- |" + " --- |" * (len(RULES) + 2)]
    per = {}
    for m in methods:
        for r in RULES:
            per[(m, r)] = np.array([np.mean([x[m]["events"][e][r]["iou"] for x in res]) for e in evs])
        fn = np.mean([x[m]["summary"]["crc_0.2"]["mean_fnr"] for x in res])
        meet = np.mean([x[m]["summary"]["crc_0.2"]["meeting"] for x in res])
        L.append(f"| {m} | " + " | ".join(f"{per[(m, r)].mean():.3f}" for r in RULES) + f" | {fn:.3f} | {meet:.1f}/{len(evs)} |")
    L.append("")
    tests = [("H1", "tent", "crc_0.2"), ("H2", "adabn", "crc_0.2"), ("H3", "none", "default")]
    ps, out = [], {}
    for h, ref, r in tests:
        d = per[(a.primary, r)] - per[(ref, r)]
        p = float(wilcoxon(d, alternative="greater").pvalue) if np.any(d != 0) else 1.0
        ps.append(p)
        out[h] = {"ref": ref, "rule": r, "mean_gain": float(d.mean()), "wins": int((d > 0).sum()), "n": len(d), "p": p}
    order = np.argsort(ps)
    adj, running = [0.0] * 3, 0.0
    for rank, i in enumerate(order):  # Holm step-down
        running = max(running, min(1.0, (3 - rank) * ps[i]))
        adj[i] = running
    for (h, ref, r), pa in zip(tests, adj):
        o = out[h]
        o["p_holm"], o["supported"] = pa, bool(pa < 0.05 and o["mean_gain"] > 0)
        L.append(f"- {h}: {a.primary} vs {ref} ({r}): mean gain {o['mean_gain']:+.3f}, better on {o['wins']}/{o['n']} events, "
                 f"p {o['p']:.3g}, Holm p {pa:.3g} -> {'SUPPORTED' if o['supported'] else 'not supported'}")
    fn = float(np.mean([x[a.primary]["summary"]["crc_0.2"]["mean_fnr"] for x in res]))
    out["H4"] = {"mean_miss_rate": fn, "supported": fn <= 0.20}
    L.append(f"- H4: mean miss rate under the 20% guarantee {fn:.3f} -> {'SUPPORTED' if fn <= 0.20 else 'not supported'}")
    d = per[(a.primary, "crc_0.2")] - per[(secondary, "crc_0.2")]
    L.append(f"- (not tested) {a.primary} vs {secondary} under the 20% guarantee: {d.mean():+.3f}, better on {int((d > 0).sum())}/{len(d)}")
    txt = "\n".join(L)
    open(os.path.join(a.out, "analysis_mmflood.md"), "w").write(txt)
    json.dump(out, open(os.path.join(a.out, "analysis_mmflood.json"), "w"), indent=1)
    print(txt)
    print("MM_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
