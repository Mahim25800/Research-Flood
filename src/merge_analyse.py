"""
Merge the Kaggle runs (fresh histograms) with the PC runs (all earlier methods), check that the
re-computed methods reproduce the PC numbers, and compare the adaptive physics rule with the rest.

  Sen1Floods11: runs_loeo/tta_hist_local.npz (PC) + runs_loeo/tta_hist.npz (Kaggle)
  Kuro Siwo:    runs_ks/ks_hist_m{s}_v2_local.npz (PC) + runs_ks/ks_hist_m{s}_v2.npz (Kaggle), s = 0, 1, 2

Writes out/analysis_physa.md and prints it.
"""
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

from tta_px1 import BINS, compute, curves

RULES = ["default", "crc_0.2", "crc_0.3"]
K05 = BINS // 2


def repro(local, new, label, lines):
    diffs = []
    for k, h in new.items():
        if k in local and k.split("|")[0] in ("none", "adabn_phys"):
            a, b = curves(local[k][0])[2][K05], curves(h[0])[2][K05]
            diffs.append((abs(a - b), k, a, b))
    if diffs:
        diffs.sort(reverse=True)
        lines.append(f"Reproducibility ({label}): {len(diffs)} re-computed histograms, max |IoU difference| at 0.5 = "
                     f"{diffs[0][0]:.4f} ({diffs[0][1]}: PC {diffs[0][2]:.4f}, Kaggle {diffs[0][3]:.4f}); "
                     f"median {np.median([d[0] for d in diffs]):.4f}")


def merged(local, new):
    m = dict(local)
    m.update({k: v for k, v in new.items() if k.split("|")[0] in ("physa", "adabn_physa")})
    return m


def paired(per_a, per_b):
    evs = sorted(set(per_a) & set(per_b))
    d = np.array([per_a[e] - per_b[e] for e in evs])
    try:
        p = wilcoxon(d).pvalue
    except Exception:
        p = float("nan")
    return d.mean(), int((d > 0).sum()), len(d), p


def table(results_list, key, lines, title, events_of):
    """results_list: one compute() output per model; averages IoU per event over models."""
    methods = [m for m in ["none", "tent", "st", "phys", "adabn", "adabn_phys", "physa", "adabn_physa"]
               if all(m in r[key] for r in results_list)]
    lines += [f"### {title}", "", "| Method | " + " | ".join(f"{r} IoU" for r in RULES) + " | 20% rule: mean miss rate | floods meeting 20% |",
              "| --- |" + " --- |" * (len(RULES) + 2)]
    per = {}
    for m in methods:
        cells = []
        for r in RULES:
            per[(m, r)] = {e: np.mean([res[key][m]["events"][e][r]["iou"] for res in results_list]) for e in events_of(results_list[0][key][m])}
            cells.append(f"{np.mean([res[key][m]['summary'][r]['mean_iou'] for res in results_list]):.3f}")
        s = [res[key][m]["summary"]["crc_0.2"] for res in results_list]
        cells += [f"{np.mean([x['mean_fnr'] for x in s]):.3f}", f"{np.mean([x['meeting'] for x in s]):.1f}"]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")
    for a, b in [("adabn_physa", "adabn_phys"), ("physa", "phys"), ("adabn_physa", "tent"), ("adabn_physa", "none")]:
        if (a, "default") in per and (b, "default") in per:
            out = []
            for r in RULES:
                mu, w, n, p = paired(per[(a, r)], per[(b, r)])
                out.append(f"{r}: {mu:+.3f}, {w}/{n} floods, p {p:.3g}")
            lines.append(f"- {a} vs {b}: " + "; ".join(out))
    lines.append("")


def main(runs_loeo="runs_loeo", runs_ks="runs_ks", out="out"):
    os.makedirs(out, exist_ok=True)
    lines = ["# Adaptive physics rule (per-flood radar thresholds + HAND)", ""]
    evs = lambda block: list(block["events"].keys())

    p_loc, p_new = os.path.join(runs_loeo, "tta_hist_local.npz"), os.path.join(runs_loeo, "tta_hist.npz")
    if os.path.exists(p_loc) and os.path.exists(p_new):
        loc, new = dict(np.load(p_loc)), dict(np.load(p_new))
        repro(loc, new, "Sen1Floods11", lines)
        res = compute(merged(loc, new))
        for key in ["V0|all", "V1|all", "V0|clean", "V1|clean"]:
            table([res], key, lines, f"Sen1Floods11 leave-one-flood-out, {key} (11 floods, 1 model per flood)", evs)

    ks_res = []
    for s in range(3):
        p_loc, p_new = os.path.join(runs_ks, f"ks_hist_m{s}_v2_local.npz"), os.path.join(runs_ks, f"ks_hist_m{s}_v2.npz")
        if os.path.exists(p_loc) and os.path.exists(p_new):
            loc, new = dict(np.load(p_loc)), dict(np.load(p_new))
            repro(loc, new, f"Kuro Siwo model {s}", lines)
            mg = merged(loc, new)
            events = sorted({k.split("|")[2] for k in mg if k.count("|") == 2})
            ks_res.append(compute(mg, events))
    if ks_res:
        for key in ["V0|all", "V1|all"]:
            table(ks_res, key, lines, f"Kuro Siwo, {key}, mean over {len(ks_res)} models (26 events)", evs)

    txt = "\n".join(lines)
    with open(os.path.join(out, "analysis_physa.md"), "w") as f:
        f.write(txt)
    print(txt)
    print("ANALYSIS_DONE")


if __name__ == "__main__":
    main(*sys.argv[1:])
