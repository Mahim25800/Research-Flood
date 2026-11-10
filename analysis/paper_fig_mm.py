"""MMFlood diagnosis figure: labelled water fraction per event across datasets, and MMFlood per-activation IoU vs water fraction.

Inputs: KS / Sen1Floods11 histograms (water and dry pixel counts per event), mm_activations_used.json (pixel counts per
activation from the px1-mmflood-final log) and mm_per_activation.md (per-activation IoU from the px1-mmflood-activation log,
model-averaged, two decimals).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/Research"
OUT = os.path.join(HERE, "..", "paper", "figures")


def wf_hist(path):
    d = np.load(path)
    evs = sorted({k.split("|")[-1] for k in d.files if k.startswith("none|V0|")})
    return np.array([d[f"none|V0|{e}"][0][1].sum() / d[f"none|V0|{e}"][0].sum() for e in evs])


wf = {"Sen1Floods11": wf_hist(os.path.join(SRC, "runs_loeo", "tta_hist.npz")),
      "Kuro Siwo": wf_hist(os.path.join(SRC, "runs_ks", "ks_hist_m0_v2.npz"))}
acts = json.load(open(os.path.join(HERE, "..", "data", "mm_activations_used.json")))
rows = []
for line in open(os.path.join(HERE, "..", "data", "mm_per_activation.md")):
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    v = {m: tuple(map(float, c[4 + i].split("/"))) for i, m in enumerate(["none", "tent", "adabn", "adabn_phys", "adabn_physa"])}
    a = acts[c[0]]
    rows.append({"act": c[0], "country": c[1], "wf": a["water"] / (a["water"] + a["dry"]), **v})
wf["MMFlood"] = np.array([r["wf"] for r in rows])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.5, "legend.frameon": False, "pdf.fonttype": 42})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1, 1.35]})
rng = np.random.default_rng(0)
cols = {"Sen1Floods11": "#2a78d6", "Kuro Siwo": "#eb6834", "MMFlood": "#1baf7a"}
mk = {"Sen1Floods11": "o", "Kuro Siwo": "s", "MMFlood": "^"}
for i, (k, v) in enumerate(wf.items()):
    ax[0].scatter(100 * v, i + rng.uniform(-0.18, 0.18, len(v)), s=16, color=cols[k], marker=mk[k], edgecolors="#fcfcfb",
                  linewidths=0.5, zorder=3)
    med = 100 * np.median(v)
    ax[0].plot([med, med], [i - 0.3, i + 0.3], color="#0b0b0b", lw=1.4, zorder=4)
    ax[0].text(med, i + 0.36, f"median {med:.1f}%", ha="center", fontsize=6.8, color="#0b0b0b")
ax[0].set_xscale("log")
ax[0].set_xticks([0.5, 1, 2, 5, 10, 20, 50]); ax[0].set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50"])
ax[0].set_yticks(range(3)); ax[0].set_yticklabels(list(wf)); ax[0].set_ylim(-0.5, 2.7)
ax[0].set_xlabel("Labelled flood water per event (% of valid pixels)"); ax[0].grid(axis="y", visible=False)

x = 100 * wf["MMFlood"]
for m, st in [("none", dict(color="#6b6a66", marker="o", label="No adaptation")),
              ("adabn_physa", dict(color="#2a78d6", marker="D", label="Proposed"))]:
    ax[1].scatter(x, [r[m][0] for r in rows], s=18, edgecolors="#fcfcfb", linewidths=0.5, zorder=3, **st)
for r, xi in zip(rows, x):
    ax[1].plot([xi, xi], [r["none"][0], r["adabn_physa"][0]], color="#c9c8c3", lw=0.8, zorder=1)
for r, xi in zip(rows, x):
    if r["act"] in ("EMSR107", "EMSR438", "EMSR514"):
        ax[1].annotate(f"{r['act']} ({r['country']})", (xi, r["adabn_physa"][0]), xytext=(4, 4), textcoords="offset points",
                       fontsize=6.3, color="#52514e")
rho, p = spearmanr(x, [r["adabn_physa"][0] for r in rows])
ax[1].set_xscale("log"); ax[1].set_xticks([0.5, 1, 2, 5, 10]); ax[1].set_xticklabels(["0.5", "1", "2", "5", "10"])
ax[1].set_xlabel("Labelled flood water in the activation (%)"); ax[1].set_ylabel("IoU at threshold 0.5")
ax[1].text(0.02, 0.96, f"Spearman $\\rho$ = {rho:.2f} (p = {p:.0e}), proposed", transform=ax[1].transAxes, fontsize=6.8,
           color="#52514e", va="top")
ax[1].legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=7, markerscale=1.2, handletextpad=0.2)
for a_, t in zip(ax, "ab"):
    a_.text(-0.12 if t == "a" else -0.1, 1.03, f"({t})", transform=a_.transAxes, fontsize=8.5, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_mmflood_water.pdf")); fig.savefig(os.path.join(OUT, "fig_mmflood_water.png"), dpi=200)
res = {k: {"median": float(np.median(v)), "mean": float(np.mean(v)), "n": int(len(v))} for k, v in wf.items()}
res["mmflood_spearman_proposed"] = [float(rho), float(p)]
rho0, p0 = spearmanr(x, [r["none"][0] for r in rows]); res["mmflood_spearman_none"] = [float(rho0), float(p0)]
small = x < 2
res["mmflood_iou_below2pct"] = [int(small.sum()), float(np.mean([r["adabn_physa"][0] for r, s in zip(rows, small) if s]))]
res["mmflood_iou_above5pct"] = [int((x >= 5).sum()), float(np.mean([r["adabn_physa"][0] for r, s in zip(rows, x >= 5) if s]))]
json.dump(res, open(os.path.join(OUT, "water_fraction.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
