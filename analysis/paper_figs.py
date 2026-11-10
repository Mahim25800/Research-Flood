"""
Paper figures and extra numbers from the stored probability histograms (no model runs).

Inputs (copied from F:\\Coding\\Research):
  runs_ks/ks_hist_m{0,1,2}_v2.npz   Kuro Siwo, 26 events, three Sen1Floods11 models, radar-only variant V0
  runs_loeo/tta_hist.npz            Sen1Floods11 leave-one-flood-out, 11 floods, V0
Histogram layout: h[sub] with sub 0 = all chips; h[sub][0] = dry pixels, h[sub][1] = water pixels,
1000 bins of predicted water probability. Thresholds and the guarantee follow tta_px1.compute exactly.

Outputs: paper_latex/figures/fig_guarantee.pdf, fig_ks_events.pdf, figures/paper_numbers.json
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from conformal_px1 import curves  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/Research"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)

KS = [dict(np.load(os.path.join(SRC, "runs_ks", f"ks_hist_m{s}_v2.npz"))) for s in range(3)]
S1F = [dict(np.load(os.path.join(SRC, "runs_loeo", "tta_hist.npz")))]
V = "V0"
METHODS = ["none", "tent", "st", "phys", "adabn", "adabn_phys"]
LABEL = {"none": "No adaptation", "tent": "TENT", "st": "Self-training", "phys": "Physics-checked ST (fixed)",
         "adabn": "AdaBN", "adabn_phys": "AdaBN + physics (fixed rule)"}


def events(d):
    return sorted({k.split("|")[-1] for k in d if k.startswith(f"none|{V}|")})


def per_event(d, m, alpha=None):
    """Per event: (iou, fnr, precision, false-alarm rate) at 0.5 (alpha None) or under the LOO guarantee."""
    evs = [e for e in events(d) if d[f"{m}|{V}|{e}"][0, 1].sum() > 0]
    cur = {e: curves(d[f"{m}|{V}|{e}"][0]) for e in evs}
    dry = {e: d[f"{m}|{V}|{e}"][0][0].astype(float) for e in evs}
    out = {}
    for e in evs:
        if alpha is None:
            k = 500
        else:
            others = np.mean([cur[o][0] for o in evs if o != e], axis=0)
            n = len(evs) - 1
            ok = np.where(n / (n + 1) * others + 1 / (n + 1) <= alpha)[0]
            k = int(ok.max()) if len(ok) else 0
        fnr, prec, iou = cur[e]
        fp = dry[e][k:].sum()
        far = fp / max(dry[e].sum(), 1)
        out[e] = np.array([iou[k], fnr[k], prec[k], far])
    return out


def model_mean(D, m, alpha=None):
    """Average over models per event -> dict event -> [iou, fnr, prec, far]."""
    per = [per_event(d, m, alpha) for d in D]
    evs = sorted(set.intersection(*[set(p) for p in per]))
    return {e: np.mean([p[e] for p in per], axis=0) for e in evs}


numbers = {}
# ------------------------------------------------------------------ tables
for name, D in [("kuro_siwo", KS), ("sen1floods11", S1F)]:
    tab = {}
    for m in METHODS:
        if f"{m}|{V}|{events(D[0])[0]}" not in D[0]:
            continue
        row = {}
        for rule, a in [("t0.5", None), ("crc_0.1", 0.1), ("crc_0.2", 0.2), ("crc_0.3", 0.3)]:
            pe = model_mean(D, m, a)
            arr = np.array(list(pe.values()))
            row[rule] = {"iou": arr[:, 0].mean(), "fnr": arr[:, 1].mean(), "precision": arr[:, 2].mean(),
                         "false_alarm": arr[:, 3].mean(), "meeting": None if a is None else float(np.mean(
                             [np.sum(np.array(list(per_event(d, m, a).values()))[:, 1] <= a) for d in D]))}
        tab[m] = row
    numbers[name] = tab
    if name == "kuro_siwo":
        ot = [np.array([d[f"otsu|{e}"][0] for e in events(d)]) for d in D[:1]][0]  # tp, fp, fn per event
        prec = ot[:, 0] / np.maximum(ot[:, 0] + ot[:, 1], 1)
        numbers[name]["otsu"] = {"t0.5": {"iou": float(np.mean(ot[:, 0] / ot.sum(1))), "precision": float(prec.mean()),
                                          "fnr": float(np.mean(ot[:, 2] / np.maximum(ot[:, 0] + ot[:, 2], 1)))}}


def fmt(x):
    return round(float(x), 4) if isinstance(x, (float, np.floating)) else x


json.dump(json.loads(json.dumps(numbers, default=fmt)), open(os.path.join(OUT, "paper_numbers.json"), "w"), indent=1)

# ------------------------------------------------------------------ style
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
                     "legend.frameon": False, "pdf.fonttype": 42})
STYLE = {"none": dict(color="#6b6a66", ls="--", marker="o"),
         "tent": dict(color="#eb6834", ls="-", marker="s"),
         "adabn": dict(color="#1baf7a", ls="-", marker="^"),
         "adabn_phys": dict(color="#2a78d6", ls="-", marker="D")}
SHOW = ["none", "tent", "adabn", "adabn_phys"]
ALPHAS = np.round(np.arange(0.05, 0.501, 0.025), 3)

# ------------------------------------------------------------------ figure: guarantee curves
fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
curves_out = {}
for j, (name, D, n_ev) in enumerate([("Kuro Siwo (26 events, 3 models)", KS, 26), ("Sen1Floods11 (11 floods, leave one out)", S1F, 11)]):
    for m in SHOW:
        iou, fnr = [], []
        for a in ALPHAS:
            arr = np.array(list(model_mean(D, m, float(a)).values()))
            iou.append(arr[:, 0].mean()); fnr.append(arr[:, 1].mean())
        curves_out[f"{j}|{m}"] = {"alpha": ALPHAS.tolist(), "iou": iou, "fnr": fnr}
        st = STYLE[m]
        ax[0, j].plot(ALPHAS, iou, color=st["color"], ls=st["ls"], lw=1.6, marker=st["marker"], ms=3.5, markevery=4,
                      label=LABEL[m])
        ax[1, j].plot(ALPHAS, fnr, color=st["color"], ls=st["ls"], lw=1.6, marker=st["marker"], ms=3.5, markevery=4)
    ax[1, j].plot([0, 0.55], [0, 0.55], color="#0b0b0b", lw=0.8, ls=":", zorder=1)
    ax[1, j].text(0.40, 0.445, "target", rotation=0, fontsize=7, color="#52514e")
    ax[0, j].set_title(name, fontsize=8.5, loc="left", color="#0b0b0b")
    ax[0, j].set_ylim(0.2, 0.75) if j == 0 else ax[0, j].set_ylim(0.1, 0.65)
    amin = 1 / n_ev  # n = n_ev - 1 calibration floods, so 1/(n+1) = 1/n_ev
    for r in (0, 1):
        ax[r, j].axvspan(0.04, amin, color="#f0efec", zorder=0, lw=0)
    if amin > 0.04:
        ax[0, j].text(amin + 0.005, 0.12, r"$\alpha<1/(n+1)$: no informative threshold", fontsize=6.5, color="#52514e")
    ax[1, j].set_ylim(0, 0.55); ax[1, j].set_xlim(0.04, 0.51)
    ax[1, j].set_xlabel(r"Miss-rate target $\alpha$")
ax[0, 0].set_ylabel("Mean IoU under the guarantee")
ax[1, 0].set_ylabel("Realised mean miss rate")
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=7.5, handlelength=2.6)
for a_, t in zip(ax.flat, "abcd"):
    a_.text(-0.14, 1.04, f"({t})", transform=a_.transAxes, fontsize=8.5, fontweight="bold")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(OUT, "fig_guarantee.pdf")); fig.savefig(os.path.join(OUT, "fig_guarantee.png"), dpi=200)
json.dump(curves_out, open(os.path.join(OUT, "guarantee_curves.json"), "w"))

# ------------------------------------------------------------------ figure: per-event Kuro Siwo
pe = {m: model_mean(KS, m, 0.2) for m in ["none", "adabn", "adabn_phys", "tent"]}
evs = sorted(pe["adabn_phys"], key=lambda e: pe["adabn_phys"][e][0])
y = np.arange(len(evs))
fig, ax = plt.subplots(1, 2, figsize=(7.0, 4.8), sharey=True, gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.08})
for i, e in enumerate(evs):
    xs = [pe[m][e][0] for m in ["none", "adabn", "adabn_phys"]]
    ax[0].plot([min(xs), max(xs)], [i, i], color="#c9c8c3", lw=1.0, zorder=1)
for m in ["none", "adabn", "adabn_phys"]:
    st = STYLE[m]
    ax[0].scatter([pe[m][e][0] for e in evs], y, s=22, color=st["color"], marker=st["marker"], label=LABEL[m], zorder=3,
                  edgecolors="#fcfcfb", linewidths=0.6)
    ax[1].scatter([pe[m][e][1] for e in evs], y, s=22, color=st["color"], marker=st["marker"], zorder=3,
                  edgecolors="#fcfcfb", linewidths=0.6)
ax[1].axvline(0.2, color="#0b0b0b", lw=0.8, ls=":")
ax[1].text(0.205, len(evs) - 0.6, "target 0.2", fontsize=7, color="#52514e", va="top")
ax[0].set_yticks(y); ax[0].set_yticklabels([f"KS-{e}" for e in evs], fontsize=6.5)
ax[0].set_xlabel("IoU under the 20% guarantee"); ax[1].set_xlabel("Miss rate under the 20% guarantee")
ax[0].set_xlim(0, 1); ax[1].set_xlim(0, max(0.6, max(pe[m][e][1] for m in pe for e in evs) + 0.03))
ax[0].grid(axis="y", visible=False); ax[1].grid(axis="y", visible=False)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.16, wspace=0.08)
fig.savefig(os.path.join(OUT, "fig_ks_events.pdf")); fig.savefig(os.path.join(OUT, "fig_ks_events.png"), dpi=200)

gain = np.array([pe["adabn_phys"][e][0] - pe["adabn"][e][0] for e in evs])
gain_none = np.array([pe["adabn_phys"][e][0] - pe["none"][e][0] for e in evs])
numbers["ks_events_crc0.2"] = {"better_than_adabn": int((gain > 0).sum()), "mean_gain_vs_adabn": float(gain.mean()),
                               "better_than_none": int((gain_none > 0).sum()), "mean_gain_vs_none": float(gain_none.mean()),
                               "meeting_mean_over_models": {m: float(np.mean([np.sum(np.array(list(per_event(d, m, 0.2).values()))[:, 1] <= 0.2) for d in KS])) for m in pe},
                               "worst_fnr_model_mean": {m: float(max(pe[m][e][1] for e in evs)) for m in pe}}
json.dump(json.loads(json.dumps(numbers, default=fmt)), open(os.path.join(OUT, "paper_numbers.json"), "w"), indent=1)
print(json.dumps(json.loads(json.dumps(numbers, default=fmt)), indent=1))
