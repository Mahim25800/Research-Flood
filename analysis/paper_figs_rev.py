"""
Figures of the revised paper, drawn from the analysis JSON that rev_analyse.py prints (no model runs).

Usage: python paper_figs_rev.py results_rev/rev_analysis.json
Outputs in paper_latex/figures: fig_guarantee.pdf (three datasets, proposed adaptive rule), fig_ks_events.pdf,
fig_transfer.pdf (where the calibration floods come from).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "revision", "paper_wip", "figures")
os.makedirs(OUT, exist_ok=True)
R = json.load(open(sys.argv[1]))
GRID = [round(x, 2) for x in np.arange(0.04, 0.501, 0.02)]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
                     "legend.frameon": False, "pdf.fonttype": 42})
STYLE = {"none": dict(color="#6b6a66", ls="--", marker="o"), "tent": dict(color="#eb6834", ls="-", marker="s"),
         "adabn": dict(color="#1baf7a", ls="-", marker="^"), "adabn_physa": dict(color="#2a78d6", ls="-", marker="D")}
LABEL = {"none": "No adaptation", "tent": "TENT-R (running statistics)", "adabn": "AdaBN", "adabn_physa": "Proposed (AdaBN + physics check)"}
SHOW = ["none", "tent", "adabn", "adabn_physa"]
INK, MUTED = "#0b0b0b", "#52514e"

# ------------------------------------------------------------------ miss-rate target curves, three datasets
sets = [("S1F", "Sen1Floods11 (11 floods)", 11), ("KS", "Kuro Siwo (26 events)", 26), ("MM", "MMFlood (30 activations)", 30)]
fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True)
for j, (D, title, n_ev) in enumerate(sets):
    for m in SHOW:
        iou, fnr = R["curves"][D][m]
        st = STYLE[m]
        ax[0, j].plot(GRID, iou, color=st["color"], ls=st["ls"], lw=1.5, marker=st["marker"], ms=3.2, markevery=4, label=LABEL[m])
        ax[1, j].plot(GRID, fnr, color=st["color"], ls=st["ls"], lw=1.5, marker=st["marker"], ms=3.2, markevery=4)
    ax[1, j].plot([0, 0.55], [0, 0.55], color=INK, lw=0.8, ls=":", zorder=1)
    ax[0, j].set_title(title, fontsize=8.5, loc="left", color=INK)
    amin = 1 / n_ev
    for r in (0, 1):
        ax[r, j].axvspan(0.04, amin, color="#f0efec", zorder=0, lw=0)
    ax[1, j].set_ylim(0, 0.55); ax[1, j].set_xlim(0.04, 0.51)
    ax[1, j].set_xlabel(r"Miss-rate target $\alpha$")
    ax[0, j].set_ylim(0, None)
ax[1, 0].text(0.36, 0.43, "target", fontsize=7, color=MUTED)
ax[0, 0].set_ylabel("Mean IoU at the calibrated threshold")
ax[1, 0].set_ylabel("Realised mean miss rate")
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=7.5, handlelength=2.6)
for a_, t in zip(ax[0], "abc"):
    a_.set_title(f"({t}) " + a_.get_title(loc="left"), fontsize=8.5, loc="left", color=INK)
for a_, t in zip(ax[1], "def"):
    a_.set_title(f"({t})", fontsize=8.5, loc="left", color=INK)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(OUT, "fig_guarantee.pdf")); fig.savefig(os.path.join(OUT, "fig_guarantee.png"), dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ Kuro Siwo per event, 20% mean rule
ev = {m: R["events"]["KS"][m]["loeo_0.2"] for m in ["none", "adabn", "adabn_physa"]}
evs = sorted(ev["adabn_physa"], key=lambda e: ev["adabn_physa"][e][0])
y = np.arange(len(evs))
fig, ax = plt.subplots(1, 2, figsize=(7.0, 4.8), sharey=True, gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.08})
for i, e in enumerate(evs):
    xs = [ev[m][e][0] for m in ev]
    ax[0].plot([min(xs), max(xs)], [i, i], color="#c9c8c3", lw=1.0, zorder=1)
for m in ev:
    st = STYLE[m]
    for c in (0, 1):
        ax[c].scatter([ev[m][e][c] for e in evs], y, s=22, color=st["color"], marker=st["marker"], label=LABEL[m] if c == 0 else None,
                      zorder=3, edgecolors="#fcfcfb", linewidths=0.6)
ax[1].axvline(0.2, color=INK, lw=0.8, ls=":")
ax[1].text(0.205, len(evs) - 0.6, "target 0.2", fontsize=7, color=MUTED, va="top")
ax[0].set_yticks(y); ax[0].set_yticklabels([f"KS-{e}" for e in evs], fontsize=6.5)
ax[0].set_xlabel("IoU, 20% mean miss-rate target"); ax[1].set_xlabel("Miss rate, 20% mean miss-rate target")
ax[0].set_xlim(0, 1); ax[1].set_xlim(0, max(0.6, max(ev[m][e][1] for m in ev for e in evs) + 0.03))
ax[0].grid(axis="y", visible=False); ax[1].grid(axis="y", visible=False)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.16, wspace=0.08)
fig.savefig(os.path.join(OUT, "fig_ks_events.pdf")); fig.savefig(os.path.join(OUT, "fig_ks_events.png"), dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ where the calibration floods come from
RULES = [("loeo_0.2", "Same dataset\n(leave one flood out)"), ("x_S1F_0.2", "Sen1Floods11 only\n(no target labels)")]
fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.4), sharex=True)
w = 0.19
for j, (D, title) in enumerate([("KS", "Kuro Siwo (26 events)"), ("MM", "MMFlood (30 activations)")]):
    for i, m in enumerate(SHOW):
        s = R["summary"][D][m]
        for r, (rule, _) in enumerate(RULES):
            x = r + (i - 1.5) * w
            iou, fnr = s[rule][0], s[rule][1]
            ax[0, j].bar(x, iou, w * 0.92, color=STYLE[m]["color"], label=LABEL[m] if r == 0 else None, zorder=3)
            ax[1, j].bar(x, fnr, w * 0.92, color=STYLE[m]["color"], zorder=3)
            ax[0, j].text(x, iou + 0.01 * (1 if D == "KS" else 0.3), f"{iou:.2f}", ha="center", va="bottom", fontsize=6, color=INK)
            ax[1, j].text(x, fnr + 0.012, f"{fnr:.2f}", ha="center", va="bottom", fontsize=6, color=INK)
    ax[1, j].axhline(0.2, color=INK, lw=0.8, ls=":", zorder=4)
    ax[1, j].set_xticks([0, 1]); ax[1, j].set_xticklabels([t for _, t in RULES], fontsize=7.5)
    ax[0, j].set_title(title, fontsize=8.5, loc="left", color=INK)
    ax[1, j].set_ylim(0, 0.68)
    for r in (0, 1):
        ax[r, j].grid(axis="x", visible=False)
ax[1, 1].text(-0.47, 0.215, "target 0.2", fontsize=7, color=MUTED)
ax[0, 0].set_ylabel("Mean IoU"); ax[1, 0].set_ylabel("Realised mean miss rate")
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=7.5)
for a_, t in zip(ax[0], "ab"):
    a_.set_title(f"({t}) " + a_.get_title(loc="left"), fontsize=8.5, loc="left", color=INK)
for a_, t in zip(ax[1], "cd"):
    a_.set_title(f"({t})", fontsize=8.5, loc="left", color=INK)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(OUT, "fig_transfer.pdf")); fig.savefig(os.path.join(OUT, "fig_transfer.png"), dpi=200)
print("figures written")
