"""Example-map figure from the MAPROW lines that maps.py prints to the Kaggle log.

Usage: python paper_fig_maps.py maps_log.txt  -> paper/figures/fig_maps.pdf/.png and maps_rows.json
"""
import base64
import json
import os
import re
import sys
import zlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper", "figures")
METHODS = [("none", "No adaptation"), ("tent", "TENT"), ("adabn", "AdaBN"), ("adabn_physa", "Proposed")]
COL = {"tp": (0.165, 0.471, 0.839), "fp": (0.922, 0.408, 0.204), "fn": (0.98, 0.80, 0.18)}


def dec(s, dtype, shape=None):
    a = np.frombuffer(zlib.decompress(base64.b64decode(s)), dtype=dtype)
    return a.reshape(shape) if shape is not None else a


def load(path):
    if path.endswith(".json"):  # compact rows: VV averaged 2x2 and quantised to 16 levels (vv4), masks at full size
        rows = json.load(open(path))
        for r in rows:
            h2, w2 = r["vv2shape"]
            q = dec(r["vv4"], np.uint8)
            v = np.empty(h2 * w2, np.uint8)
            v[0::2] = (q >> 4) * 16 + 8
            v[1::2] = (q & 15) * 16 + 8
            r["vv_full"] = np.kron(v.reshape(h2, w2), np.ones((2, 2), np.uint8))
        return rows
    parts = {}
    for line in open(path, encoding="utf-8"):
        m = re.search(r"MAPROW (\d+) (\d+)/(\d+) (.*)$", line.rstrip("\n"))
        if m:
            parts.setdefault(int(m.group(1)), {})[int(m.group(2))] = (int(m.group(3)), m.group(4))
    rows = []
    for r in sorted(parts):
        n = next(iter(parts[r].values()))[0]
        assert sorted(parts[r]) == list(range(1, n + 1)), f"row {r}: missing chunks"
        rows.append(json.loads("".join(parts[r][j][1] for j in range(1, n + 1))))
    return rows


def overlay(vv, lab, pred):
    g = np.clip(vv / 250.0, 0, 1) * 0.75 + 0.1
    img = np.stack([g, g, g], -1)
    v = lab != 255
    for key, msk in [("tp", pred & (lab == 1)), ("fp", pred & (lab == 0)), ("fn", ~pred & (lab == 1) & v)]:
        img[msk] = COL[key]
    return img


def main(path):
    rows = load(path)
    n = len(rows)
    fig, ax = plt.subplots(n, 6, figsize=(7.0, 1.24 * n + 0.45))
    ax = np.atleast_2d(ax)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    summary = []
    for i, r in enumerate(rows):
        h, w = r["shape"]
        vv = (r["vv_full"] if "vv_full" in r else dec(r["vv"], np.uint8, (h, w))).astype(float)
        lab = dec(r["lab"], np.uint8, (h, w))
        ax[i, 0].imshow(vv, cmap="gray", vmin=0, vmax=250)
        lab_img = np.stack([np.full((h, w), 0.92)] * 3, -1)
        lab_img[lab == 1] = COL["tp"]
        lab_img[lab == 255] = (0.75, 0.75, 0.75)
        ax[i, 1].imshow(lab_img)
        for j, (m, name) in enumerate(METHODS):
            pred = np.unpackbits(dec(r["maps"][m], np.uint8))[: h * w].reshape(h, w).astype(bool)
            ax[i, 2 + j].imshow(overlay(vv, lab, pred))
            ax[i, 2 + j].text(0.03, 0.04, f"IoU {r['iou'][m]:.2f}\nmiss {r['fnr'][m]:.2f}", transform=ax[i, 2 + j].transAxes,
                              fontsize=5.6, color="white", va="bottom",
                              bbox=dict(boxstyle="round,pad=0.15", fc=(0, 0, 0, 0.55), ec="none"))
        ax[i, 0].set_ylabel(f"KS-{r['event']}", fontsize=7)
        summary.append({"event": r["event"], "chip": r["chip"], "iou": r["iou"], "fnr": r["fnr"], "k": r["k"]})
    for j, t in enumerate(["Sentinel-1 VV", "Label"] + [nm for _, nm in METHODS]):
        ax[0, j].set_title(t, fontsize=7.5)
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_linewidth(0.4); s.set_color("#9a9994")
    handles = [Patch(color=COL["tp"], label="water, correct"), Patch(color=COL["fp"], label="false alarm"),
               Patch(color=COL["fn"], label="missed water"), Patch(color=(0.75, 0.75, 0.75), label="no label")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.05, right=0.995, top=0.95, bottom=0.07, wspace=0.04, hspace=0.06)
    fig.savefig(os.path.join(OUT, "fig_maps.pdf"), dpi=300)
    fig.savefig(os.path.join(OUT, "fig_maps.png"), dpi=200)
    json.dump(summary, open(os.path.join(OUT, "maps_rows.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
