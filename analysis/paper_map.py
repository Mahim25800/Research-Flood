"""Study-site map: Sen1Floods11 events, Kuro Siwo test events, MMFlood test activations."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

HERE = os.path.dirname(os.path.abspath(__file__))
S1F_GEO = sys.argv[1]          # Sen1Floods11_Metadata.geojson
MM_ACTS = sys.argv[2]          # MMFlood data/activations.json (github.com/edornd/mmflood)
KS_HIST = sys.argv[3]          # ks_hist_m0_v2.npz (to list the 26 test events)
OUT = os.path.join(HERE, "..", "paper", "figures")

g = json.load(open(S1F_GEO))
s1f = {}
for f in g["features"]:
    if f["properties"]["location"] == "Colombia":  # no hand-labelled chips from this event in our data
        continue
    ring = np.array(f["geometry"]["coordinates"][0])
    s1f.setdefault(f["properties"]["location"], []).append(ring.mean(0))
s1f = {k: np.mean(v, 0) for k, v in s1f.items()}

ks_all = json.load(open(os.path.join(HERE, "..", "data", "ks_events.json")))
ks_test = sorted({k.split("|")[-1] for k in np.load(KS_HIST).files if k.startswith("none|V0|")})
ks = {e: (ks_all[e]["lon"], ks_all[e]["lat"]) for e in ks_test}

import re
txt = open(MM_ACTS).read()
try:
    mm_meta = json.loads(txt)
except json.JSONDecodeError:
    mm_meta = json.loads(re.sub(r",\s*([}\]])", r"\1", txt))
used = json.load(open(os.path.join(HERE, "..", "data", "mm_activations_used.json")))
mm = {a: (mm_meta[a]["lon"], mm_meta[a]["lat"]) for a in used}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
fig = plt.figure(figsize=(7.0, 3.55))
ax = fig.add_axes([0.04, 0.17, 0.94, 0.82])
m = Basemap(projection="mill", llcrnrlat=-48, urcrnrlat=74, llcrnrlon=-130, urcrnrlon=165, resolution="c", ax=ax)
m.drawmapboundary(fill_color="#f4f7fa", linewidth=0.4, color="#9a9994")
m.fillcontinents(color="#e4e3df", lake_color="#f4f7fa")
m.drawcoastlines(linewidth=0.25, color="#9a9994")
m.drawcountries(linewidth=0.15, color="#b8b7b1")
m.drawparallels(np.arange(-40, 80, 20), labels=[1, 0, 0, 0], linewidth=0.2, color="#c9c8c3", fontsize=6, dashes=[1, 0])
m.drawmeridians(np.arange(-120, 180, 40), labels=[0, 0, 0, 1], linewidth=0.2, color="#c9c8c3", fontsize=6, dashes=[1, 0])

sets = [("Sen1Floods11 (11 events; training and leave-one-out)", s1f, "o", "#2a78d6", 46),
        ("Kuro Siwo (26 test events)", ks, "s", "#eb6834", 30),
        ("MMFlood (30 test activations, pre-registered)", mm, "^", "#1baf7a", 36)]
for label, pts, mk, col, sz in sets:
    xy = np.array(list(pts.values()))
    x, y = m(xy[:, 0], xy[:, 1])
    ax.scatter(x, y, s=sz, marker=mk, color=col, edgecolors="#fcfcfb", linewidths=0.7, zorder=5, label=label, alpha=0.95)
fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.2)
fig.savefig(os.path.join(OUT, "fig_sites.pdf")); fig.savefig(os.path.join(OUT, "fig_sites.png"), dpi=200)
print(len(s1f), len(ks), len(mm))
