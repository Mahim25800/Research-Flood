"""
MMFlood analysis with the EMS activation as the unit (decided 9 Nov 2026 03:20 BDT, before any MMFlood result was seen).

MMFlood stores each activation as several region folders (EMSR107-1, EMSR107-5, ...). Adaptation runs per folder,
but folders of one activation are the same flood, so for the statistics and for the leave-one-out guarantee they are
merged: the pixel-count histograms of all folders of an activation are summed, giving one curve per activation.
Then the pre-registered analysis (mm_analyse.py: H1-H4, one-sided Wilcoxon, Holm) is run unchanged on the merged data.

Usage: python mm_analyse_act.py --primary adabn_physa --runs runs_mm --mm data_mm --out out_act
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np


def merge(runs, dst):
    os.makedirs(dst, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(runs, "mm_hist_m*_p*.npz"))):
        s = os.path.basename(p).split("_")[2]  # m0, m1, m2
        out_p = os.path.join(dst, f"mm_hist_{s}_p0.npz")
        acc = dict(np.load(out_p)) if os.path.exists(out_p) else {}
        for k, v in np.load(p).items():
            parts = k.split("|")
            parts[-1] = parts[-1].split("-")[0]  # EMSR107-5 -> EMSR107
            kk = "|".join(parts)
            acc[kk] = acc[kk] + v if kk in acc else v.copy()
        np.savez(out_p, **acc)
    n = {k.split("|")[-1] for p in glob.glob(os.path.join(dst, "*.npz")) for k in np.load(p).keys()}
    print(f"merged into {len(n)} activations", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="adabn_physa")
    ap.add_argument("--runs", default="runs_mm")
    ap.add_argument("--mm", default="data_mm")
    ap.add_argument("--out", default="out_act")
    a = ap.parse_args()
    merged = a.runs.rstrip("/") + "_act"
    if os.path.isdir(merged):
        raise SystemExit(f"{merged} exists; remove it first so histograms are not added twice")
    merge(a.runs, merged)
    sys.exit(subprocess.call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mm_analyse.py"),
                              "--primary", a.primary, "--runs", merged, "--mm", a.mm, "--out", a.out]))
