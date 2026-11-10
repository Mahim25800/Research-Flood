"""
MMFlood sanity check (why does every model stay near 0.2 IoU?).

A. Model-free reference from the final-test histograms: Otsu threshold on VV per region folder (already stored as
   tp/fp/fn counts), pooled per activation -> IoU per activation, compared with the models.
B. Raw-data checks on a few activations, at the native 20 m grid and on our 10 m chips:
   value scale of s1_raw (linear power expected: VV mean ~0.05 per the MMFlood code), mask values,
   backscatter of labelled water vs dry, share of labelled water that is dark, Otsu IoU native vs chips
   (a large drop on the chips would point to a resampling/alignment error), and a figure of VV with the label outline.

Usage: python mm_check.py --hist-dir <dir with mm_hist_m0_p*.npz and index.csv> --raw mmflood_raw --out out_check
"""
import argparse
import glob
import json
import os

import numpy as np


def otsu_thr(v):
    hist, edges = np.histogram(v, bins=256)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    mu0 = np.cumsum(hist * c) / np.maximum(w0, 1)
    mu1 = (np.sum(hist * c) - np.cumsum(hist * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (mu0 - mu1) ** 2)])


def iou(pred, lab):
    tp = np.sum(pred & (lab == 1)); fp = np.sum(pred & (lab == 0)); fn = np.sum(~pred & (lab == 1))
    return float(tp / max(1, tp + fp + fn))


def part_a(hist_dir, out):
    import csv
    files = sorted(glob.glob(os.path.join(hist_dir, "**", "mm_hist_m0_p*.npz"), recursive=True))
    cnt = {}
    for f in files:
        z = np.load(f)
        for k in z.files:
            if k.startswith("otsu|"):
                a = k.split("|")[1].split("-")[0]
                cnt[a] = cnt.get(a, 0) + z[k][0]
    rows = {a: {"otsu_iou": float(c[0] / max(1, c.sum()))} for a, c in sorted(cnt.items())}
    idx = glob.glob(os.path.join(hist_dir, "**", "index.csv"), recursive=True)
    if idx:
        w, d = {}, {}
        for r in csv.DictReader(open(idx[0])):
            a = r["event"].split("-")[0]
            w[a] = w.get(a, 0) + int(r["water_px"]); d[a] = d.get(a, 0) + int(r["dry_px"])
        for a in rows:
            rows[a]["water_fraction"] = w.get(a, 0) / max(1, w.get(a, 0) + d.get(a, 0))
    ious = [r["otsu_iou"] for r in rows.values()]
    pooled = np.sum(list(cnt.values()), axis=0)
    print(f"A. Otsu on VV (no model), {len(rows)} activations: mean IoU {np.mean(ious):.3f}, median {np.median(ious):.3f}, "
          f"pixel-pooled IoU {pooled[0] / max(1, pooled.sum()):.3f}", flush=True)
    for a, r in rows.items():
        print(f"   {a}: Otsu IoU {r['otsu_iou']:.3f}, labelled water {100 * r.get('water_fraction', 0):.1f}% of valid pixels", flush=True)
    json.dump(rows, open(os.path.join(out, "otsu_per_activation.json"), "w"), indent=1)
    return rows


def part_b(raw, out, folders):
    import rasterio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mm_prep
    _, dirs = mm_prep.download(raw)
    res, panels = {}, []
    for fo in folders:
        if fo not in dirs:
            print(f"   {fo}: not found", flush=True); continue
        base = dirs[fo]
        name = sorted(os.listdir(os.path.join(base, "s1_raw")))[0]
        with rasterio.open(os.path.join(base, "s1_raw", name)) as s:
            a = s.read().astype(np.float32); nod = s.nodata
        with rasterio.open(os.path.join(base, "mask", name)) as m:
            mk = m.read(1)
        ok = np.isfinite(a[0]) & (a[0] > 0) & np.isfinite(a[1]) & (a[1] > 0)
        vv = np.where(ok, 10 * np.log10(np.maximum(a[0], 1e-6)), np.nan)
        lab = np.full(mk.shape, 255, np.uint8); lab[(mk == 0) & ok] = 0; lab[(mk == 1) & ok] = 1
        w, d = vv[lab == 1], vv[lab == 0]
        thr = otsu_thr(vv[ok & (vv > -49)])
        r = {"image": name, "raw_vv_median": float(np.nanmedian(a[0][ok])), "raw_vh_median": float(np.nanmedian(a[1][ok])),
             "nodata": None if nod is None else float(nod), "mask_values": sorted(set(np.unique(mk).tolist())),
             "water_frac": float(np.mean(lab[lab != 255] == 1)), "vv_db_median_water": float(np.median(w)) if w.size else None,
             "vv_db_median_dry": float(np.median(d)) if d.size else None,
             "water_dark_share": float(np.mean(w < -15)) if w.size else None, "otsu_thr_db": thr,
             "otsu_iou_native": iou((vv < thr) & ok, lab)}
        chips = mm_prep.image_chips(base, name)
        if chips:
            cv = np.concatenate([c["vv"].ravel() for c in chips]); cl = np.concatenate([c["lab"].ravel() for c in chips])
            r["otsu_iou_chips_same_thr"] = iou(cv < thr, cl)
            r["chips_water_frac"] = float(np.mean(cl[cl != 255] == 1))
        res[fo] = r
        print(f"   {fo}: {json.dumps(r)}", flush=True)
        # crop with the most labelled water for the figure
        S = min(1024, *vv.shape)
        wy, wx = np.where(lab == 1)
        cy, cx = (int(np.median(wy)), int(np.median(wx))) if wy.size else (vv.shape[0] // 2, vv.shape[1] // 2)
        y0, x0 = max(0, min(cy - S // 2, vv.shape[0] - S)), max(0, min(cx - S // 2, vv.shape[1] - S))
        panels.append((fo, vv[y0:y0 + S, x0:x0 + S], lab[y0:y0 + S, x0:x0 + S]))
    if panels:
        fig, ax = plt.subplots(2, len(panels), figsize=(3.2 * len(panels), 6.6), squeeze=False)
        for i, (fo, v, l) in enumerate(panels):
            ax[0, i].imshow(np.clip(v, -25, 0), cmap="gray"); ax[0, i].set_title(f"{fo} VV (dB)", fontsize=8)
            ax[1, i].imshow(np.clip(v, -25, 0), cmap="gray"); ax[1, i].contour(l == 1, levels=[0.5], colors="cyan", linewidths=0.5)
            ax[1, i].set_title("label outline", fontsize=8)
            for a_ in ax[:, i]:
                a_.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(out, "mmflood_check.png"), dpi=150)
    json.dump(res, open(os.path.join(out, "raw_checks.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist-dir", required=True)
    ap.add_argument("--raw", default="mmflood_raw")
    ap.add_argument("--out", default="out_check")
    ap.add_argument("--folders", default="EMSR107-5,EMSR167-2,EMSR184-4,EMSR203-3,EMSR265-14,EMSR438-5,EMSR479-0,EMSR514-0")
    ap.add_argument("--skip-raw", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    part_a(a.hist_dir, a.out)
    if not a.skip_raw:
        print("B. raw-data checks", flush=True)
        part_b(a.raw, a.out, a.folders.split(","))
    print("MM_CHECK_DONE", flush=True)
