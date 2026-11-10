"""
PX1 stage 2: can terrain physics fix the baseline's mistakes?

For every test and Bolivia chip this script
  1. predicts water with the trained baseline (results/unet_seed0.pt),
  2. counts true positives, false alarms (FP) and misses (FN) per flood event,
  3. fetches Copernicus GLO-30 elevation for the chip (cached in data/DEM for later
     physics training), and derives slope and height above the chip's low ground,
  4. checks how many false alarms sit on terrain where flood water is implausible,
  5. applies one fixed terrain rule after prediction (dry if slope > 5 deg or
     height above low ground > 15 m) and re-scores IoU,
  6. saves prediction maps for the weakest events.

The rule's thresholds are fixed in advance (not tuned on the test set).

Go/no-go criteria, set before running:
  GO if, pooled over the three weakest events,
     (a) false alarms are at least 40% of the errors, AND
     (b) at least 30% of those false alarms are terrain-implausible, AND
     (c) the terrain rule raises their pooled IoU by at least 0.03
         without lowering pooled IoU on all other events by more than 0.01.
  PARTIAL if exactly two of (a)-(c) hold. Otherwise NO-GO.

Usage:  python diagnose_errors.py --data data --model results/unet_seed0.pt --out diagnosis
"""
import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import rasterio
import torch
from rasterio.warp import Resampling, reproject

from sen1floods11_unet_baseline import IGNORE, UNet, channel_stats, event_of, load_label, load_s1, read_split

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")

SLOPE_MAX_DEG = 5.0
RELH_MAX_M = 15.0
DEM_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/"
           "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
           "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")


# ---------------------------------------------------------------- elevation
def tile_urls(bounds):
    left, bottom, right, top = bounds
    urls = []
    for lat in range(math.floor(bottom), math.floor(top) + 1):
        for lon in range(math.floor(left), math.floor(right) + 1):
            urls.append(DEM_URL.format(ns="N" if lat >= 0 else "S", lat=abs(lat),
                                       ew="E" if lon >= 0 else "W", lon=abs(lon)))
    return urls


def fetch_dem(s1_path, cache_path):
    """Copernicus DEM resampled onto the chip's grid; cached as GeoTIFF."""
    if os.path.exists(cache_path):
        with rasterio.open(cache_path) as src:
            return src.read(1).astype(np.float32)
    with rasterio.open(s1_path) as ref:
        profile, bounds = ref.profile, ref.bounds
        if ref.crs.is_projected:
            from rasterio.warp import transform_bounds
            bounds = transform_bounds(ref.crs, "EPSG:4326", *bounds)
    dem = np.full((profile["height"], profile["width"]), np.nan, np.float32)
    for url in tile_urls(bounds):
        try:
            with rasterio.open("/vsicurl/" + url) as src:
                part = np.full_like(dem, np.nan)
                reproject(rasterio.band(src, 1), part, dst_transform=profile["transform"],
                          dst_crs=profile["crs"], dst_nodata=np.nan, resampling=Resampling.bilinear)
                dem = np.where(np.isnan(dem), part, dem)
        except Exception:
            continue  # no tile over open sea, or a network failure
    if np.all(np.isnan(dem)):
        return None
    dem = np.where(np.isnan(dem), np.nanmin(dem), dem)  # sea -> lowest land value
    out = profile.copy()
    out.update(count=1, dtype="float32", nodata=None, compress="deflate")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with rasterio.open(cache_path, "w", **out) as dst:
        dst.write(dem, 1)
    return dem


def slope_and_relh(dem, s1_path):
    with rasterio.open(s1_path) as ref:
        t, crs, b = ref.transform, ref.crs, ref.bounds
    if crs.is_geographic:
        lat = math.radians((b.top + b.bottom) / 2)
        dx, dy = abs(t.a) * 111320 * math.cos(lat), abs(t.e) * 110540
    else:
        dx, dy = abs(t.a), abs(t.e)
    gy, gx = np.gradient(dem, dy, dx)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    relh = dem - np.percentile(dem, 2)  # height above the chip's low ground (HAND proxy)
    return slope, relh


# ---------------------------------------------------------------- helpers
def counts(pred, lab):
    m = lab != IGNORE
    p, t = (pred == 1) & m, (lab == 1) & m
    return np.array([np.sum(p & t), np.sum(p & ~t & m), np.sum(~p & t)], dtype=np.int64)  # TP, FP, FN


def iou(c):
    d = c[0] + c[1] + c[2]
    return float(c[0] / d) if d else float("nan")


def save_map(path, vv, lab, pred, dem, implausible, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = lab != IGNORE
    err = np.zeros(lab.shape + (3,), np.float32) + 0.15
    err[(pred == 1) & (lab == 1)] = (0.2, 0.45, 0.9)   # hit: blue
    err[(pred == 1) & (lab == 0)] = (0.9, 0.3, 0.25)   # false alarm: red
    err[(pred == 0) & (lab == 1)] = (0.95, 0.75, 0.2)  # miss: amber
    err[~m] = 0.5
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    ax[0].imshow(vv, cmap="gray", vmin=-25, vmax=0); ax[0].set_title("Sentinel-1 VV (dB)")
    ax[1].imshow(np.where(m, lab, np.nan), cmap="Blues", vmin=0, vmax=1); ax[1].set_title("Label (water)")
    ax[2].imshow(err); ax[2].set_title("Blue hit, red false alarm, amber miss")
    im = ax[3].imshow(dem, cmap="terrain"); ax[3].set_title("Elevation (m); hatched = implausible")
    ax[3].contourf(implausible.astype(float), levels=[0.5, 1.5], hatches=["///"], colors="none")
    fig.colorbar(im, ax=ax[3], fraction=0.046)
    for a in ax:
        a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default=os.path.join("results", "unet_seed0.pt"))
    ap.add_argument("--out", default="diagnosis")
    ap.add_argument("--all-dem", action="store_true", help="also cache DEM for train/valid chips")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = {k: read_split(args.data, k) for k in ["train", "valid", "test", "bolivia"]}
    mean, std = channel_stats(args.data, splits["train"])
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    if args.all_dem:
        todo = splits["train"] + splits["valid"]
        for i, (s, _) in enumerate(todo):
            fetch_dem(os.path.join(args.data, "S1Hand", s),
                      os.path.join(args.data, "DEM", s.replace("S1Hand", "DEM")))
            if (i + 1) % 50 == 0:
                print(f"cached DEM {i + 1}/{len(todo)}", flush=True)

    ev = defaultdict(lambda: {"base": np.zeros(3, np.int64), "rule": np.zeros(3, np.int64),
                              "fp": 0, "fp_implausible": 0, "tp": 0, "tp_implausible": 0,
                              "fn_vv": [], "tp_vv": [], "chips": 0, "no_dem": 0})
    keep = {}
    chips = splits["test"] + splits["bolivia"]
    for i, (s, l) in enumerate(chips):
        s_path = os.path.join(args.data, "S1Hand", s)
        raw = load_s1(s_path)
        lab = load_label(os.path.join(args.data, "LabelHand", l))
        x = torch.from_numpy(((raw - mean[:, None, None]) / std[:, None, None])[None]).to(device)
        with torch.no_grad():
            pred = model(x).argmax(1)[0].cpu().numpy()
        e = ev[event_of(s)]
        e["chips"] += 1
        c = counts(pred, lab)
        e["base"] += c

        dem = fetch_dem(s_path, os.path.join(args.data, "DEM", s.replace("S1Hand", "DEM")))
        if dem is None:
            e["no_dem"] += 1
            e["rule"] += c
            continue
        slope, relh = slope_and_relh(dem, s_path)
        implausible = (slope > SLOPE_MAX_DEG) | (relh > RELH_MAX_M)
        m = lab != IGNORE
        fp = (pred == 1) & (lab == 0) & m
        tp = (pred == 1) & (lab == 1)
        fn = (pred == 0) & (lab == 1)
        e["fp"] += int(fp.sum()); e["fp_implausible"] += int((fp & implausible).sum())
        e["tp"] += int(tp.sum()); e["tp_implausible"] += int((tp & implausible).sum())
        if fn.any():
            e["fn_vv"].append(float(raw[0][fn].mean()))
        if tp.any():
            e["tp_vv"].append(float(raw[0][tp].mean()))
        e["rule"] += counts(np.where(implausible, 0, pred), lab)
        score = int(c[1] + c[2])  # errors on this chip
        if score > keep.get(event_of(s), (0,))[0]:
            keep[event_of(s)] = (score, s_path, raw[0], lab, pred, dem, implausible, s)
        if (i + 1) % 20 == 0:
            print(f"diagnosed {i + 1}/{len(chips)} chips", flush=True)

    rows = {}
    for name, e in sorted(ev.items()):
        b, r = e["base"], e["rule"]
        errs = b[1] + b[2]
        rows[name] = {
            "chips": e["chips"], "chips_without_dem": e["no_dem"],
            "iou": round(iou(b), 4), "iou_with_terrain_rule": round(iou(r), 4),
            "precision": round(float(b[0] / max(1, b[0] + b[1])), 4),
            "recall": round(float(b[0] / max(1, b[0] + b[2])), 4),
            "false_alarm_share_of_errors": round(float(b[1] / max(1, errs)), 4),
            "false_alarms_terrain_implausible": round(e["fp_implausible"] / max(1, e["fp"]), 4),
            "hits_terrain_implausible": round(e["tp_implausible"] / max(1, e["tp"]), 4),
            "mean_vv_of_misses_db": round(float(np.mean(e["fn_vv"])), 2) if e["fn_vv"] else None,
            "mean_vv_of_hits_db": round(float(np.mean(e["tp_vv"])), 2) if e["tp_vv"] else None,
        }

    weakest = sorted([n for n in rows if n != "Bolivia"], key=lambda n: rows[n]["iou"])[:3]
    others = [n for n in rows if n not in weakest]
    pool = lambda names, key: sum((ev[n][key] for n in names), np.zeros(3, np.int64))
    wb, wr, ob, orr = pool(weakest, "base"), pool(weakest, "rule"), pool(others, "base"), pool(others, "rule")
    fp_w = sum(ev[n]["fp"] for n in weakest)
    a = float(wb[1] / max(1, wb[1] + wb[2])) >= 0.40
    b_ = sum(ev[n]["fp_implausible"] for n in weakest) / max(1, fp_w) >= 0.30
    c_ = (iou(wr) - iou(wb) >= 0.03) and (iou(orr) - iou(ob) >= -0.01)
    n_ok = sum([a, b_, c_])
    verdict = "GO" if n_ok == 3 else "PARTIAL" if n_ok == 2 else "NO-GO"
    summary = {
        "rule": f"dry if slope > {SLOPE_MAX_DEG} deg or height above chip low ground > {RELH_MAX_M} m",
        "weakest_events": weakest,
        "weakest_pooled": {"iou": round(iou(wb), 4), "iou_with_rule": round(iou(wr), 4),
                           "false_alarm_share": round(float(wb[1] / max(1, wb[1] + wb[2])), 4),
                           "false_alarms_implausible": round(sum(ev[n]["fp_implausible"] for n in weakest) / max(1, fp_w), 4)},
        "others_pooled": {"iou": round(iou(ob), 4), "iou_with_rule": round(iou(orr), 4)},
        "all_pooled": {"iou": round(iou(wb + ob), 4), "iou_with_rule": round(iou(wr + orr), 4)},
        "criteria": {"a_false_alarms_ge_40pct": a, "b_implausible_ge_30pct": b_, "c_rule_helps": c_},
        "verdict": verdict,
        "per_event": rows,
    }
    with open(os.path.join(args.out, "diagnosis.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_event"}, indent=2))

    for name in weakest + ["Bolivia"]:
        if name in keep:
            _, _, vv, lab, pred, dem, imp, s = keep[name]
            save_map(os.path.join(args.out, f"map_{name}.png"), vv, lab, pred, dem, imp,
                     f"{name}: {s} (IoU event {rows[name]['iou']})")
    print("DIAGNOSIS_DONE")


if __name__ == "__main__":
    main()
