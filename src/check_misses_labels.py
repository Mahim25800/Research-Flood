"""
PX1 stage 2b: (1) can "low ground fills first" recover the baseline's misses?
              (2) are the weak events' labels consistent with what the radar sees?

(1) Fill test. For every pixel, find the nearest pixel the model called water.
    A pixel is "fillable" if it lies within 50 px (500 m) of that water and no more
    than 1 m above it (a local bathtub). The fill rule marks fillable pixels as water.
    We report the share of misses that are fillable (recoverable), the share of true
    dry pixels that are fillable (cost), and IoU before/after the rule.
    Thresholds are fixed in advance.
    GO-fill if the rule raises pooled IoU of the three weakest events by >= 0.03 and
    lowers pooled IoU of the other events by no more than 0.01.

(2) Label check. Per event: how much labelled water looks bright on radar
    (VV > -15 dB) and how much labelled dry land looks dark (VV < -18 dB), compared
    with the strong events. For Pakistan, Ghana and USA, downloads the Sentinel-2
    optical chip the labels were drawn on and saves radar / optical / label panels.
    Also looks for the dataset's metadata file with S1 and S2 acquisition dates.

Usage:  python check_misses_labels.py --data data --model results/unet_seed0.pt --out diagnosis
"""
import argparse
import json
import os
import socket
import urllib.parse
import urllib.request
from collections import defaultdict

import numpy as np
import rasterio
import torch
from scipy import ndimage

from sen1floods11_unet_baseline import IGNORE, UNet, channel_stats, event_of, load_label, load_s1, read_split

socket.setdefaulttimeout(60)
FILL_DIST_PX = 50
FILL_HEIGHT_M = 1.0
BRIGHT_WATER_DB = -15.0
DARK_DRY_DB = -18.0
BUCKET = "sen1floods11"
CHECK_EVENTS = ["Pakistan", "Ghana", "USA"]


def counts(pred, lab):
    m = lab != IGNORE
    p, t = (pred == 1) & m, (lab == 1) & m
    return np.array([np.sum(p & t), np.sum(p & ~t & m), np.sum(~p & t)], np.int64)


def iou(c):
    d = c.sum()
    return float(c[0] / d) if d else float("nan")


def fillable_mask(pred, dem):
    water = pred == 1
    if not water.any():
        return np.zeros_like(water)
    dist, (iy, ix) = ndimage.distance_transform_edt(~water, return_indices=True)
    height_above_water = dem - dem[iy, ix]
    return (~water) & (dist <= FILL_DIST_PX) & (height_above_water <= FILL_HEIGHT_M)


def gcs_list(prefix, delimiter=None):
    q = {"prefix": prefix, "maxResults": "1000", "fields": "items(name,size),prefixes,nextPageToken"}
    if delimiter:
        q["delimiter"] = delimiter
    url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?" + urllib.parse.urlencode(q)
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def gcs_get(name, dest):
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(f"https://storage.googleapis.com/{BUCKET}/" + urllib.parse.quote(name), dest + ".part")
        os.replace(dest + ".part", dest)
    return dest


def save_panel(path, vv, vh, lab, s2_path, pred, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = lab != IGNORE
    n = 5 if s2_path else 4
    fig, ax = plt.subplots(1, n, figsize=(4 * n, 4.4))
    ax[0].imshow(vv, cmap="gray", vmin=-25, vmax=0); ax[0].set_title("S1 VV (dB)")
    ax[1].imshow(vh, cmap="gray", vmin=-32, vmax=-8); ax[1].set_title("S1 VH (dB)")
    ax[2].imshow(np.where(m, lab, np.nan), cmap="Blues", vmin=0, vmax=1); ax[2].set_title("Hand label (water)")
    ax[3].imshow(np.where(m, pred, np.nan), cmap="Blues", vmin=0, vmax=1); ax[3].set_title("Baseline prediction")
    if s2_path:
        with rasterio.open(s2_path) as src:
            b = src.read().astype(np.float32)
        # Sentinel-2 L1C band order in Sen1Floods11: B1..B12 (13 bands); RGB = B4,B3,B2 -> idx 3,2,1
        rgb = np.stack([b[3], b[2], b[1]], -1)
        rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)
        ax[4].imshow(rgb); ax[4].set_title("S2 optical (labels drawn on this)")
    for a in ax:
        a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=85)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default=os.path.join("results", "unet_seed0.pt"))
    ap.add_argument("--out", default="diagnosis")
    ap.add_argument("--offline", action="store_true", help="skip Sentinel-2 and metadata downloads")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = {k: read_split(args.data, k) for k in ["train", "test", "bolivia"]}
    mean, std = channel_stats(args.data, splits["train"])
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    ev = defaultdict(lambda: {"base": np.zeros(3, np.int64), "fill": np.zeros(3, np.int64),
                              "fn": 0, "fn_fillable": 0, "tn": 0, "tn_fillable": 0,
                              "water": 0, "bright_water": 0, "dry": 0, "dark_dry": 0,
                              "relief_m": []})
    panels = defaultdict(list)
    for s, l in splits["test"] + splits["bolivia"]:
        s_path = os.path.join(args.data, "S1Hand", s)
        raw = load_s1(s_path)
        lab = load_label(os.path.join(args.data, "LabelHand", l))
        x = torch.from_numpy(((raw - mean[:, None, None]) / std[:, None, None])[None]).to(device)
        with torch.no_grad():
            pred = model(x).argmax(1)[0].cpu().numpy()
        e, name = ev[event_of(s)], event_of(s)
        m = lab != IGNORE
        e["base"] += counts(pred, lab)
        e["water"] += int(((lab == 1) & m).sum()); e["bright_water"] += int(((lab == 1) & m & (raw[0] > BRIGHT_WATER_DB)).sum())
        e["dry"] += int(((lab == 0) & m).sum()); e["dark_dry"] += int(((lab == 0) & m & (raw[0] < DARK_DRY_DB)).sum())

        dem_path = os.path.join(args.data, "DEM", s.replace("S1Hand", "DEM"))
        if os.path.exists(dem_path):
            with rasterio.open(dem_path) as src:
                dem = src.read(1).astype(np.float32)
            e["relief_m"].append(float(np.percentile(dem, 98) - np.percentile(dem, 2)))
            fill = fillable_mask(pred, dem)
            fn = (pred == 0) & (lab == 1) & m
            tn = (pred == 0) & (lab == 0) & m
            e["fn"] += int(fn.sum()); e["fn_fillable"] += int((fn & fill).sum())
            e["tn"] += int(tn.sum()); e["tn_fillable"] += int((tn & fill).sum())
            e["fill"] += counts(np.where(fill, 1, pred), lab)
        else:
            e["fill"] += counts(pred, lab)
        if name in CHECK_EVENTS:
            errs = int(((pred != lab) & m).sum())
            panels[name].append((errs, s, l, raw, lab, pred))

    rows = {}
    for name, e in sorted(ev.items()):
        rows[name] = {
            "iou": round(iou(e["base"]), 4),
            "iou_with_fill_rule": round(iou(e["fill"]), 4),
            "misses_fillable": round(e["fn_fillable"] / max(1, e["fn"]), 4),
            "true_dry_fillable": round(e["tn_fillable"] / max(1, e["tn"]), 4),
            "label_water_bright_on_radar": round(e["bright_water"] / max(1, e["water"]), 4),
            "label_dry_dark_on_radar": round(e["dark_dry"] / max(1, e["dry"]), 4),
            "median_chip_relief_m": round(float(np.median(e["relief_m"])), 1) if e["relief_m"] else None,
        }
    weakest = sorted([n for n in rows if n != "Bolivia"], key=lambda n: rows[n]["iou"])[:3]
    others = [n for n in rows if n not in weakest]
    pool = lambda names, k: sum((ev[n][k] for n in names), np.zeros(3, np.int64))
    wb, wf, ob, of = pool(weakest, "base"), pool(weakest, "fill"), pool(others, "base"), pool(others, "fill")
    go = (iou(wf) - iou(wb) >= 0.03) and (iou(of) - iou(ob) >= -0.01)
    summary = {
        "fill_rule": f"water if within {FILL_DIST_PX} px of predicted water and <= {FILL_HEIGHT_M} m above it",
        "weakest_events": weakest,
        "weakest_pooled": {"iou": round(iou(wb), 4), "iou_with_fill": round(iou(wf), 4),
                           "misses_fillable": round(sum(ev[n]["fn_fillable"] for n in weakest) / max(1, sum(ev[n]["fn"] for n in weakest)), 4),
                           "true_dry_fillable": round(sum(ev[n]["tn_fillable"] for n in weakest) / max(1, sum(ev[n]["tn"] for n in weakest)), 4)},
        "others_pooled": {"iou": round(iou(ob), 4), "iou_with_fill": round(iou(of), 4)},
        "fill_verdict": "GO" if go else "NO-GO",
        "per_event": rows,
    }

    if not args.offline:
        try:
            top = gcs_list("v1.1/", delimiter="/")
            summary["bucket_v1.1_root"] = [it["name"] for it in top.get("items", [])] + top.get("prefixes", [])
            meta = [n for n in summary["bucket_v1.1_root"] if "etadata" in n or n.endswith((".geojson", ".csv"))]
            for n in meta[:3]:
                p = gcs_get(n, os.path.join(args.data, "meta", os.path.basename(n)))
                if p.endswith(".geojson"):
                    with open(p) as f:
                        feats = json.load(f).get("features", [])
                    summary["acquisition_dates"] = {
                        str(ft["properties"].get("location", ft["properties"].get("name", i))): {k: v for k, v in ft["properties"].items() if "date" in k.lower()}
                        for i, ft in enumerate(feats)}
        except Exception as ex:
            summary["metadata_error"] = str(ex)

    for name in CHECK_EVENTS:
        for errs, s, l, raw, lab, pred in sorted(panels[name], key=lambda t: -t[0])[:3]:
            s2_path = None
            if not args.offline:
                s2 = s.replace("S1Hand", "S2Hand")
                try:
                    s2_path = gcs_get(f"v1.1/data/flood_events/HandLabeled/S2Hand/{s2}", os.path.join(args.data, "S2Hand", s2))
                except Exception as ex:
                    summary.setdefault("s2_errors", []).append(f"{s2}: {ex}")
            save_panel(os.path.join(args.out, f"labelcheck_{s.replace('_S1Hand.tif', '')}.png"),
                       raw[0], raw[1], lab, s2_path, pred, f"{s} ({errs} wrong pixels)")

    with open(os.path.join(args.out, "check_misses_labels.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_event", "acquisition_dates")}, indent=2))
    print("CHECK_DONE")


if __name__ == "__main__":
    main()
