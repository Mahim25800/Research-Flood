"""
PX1 stage 2c: flag hand labels that are unreliable, without using any model.

For each of the 446 hand-labelled chips:
  cloud_frac        share of Sentinel-2 pixels that look like cloud (blue B2 > 0.20 reflectance)
  bright_water      share of labelled-water pixels that are bright on radar (VV > -15 dB)
  dark_dry          share of labelled-dry pixels that are very dark on radar (VV < -18 and VH < -25 dB)

A chip is "suspect" if any of these holds (thresholds fixed in advance):
  cloud_frac > 0.30
  bright_water > 0.60, with at least 1,000 labelled-water pixels
  dark_dry > 0.10

Writes audit/label_audit.csv, audit/clean_chips.txt and audit/summary.json.
Usage:  python audit_labels.py --data data --out audit
"""
import argparse
import csv
import json
import os
import socket
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio

from sen1floods11_unet_baseline import IGNORE, event_of, load_label, load_s1, read_split

socket.setdefaulttimeout(60)
CLOUD_BLUE = 2000          # L1C digital numbers (reflectance x 10000)
T_CLOUD, T_BRIGHT, T_DARK, MIN_WATER = 0.30, 0.60, 0.10, 1000


def fetch_s2(name, data_dir):
    dest = os.path.join(data_dir, "S2Hand", name)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = "https://storage.googleapis.com/sen1floods11/" + urllib.parse.quote(
        f"v1.1/data/flood_events/HandLabeled/S2Hand/{name}")
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(url, dest + ".part")
            os.replace(dest + ".part", dest)
            return dest
        except Exception:
            if attempt == 3:
                return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="audit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pairs = []
    for split in ["train", "valid", "test", "bolivia"]:
        pairs += [(split, s, l) for s, l in read_split(args.data, split)]
    s2_names = [s.replace("S1Hand", "S2Hand") for _, s, _ in pairs]
    with ThreadPoolExecutor(8) as ex:
        s2_paths = list(ex.map(lambda n: fetch_s2(n, args.data), s2_names))
    print(f"Sentinel-2 chips available: {sum(p is not None for p in s2_paths)}/{len(pairs)}", flush=True)

    rows, per_event = [], defaultdict(lambda: [0, 0])
    for (split, s, l), s2_path in zip(pairs, s2_paths):
        raw = load_s1(os.path.join(args.data, "S1Hand", s))
        lab = load_label(os.path.join(args.data, "LabelHand", l))
        m = lab != IGNORE
        water, dry = (lab == 1) & m, (lab == 0) & m
        bright_water = float(((raw[0] > -15) & water).sum() / max(1, water.sum()))
        dark_dry = float(((raw[0] < -18) & (raw[1] < -25) & dry).sum() / max(1, dry.sum()))
        cloud = None
        if s2_path:
            with rasterio.open(s2_path) as src:
                blue = src.read(2).astype(np.float32)
            valid = blue > 0
            cloud = float(((blue > CLOUD_BLUE) & valid).sum() / max(1, valid.sum()))
        reasons = []
        if cloud is not None and cloud > T_CLOUD:
            reasons.append("cloud")
        if water.sum() >= MIN_WATER and bright_water > T_BRIGHT:
            reasons.append("bright_water")
        if dark_dry > T_DARK:
            reasons.append("dark_dry")
        rows.append({"split": split, "chip": s, "event": event_of(s), "water_px": int(water.sum()),
                     "cloud_frac": None if cloud is None else round(cloud, 4),
                     "bright_water": round(bright_water, 4), "dark_dry": round(dark_dry, 4),
                     "suspect": int(bool(reasons)), "reasons": "+".join(reasons)})
        per_event[(split, event_of(s))][0] += 1
        per_event[(split, event_of(s))][1] += int(bool(reasons))

    with open(os.path.join(args.out, "label_audit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out, "clean_chips.txt"), "w") as f:
        f.write("\n".join(r["chip"] for r in rows if not r["suspect"]) + "\n")

    reason_counts = defaultdict(int)
    for r in rows:
        for x in filter(None, r["reasons"].split("+")):
            reason_counts[x] += 1
    summary = {
        "thresholds": {"cloud": T_CLOUD, "bright_water": T_BRIGHT, "dark_dry": T_DARK, "min_water_px": MIN_WATER},
        "suspect_total": sum(r["suspect"] for r in rows), "chips_total": len(rows),
        "suspect_by_split": {sp: sum(r["suspect"] for r in rows if r["split"] == sp) for sp in ["train", "valid", "test", "bolivia"]},
        "reason_counts": dict(reason_counts),
        "suspect_by_event": {f"{sp}/{ev}": f"{v[1]}/{v[0]}" for (sp, ev), v in sorted(per_event.items())},
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("AUDIT_DONE")


if __name__ == "__main__":
    main()
