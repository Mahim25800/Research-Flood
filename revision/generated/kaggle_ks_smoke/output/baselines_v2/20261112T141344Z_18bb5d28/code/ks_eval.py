"""
PX1 stage 6: cross-dataset test on Kuro Siwo (up to 43 flood events, none seen in training).

Models: the Sen1Floods11 models trained on the standard training split (runs/V0_seed*, V1_seed*).
Every Kuro Siwo flood event is unseen. For each event we apply the same label-free
adaptation methods as tta_px1.py (none, tent, st, phys) and store probability histograms;
then the miss-rate guarantee is calibrated leave-one-event-out across the Kuro Siwo events,
so with ~40 events the guarantee is far less conservative than with 11.

Inputs per chip: post-flood VV, VH (dB) and the Kuro Siwo DEM (10 m grid). Labels: permanent
water and flood water both count as water; class 3 and invalid pixels are ignored.

Usage:  python ks_eval.py --data data --ks data_ks --runs runs --model-seed 0 --out runs_ks
        python ks_eval.py --analyse-only --model-seed 0
"""
import argparse
import csv
import glob
import os
import time

os.environ.setdefault("GDAL_CACHEMAX", "64")

import numpy as np

from tta_px1 import BINS, COMBOS, METHODS, VARIANTS, hand_channel, compute, curves, otsu_counts, report, run_one, save

MIN_CHIPS = 8
V = "_v2"  # v2: events merged across Kuro Siwo splits (v1 dropped the test chips of events in both)


def terrain_10m(dem):
    gy, gx = np.gradient(dem.astype(np.float64), 10.0, 10.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    relh = dem - np.percentile(dem, 2)
    return np.stack([np.clip(relh / 50.0, 0, 2), np.clip(slope / 10.0, 0, 3)]).astype(np.float32)


def load_ks_event(files):
    ev = {"raw": [], "terr": [], "lab": [], "clean": []}
    for f in files:
        z = np.load(f)
        raw = np.clip(np.stack([z["vv"], z["vh"]]).astype(np.float32), -50, 5)
        dem = z["dem"].astype(np.float32)
        if not np.all(np.isfinite(dem)):
            dem = np.nan_to_num(dem, nan=float(np.nanmedian(dem)) if np.isfinite(dem).any() else 0.0)
        m = z["mask"]
        lab = np.full(m.shape, 255, np.uint8)
        lab[m == 0] = 0
        lab[(m == 1) | (m == 2)] = 1
        lab[(z["valid"] == 0) | ~np.isfinite(z["vv"]) | ~np.isfinite(z["vh"])] = 255
        raw = np.nan_to_num(raw, nan=-50.0)
        ev["raw"].append(raw)
        hp = f.replace("data_ks", "data_ks_hand", 1)[:-4] + ".npy"
        hand = np.load(hp) if os.path.exists(hp) else None
        ev["hand_ok"] = ev.get("hand_ok", 0) + (hand is not None)
        ev["terr"].append(hand_channel(hand, terrain_10m(dem)))
        ev["lab"].append(lab)
        ev["clean"].append(False)
    return ev


def events_of(ks):
    """Event id -> chip files, merged across the test_GRD and train_GRD folders (an event can be in both)."""
    ev = {}
    for d in sorted(glob.glob(os.path.join(ks, "*_GRD", "*"))):
        ev.setdefault(os.path.basename(d), []).extend(sorted(glob.glob(os.path.join(d, "*.npz"))))
    return {e: f for e, f in sorted(ev.items()) if len(f) >= MIN_CHIPS}


def phase1(args):
    import torch
    from sen1floods11_unet_baseline import channel_stats, read_split
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"ks_hist_m{args.model_seed}{V}.npz")
    done = dict(np.load(path)) if os.path.exists(path) else {}
    mean, std = channel_stats(args.data, read_split(args.data, "train"))
    events = events_of(args.ks)
    print(f"{len(events)} Kuro Siwo events with >= {MIN_CHIPS} chips", flush=True)
    for e, files in events.items():
        todo = [(m, v) for v in VARIANTS for m in args.methods if f"{m}|{v}|{e}" not in done]
        if not todo and f"otsu|{e}" in done:
            continue
        ev = load_ks_event(files)
        if f"otsu|{e}" not in done:
            done[f"otsu|{e}"], thr = otsu_counts(ev)
        for m, v in todo:
            t0 = time.time()
            ckpt = os.path.join(args.runs, f"{v}_seed{args.model_seed}", "model.pt")
            h = run_one(m, v, e, ev, mean, std, args, device, ckpt=ckpt)
            done[f"{m}|{v}|{e}"] = h
            fnr, _, iou = curves(h[0])
            print(f"{m:10s} {v} event {e:>4s} ({len(files)} chips): miss rate at 0.5 {fnr[BINS // 2]:.3f}, "
                  f"IoU {iou[BINS // 2]:.3f}, {time.time() - t0:.0f}s", flush=True)
            save(path, done)
        save(path, done)
        if device == "cuda":
            torch.cuda.empty_cache()


def analyse(args):
    data = dict(np.load(os.path.join(args.out, f"ks_hist_m{args.model_seed}{V}.npz")))
    events = sorted({k.split("|")[2] for k in data if k.count("|") == 2})
    report(compute(data, events), os.path.join(args.out, f"ks_results_m{args.model_seed}{V}.json"),
           os.path.join(args.out, f"ks_summary_m{args.model_seed}{V}.md"))
    print("KS_EVAL_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ks", default="data_ks")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs_ks")
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--methods", default=",".join(METHODS + COMBOS))
    a = ap.parse_args()
    a.methods = a.methods.split(",")
    if not a.analyse_only:
        phase1(a)
    analyse(a)
