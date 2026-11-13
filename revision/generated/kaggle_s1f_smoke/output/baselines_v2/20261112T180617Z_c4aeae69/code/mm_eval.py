"""
MMFlood final test (pre-registered 9 Nov 2026). Same adaptation code and settings as ks_eval.py.

Per event (chips from mm_prep.py) and model, store probability histograms for each method; the
analysis (mm_analyse.py) pools the parts and models and runs the pre-registered tests.

Usage:  python mm_eval.py --data data --mm data_mm --runs runs --model-seed 0 --part 0 --nparts 2 --out runs_mm
"""
import argparse
import csv
import os
import time

os.environ.setdefault("GDAL_CACHEMAX", "64")
import numpy as np

from ks_eval import terrain_10m
from tta_px1 import BINS, curves, hand_channel, otsu_counts, run_one, save

METHODS_MM = ["none", "tent", "adabn", "adabn_phys", "adabn_physa"]
MIN_CHIPS = 8


def load_mm_event(files):
    ev = {"raw": [], "terr": [], "lab": [], "clean": []}
    for f in files:
        z = np.load(f)
        ev["raw"].append(np.clip(np.stack([z["vv"], z["vh"]]).astype(np.float32), -50, 5))
        hand = z["hand"]
        ev["hand_ok"] = ev.get("hand_ok", 0) + int(np.any(hand != 0))
        ev["terr"].append(hand_channel(hand, terrain_10m(z["dem"])))
        ev["lab"].append(z["lab"])
        ev["clean"].append(False)
    return ev


def events_of(mm):
    ev = {}
    for r in csv.DictReader(open(os.path.join(mm, "index.csv"))):
        ev.setdefault(r["event"], []).append(r["file"])
    return {e: sorted(f) for e, f in sorted(ev.items()) if len(f) >= MIN_CHIPS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--mm", default="data_mm")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs_mm")
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--variants", default="V0")
    ap.add_argument("--methods", default=",".join(METHODS_MM))
    a = ap.parse_args()
    a.methods, variants = a.methods.split(","), a.variants.split(",")

    import torch
    from sen1floods11_unet_baseline import channel_stats, read_split
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"mm_hist_m{a.model_seed}_p{a.part}.npz")
    done = dict(np.load(path)) if os.path.exists(path) else {}
    mean, std = channel_stats(a.data, read_split(a.data, "train"))
    events = list(events_of(a.mm).items())[a.part::a.nparts]
    print(f"part {a.part}/{a.nparts}: {len(events)} MMFlood events", flush=True)
    for e, files in events:
        todo = [(m, v) for v in variants for m in a.methods if f"{m}|{v}|{e}" not in done]
        if not todo and f"otsu|{e}" in done:
            continue
        ev = load_mm_event(files)
        if f"otsu|{e}" not in done:
            done[f"otsu|{e}"], _ = otsu_counts(ev)
        for m, v in todo:
            t0 = time.time()
            ckpt = os.path.join(a.runs, f"{v}_seed{a.model_seed}", "model.pt")
            h = run_one(m, v, e, ev, mean, std, a, device, ckpt=ckpt)
            done[f"{m}|{v}|{e}"] = h
            fnr, _, iou = curves(h[0])
            print(f"{m:11s} {v} event {e} ({len(files)} chips): miss rate at 0.5 {fnr[BINS // 2]:.3f}, IoU {iou[BINS // 2]:.3f}, "
                  f"{time.time() - t0:.0f}s", flush=True)
            save(path, done)
        save(path, done)
        if device == "cuda":
            torch.cuda.empty_cache()
    print("MM_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
