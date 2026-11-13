"""
PX1 stage 3: does elevation help, as an input and as a soft physics loss?

Variants (pick with flags):
  V0   radar only                              --inputs s1
  V0c  radar only, trained on clean chips      --inputs s1 --train-subset clean
  V1   radar + elevation features              --inputs s1dem
  V2   radar + elevation + soft gravity loss   --inputs s1dem --phys 0.1

Elevation features (from the cached Copernicus DEM): height above the chip's low
ground (2nd percentile), divided by 50 m and clipped to [0, 2], and slope in degrees,
divided by 10 and clipped to [0, 3].

Gravity loss (soft version of EvaNet's elevation loss): for neighbouring pixels i, j
whose elevations differ by more than 0.2 m, with j lower, penalise p_i * (1 - p_j):
water on the higher pixel but not on the lower one. Off for 5 epochs, then ramps up
to full weight by epoch 15.

Evaluates on test and Bolivia, on all chips and on clean chips (audit/clean_chips.txt),
pooled and per event. Same U-Net, crops, augmentation and schedule as the baseline.

Usage:  python train_px1.py --variant V1 --inputs s1dem --seed 0
"""
import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

os.environ.setdefault("GDAL_CACHEMAX", "64")  # cap GDAL's block cache (MB)

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sen1floods11_unet_baseline import IGNORE, UNet, channel_stats, event_of, load_label, load_s1, read_split

DZ_MIN = 0.2


def process_mem_mb():
    """Resident memory of this process in MB (psutil if available)."""
    try:
        import psutil
        return int(psutil.Process().memory_info().rss / 2**20)
    except Exception:
        return -1


def dem_path(data_dir, s):
    return os.path.join(data_dir, "DEM", s.replace("S1Hand", "DEM"))


def load_dem(data_dir, s, s1_path):
    p = dem_path(data_dir, s)
    if not os.path.exists(p):
        return np.zeros((512, 512), np.float32)
    with rasterio.open(p) as src:
        return src.read(1).astype(np.float32)


def terrain_features(dem, s1_path):
    with rasterio.open(s1_path) as ref:
        t, crs, b = ref.transform, ref.crs, ref.bounds
    if crs.is_geographic:
        lat = math.radians((b.top + b.bottom) / 2)
        dx, dy = abs(t.a) * 111320 * math.cos(lat), abs(t.e) * 110540
    else:
        dx, dy = abs(t.a), abs(t.e)
    gy, gx = np.gradient(dem, dy, dx)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    relh = dem - np.percentile(dem, 2)
    return np.stack([np.clip(relh / 50.0, 0, 2), np.clip(slope / 10.0, 0, 3)]).astype(np.float32)


class Chips(Dataset):
    def __init__(self, data_dir, pairs, mean, std, train, use_dem):
        self.data_dir, self.pairs, self.train, self.use_dem = data_dir, pairs, train, use_dem
        self.mean, self.std = mean[:, None, None], std[:, None, None]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        s, l = self.pairs[i]
        s1_path = os.path.join(self.data_dir, "S1Hand", s)
        x = (load_s1(s1_path) - self.mean) / self.std
        y = load_label(os.path.join(self.data_dir, "LabelHand", l))
        dem = load_dem(self.data_dir, s, s1_path)
        if self.use_dem:
            x = np.concatenate([x, terrain_features(dem, s1_path)], 0)
        if self.train:
            r, c = random.randint(0, 256), random.randint(0, 256)
            x, y, dem = x[:, r:r + 256, c:c + 256], y[r:r + 256, c:c + 256], dem[r:r + 256, c:c + 256]
            if random.random() < 0.5:
                x, y, dem = x[:, :, ::-1], y[:, ::-1], dem[:, ::-1]
            if random.random() < 0.5:
                x, y, dem = x[:, ::-1, :], y[::-1, :], dem[::-1, :]
            k = random.randint(0, 3)
            x, y, dem = np.rot90(x, k, (1, 2)), np.rot90(y, k), np.rot90(dem, k)
        return (torch.from_numpy(x.copy()), torch.from_numpy(y.copy()),
                torch.from_numpy(dem.copy()), s)


def gravity_loss(logits, dem):
    p = logits.softmax(1)[:, 1]
    total, n = logits.new_zeros(()), 0
    for dz, pa, pb in [(dem[:, :, 1:] - dem[:, :, :-1], p[:, :, :-1], p[:, :, 1:]),
                       (dem[:, 1:, :] - dem[:, :-1, :], p[:, :-1, :], p[:, 1:, :])]:
        b_lower = dz < -DZ_MIN   # pixel b lower than a
        a_lower = dz > DZ_MIN
        total = total + (pa * (1 - pb))[b_lower].sum() + (pb * (1 - pa))[a_lower].sum()
        n += int(b_lower.sum() + a_lower.sum())
    return total / max(1, n)


@torch.no_grad()
def evaluate(model, data_dir, pairs, mean, std, use_dem, device, clean):
    model.eval()
    ds = Chips(data_dir, pairs, mean, std, False, use_dem)
    pooled = {"all": np.zeros(3), "clean": np.zeros(3)}
    per_event = defaultdict(lambda: {"all": np.zeros(3), "clean": np.zeros(3)})
    for i in range(len(ds)):
        x, y, _, s = ds[i]
        pred = model(x[None].to(device)).argmax(1)[0].cpu().numpy()
        t = y.numpy()
        m = t != IGNORE
        c = np.array([np.sum((pred == 1) & (t == 1) & m), np.sum((pred == 1) & (t == 0) & m), np.sum((pred == 0) & (t == 1) & m)])
        keys = ["all"] + (["clean"] if s in clean else [])
        for k in keys:
            pooled[k] += c
            per_event[event_of(s)][k] += c
    iou = lambda c: round(float(c[0] / c.sum()), 4) if c.sum() else None
    return ({k: iou(v) for k, v in pooled.items()},
            {e: {k: iou(v) for k, v in d.items()} for e, d in sorted(per_event.items())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--inputs", choices=["s1", "s1dem"], default="s1")
    ap.add_argument("--phys", type=float, default=0.0)
    ap.add_argument("--train-subset", choices=["all", "clean"], default="all")
    ap.add_argument("--clean-list", default=os.path.join("audit", "clean_chips.txt"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (0 = load in main process; saves RAM on Windows)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--holdout", default="", help="flood event to hold out entirely (leave-one-event-out)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_dem = args.inputs == "s1dem"
    with open(args.clean_list) as f:
        clean = {ln.strip() for ln in f if ln.strip()}
    splits = {k: read_split(args.data, k) for k in ["train", "valid", "test", "bolivia"]}
    if args.limit:
        splits = {k: v[:args.limit] for k, v in splits.items()}
    if args.holdout:  # train and select on the other events; test on every chip of the held-out event
        every = splits["train"] + splits["valid"] + splits["test"] + splits["bolivia"]
        splits["holdout"] = [p for p in every if event_of(p[0]) == args.holdout]
        for k in ["train", "valid"]:
            splits[k] = [p for p in splits[k] if event_of(p[0]) != args.holdout]
        if not splits["holdout"]:
            raise SystemExit(f"no chips for event {args.holdout}")
    train_pairs = splits["train"] if args.train_subset == "all" else [p for p in splits["train"] if p[0] in clean]
    print(f"{args.variant} seed {args.seed}: {len(train_pairs)} training chips, inputs={args.inputs}, phys={args.phys}, device={device}", flush=True)

    mean, std = channel_stats(args.data, splits["train"])
    dl = DataLoader(Chips(args.data, train_pairs, mean, std, True, use_dem), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers, drop_last=True)
    model = UNet(cin=4 if use_dem else 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device=device), ignore_index=IGNORE)

    tag = f"_hold{args.holdout}" if args.holdout else ""
    run_dir = os.path.join(args.out, f"{args.variant}{tag}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    best, ckpt, last = -1.0, os.path.join(run_dir, "model.pt"), os.path.join(run_dir, "last.pt")
    start = 0
    if os.path.exists(last):  # resume after a crash
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, best = st["epoch"], st["best"]
        random.setstate(st["py_rng"]); np.random.set_state(st["np_rng"]); torch.set_rng_state(st["torch_rng"])
        print(f"resumed from epoch {start}, best valid IoU {best:.4f}", flush=True)
    for ep in range(start, args.epochs):
        model.train(); t0 = time.time(); tot = phys_tot = 0.0
        lam = args.phys * min(1.0, max(0.0, (ep - 5) / 10))
        for x, y, dem, _ in dl:
            x, y, dem = x.to(device), y.to(device), dem.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            if lam > 0:
                g = gravity_loss(logits, dem)
                loss = loss + lam * g
                phys_tot += g.item()
            loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        v, _ = evaluate(model, args.data, splits["valid"], mean, std, use_dem, device, clean)
        if v["all"] > best:
            best = v["all"]; torch.save(model.state_dict(), ckpt)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep + 1, "best": best, "py_rng": random.getstate(), "np_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state()}, last)
        print(f"epoch {ep + 1:3d} loss {tot / max(1, len(dl)):.4f} phys {phys_tot / max(1, len(dl)):.4f} "
              f"valid IoU {v['all']:.4f} {time.time() - t0:.0f}s mem {process_mem_mb()} MB", flush=True)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    res = {"variant": args.variant, "seed": args.seed, "inputs": args.inputs, "phys": args.phys,
           "train_subset": args.train_subset, "train_chips": len(train_pairs), "best_valid_iou": round(best, 4),
           "holdout": args.holdout or None, "holdout_chips": len(splits.get("holdout", []))}
    for k in (["holdout"] if args.holdout else ["test", "bolivia"]):
        pooled, per = evaluate(model, args.data, splits[k], mean, std, use_dem, device, clean)
        res[f"{k}_iou"] = pooled; res[f"{k}_per_event"] = per
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if "per_event" not in k}), flush=True)
    print("RUN_DONE", flush=True)


if __name__ == "__main__":
    main()
