"""
PX1 step 3: U-Net baseline on Sen1Floods11 (Sentinel-1 only, hand-labelled split).

Reports water IoU on the official test split and on the held-out Bolivia event,
plus IoU per flood event, so it doubles as the first "unseen event" measurement.

1) Get the data (about 1 GB; needs the Google Cloud SDK, no account needed):
     mkdir -p data && cd data
     gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand .
     gsutil -m cp -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand .
     gsutil -m cp -r gs://sen1floods11/v1.1/splits/flood_handlabeled .
     cd ..
2) Install:  pip install torch rasterio numpy
3) Run:      python sen1floods11_unet_baseline.py --data data --epochs 60
   Quick check: add  --epochs 2 --limit 20

Labels: -1 = no data (ignored), 0 = not water, 1 = water.
"""
import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

IGNORE = 255


# ---------------------------------------------------------------- data
def read_split(data_dir, name):
    path = os.path.join(data_dir, "flood_handlabeled", f"flood_{name}_data.csv")
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r]
    return [(r[0].strip(), r[1].strip()) for r in rows]


def event_of(fname):
    return fname.split("_")[0]  # e.g. "Bolivia_103757_S1Hand.tif" -> "Bolivia"


def load_s1(path):
    with rasterio.open(path) as src:
        x = src.read().astype(np.float32)  # (2, H, W): VV, VH in dB
    x = np.nan_to_num(x, nan=-50.0, posinf=0.0, neginf=-50.0)
    return np.clip(x, -50.0, 5.0)


def load_label(path):
    with rasterio.open(path) as src:
        y = src.read(1).astype(np.int64)
    y[y < 0] = IGNORE
    return y


class Chips(Dataset):
    def __init__(self, data_dir, pairs, mean, std, train):
        self.items = [(os.path.join(data_dir, "S1Hand", s), os.path.join(data_dir, "LabelHand", l), s)
                      for s, l in pairs]
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s_path, l_path, name = self.items[i]
        x = (load_s1(s_path) - self.mean) / self.std
        y = load_label(l_path)
        if self.train:
            r, c = random.randint(0, x.shape[1] - 256), random.randint(0, x.shape[2] - 256)
            x, y = x[:, r:r + 256, c:c + 256], y[r:r + 256, c:c + 256]
            if random.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if random.random() < 0.5:
                x, y = x[:, ::-1, :], y[::-1, :]
            k = random.randint(0, 3)
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()), name


def channel_stats(data_dir, pairs):
    vals = [load_s1(os.path.join(data_dir, "S1Hand", s)).reshape(2, -1) for s, _ in pairs]
    v = np.concatenate(vals, axis=1)
    return v.mean(1).astype(np.float32), v.std(1).astype(np.float32) + 1e-6


# ---------------------------------------------------------------- model
def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, cin=2, ncls=2, w=32):
        super().__init__()
        ch = [w, w * 2, w * 4, w * 8, w * 16]
        self.downs = nn.ModuleList([block(cin, ch[0])] + [block(ch[i], ch[i + 1]) for i in range(4)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(4))])
        self.decs = nn.ModuleList([block(ch[i] * 2, ch[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(ch[0], ncls, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.downs):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.ups, self.decs):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.head(x)


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, loader, device):
    """Water IoU pooled over all pixels, plus pooled IoU per flood event."""
    model.eval()
    tot = np.zeros(3)  # intersection, union, labelled pixels
    per_event = defaultdict(lambda: np.zeros(3))
    for x, y, names in loader:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        y = y.numpy()
        for p, t, n in zip(pred, y, names):
            m = t != IGNORE
            inter = np.sum((p == 1) & (t == 1) & m)
            union = np.sum(((p == 1) | (t == 1)) & m)
            stats = np.array([inter, union, m.sum()])
            tot += stats
            per_event[event_of(n)] += stats
    iou = lambda s: float(s[0] / s[1]) if s[1] > 0 else float("nan")
    return iou(tot), {e: round(iou(s), 4) for e, s in sorted(per_event.items())}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="use only N chips per split (smoke test)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = {k: read_split(args.data, k) for k in ["train", "valid", "test", "bolivia"]}
    if args.limit:
        splits = {k: v[:args.limit] for k, v in splits.items()}
    print({k: len(v) for k, v in splits.items()}, "device:", device)

    mean, std = channel_stats(args.data, splits["train"])
    mk = lambda k, tr, bs: DataLoader(Chips(args.data, splits[k], mean, std, tr), batch_size=bs,
                                      shuffle=tr, num_workers=2, drop_last=tr)
    train_dl = mk("train", True, args.batch)
    eval_dl = {k: mk(k, False, 4) for k in ["valid", "test", "bolivia"]}

    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # water is the minority class; weight it up
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device=device), ignore_index=IGNORE)

    os.makedirs(args.out, exist_ok=True)
    best, best_path = -1.0, os.path.join(args.out, f"unet_seed{args.seed}.pt")
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); run = 0.0
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step(); run += loss.item()
        sched.step()
        v_iou, _ = evaluate(model, eval_dl["valid"], device)
        if v_iou > best:
            best = v_iou; torch.save(model.state_dict(), best_path)
        print(f"epoch {ep + 1:3d}  loss {run / max(1, len(train_dl)):.4f}  valid IoU {v_iou:.4f}  {time.time() - t0:.0f}s")

    model.load_state_dict(torch.load(best_path, map_location=device))
    res = {"seed": args.seed, "best_valid_iou": round(best, 4)}
    for k in ["test", "bolivia"]:
        iou, per = evaluate(model, eval_dl[k], device)
        res[f"{k}_iou"] = round(iou, 4); res[f"{k}_iou_per_event"] = per
    print(json.dumps(res, indent=2))
    with open(os.path.join(args.out, f"results_seed{args.seed}.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
