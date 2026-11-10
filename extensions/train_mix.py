"""
Train on Sen1Floods11 hand-labelled chips plus Kuro Siwo chips (more diverse training floods), same recipe as
train_px1.py otherwise: 256x256 random crops of Sen1Floods11 (224x224 for Kuro Siwo chips, their full size),
flips/rotations, weighted cross-entropy, AdamW + cosine, model selected on the Sen1Floods11 validation split.

Each epoch draws as many Kuro Siwo chips as there are Sen1Floods11 training chips (balanced 50/50), so Sen1Floods11
is seen exactly as often as in train_px1.py. Batches never mix the two chip sizes.

Model: build_model() from sen1floods11_unet_baseline (PX1_ARCH selects U-Net or ResNet-34 U-Net).

Usage: PX1_ARCH=r34 python train_mix.py --data data --ks data_ks --seed 0 --out runs_r34ks --lr 3e-4
"""
import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from sen1floods11_unet_baseline import IGNORE, build_model, channel_stats, read_split
from train_px1 import Chips, evaluate


def ks_chip(f, mean, std):
    z = np.load(f)
    x = np.nan_to_num(np.clip(np.stack([z["vv"], z["vh"]]).astype(np.float32), -50, 5), nan=-50.0)
    m = z["mask"]
    y = np.full(m.shape, IGNORE, np.int64)
    y[m == 0] = 0
    y[(m == 1) | (m == 2)] = 1
    y[(z["valid"] == 0) | ~np.isfinite(z["vv"]) | ~np.isfinite(z["vh"])] = IGNORE
    x = (x - mean[:, None, None]) / std[:, None, None]
    S = 224  # Kuro Siwo chip size; pad or crop anything else so batches stack
    h, w = y.shape
    if h < S or w < S:
        x = np.pad(x, ((0, 0), (0, max(0, S - h)), (0, max(0, S - w))), constant_values=0.0)
        y = np.pad(y, ((0, max(0, S - h)), (0, max(0, S - w))), constant_values=IGNORE)
    r, c = random.randint(0, y.shape[0] - S), random.randint(0, y.shape[1] - S)
    x, y = x[:, r:r + S, c:c + S], y[r:r + S, c:c + S]
    if random.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    if random.random() < 0.5:
        x, y = x[:, ::-1, :], y[::-1, :]
    k = random.randint(0, 3)
    return np.rot90(x, k, (1, 2)).copy(), np.rot90(y, k).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ks", default="data_ks")
    ap.add_argument("--variant", default="V0")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs_mix")
    ap.add_argument("--clean-list", default=os.path.join("audit", "clean_chips.txt"))
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clean = {ln.strip() for ln in open(a.clean_list) if ln.strip()} if os.path.exists(a.clean_list) else set()
    tr, va, te = (read_split(a.data, k) for k in ["train", "valid", "test"])
    ks_files = sorted(glob.glob(os.path.join(a.ks, "*_GRD", "*", "*.npz")))
    mean, std = channel_stats(a.data, tr)
    s1f = Chips(a.data, tr, mean, std, True, False)
    print(f"mix {a.variant} seed {a.seed}: {len(tr)} Sen1Floods11 chips + {len(ks_files)} Kuro Siwo chips "
          f"({len({os.path.basename(os.path.dirname(f)) for f in ks_files})} events), device={device}", flush=True)
    model = build_model(cin=2, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device=device), ignore_index=IGNORE)
    run_dir = os.path.join(a.out, f"{a.variant}_seed{a.seed}")
    os.makedirs(run_dir, exist_ok=True)
    ckpt, best = os.path.join(run_dir, "model.pt"), -1.0
    nb = len(tr) // a.batch
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        order = list(range(len(tr))); random.shuffle(order)
        ks_pick = random.sample(ks_files, min(len(ks_files), nb * a.batch))
        batches = [("s1f", order[i * a.batch:(i + 1) * a.batch]) for i in range(nb)] + \
                  [("ks", ks_pick[i * a.batch:(i + 1) * a.batch]) for i in range(len(ks_pick) // a.batch)]
        random.shuffle(batches)
        for kind, idx in batches:
            if kind == "s1f":
                xs, ys = zip(*[s1f[i][:2] for i in idx])
                x, y = torch.stack(xs), torch.stack(ys)
            else:
                xs, ys = zip(*[ks_chip(f, mean, std) for f in idx])
                x, y = torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        v, _ = evaluate(model, a.data, va, mean, std, False, device, clean)
        if v["all"] > best:
            best = v["all"]; torch.save(model.state_dict(), ckpt)
        print(f"epoch {ep + 1:3d} loss {tot / max(1, len(batches)):.4f} valid IoU {v['all']:.4f} {time.time() - t0:.0f}s", flush=True)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    test, per = evaluate(model, a.data, te, mean, std, False, device, clean)
    res = {"variant": a.variant, "seed": a.seed, "training": "Sen1Floods11 hand + Kuro Siwo", "ks_chips": len(ks_files),
           "best_valid_iou": round(best, 4), "test_iou": test, "test_per_event": per}
    json.dump(res, open(os.path.join(run_dir, "results.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "test_per_event"}), flush=True)
    print("RUN_DONE", flush=True)


if __name__ == "__main__":
    main()
