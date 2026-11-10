"""
Supervised reference that uses the same labelled target floods as the guarantee calibration.

The guarantee is calibrated on labelled floods of the target dataset. A fair question is what those labels would buy
if they were used for fine-tuning instead. Here the Kuro Siwo events are split into 5 folds (by event); the source
model is fine-tuned on the labelled chips of four folds and tested on the events of the fifth, so every event is
predicted by a model that never saw it. Histograms are stored as method "ft" in the same format as ks_eval.py.

Usage: python ks_finetune.py --data data --ks data_ks --runs runs --model-seed 0 --out runs_ks_ft
"""
import argparse
import os
import time

import numpy as np

from conformal_px1 import BINS, curves
from ks_eval import events_of, load_ks_event


def main():
    import torch
    from sen1floods11_unet_baseline import IGNORE, channel_stats, read_split
    import sen1floods11_unet_baseline as base
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ks", default="data_ks")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs_ks_ft")
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"ks_hist_m{a.model_seed}_v2.npz")
    done = dict(np.load(path)) if os.path.exists(path) else {}
    mean, std = channel_stats(a.data, read_split(a.data, "train"))
    m_t = torch.tensor(mean, device=device)[None, :, None, None]
    s_t = torch.tensor(std, device=device)[None, :, None, None]
    events = events_of(a.ks)
    names = sorted(events)
    data = {e: load_ks_event(events[e]) for e in names}
    build = getattr(base, "build_model", None)
    for fold in range(a.folds):
        test = names[fold::a.folds]
        if all(f"ft|V0|{e}" in done for e in test):
            continue
        t0 = time.time()
        torch.manual_seed(fold)
        gen = np.random.default_rng(fold)
        raw = [r for e in names if e not in test for r in data[e]["raw"]]
        lab = [l for e in names if e not in test for l in data[e]["lab"]]
        model = (build(cin=2) if build else base.UNet(cin=2)).to(device)
        model.load_state_dict(torch.load(os.path.join(a.runs, f"V0_seed{a.model_seed}", "model.pt"), map_location=device))
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device=device), ignore_index=IGNORE)
        for step in range(a.steps):
            xs, ys = [], []
            for i in gen.integers(0, len(raw), a.batch):
                H, W = lab[i].shape
                c = min(224, H, W)
                r0, c0 = gen.integers(0, H - c + 1), gen.integers(0, W - c + 1)
                x = torch.from_numpy(raw[i][:, r0:r0 + c, c0:c0 + c]); y = torch.from_numpy(lab[i][r0:r0 + c, c0:c0 + c].astype(np.int64))
                k = int(gen.integers(0, 4))
                x, y = torch.rot90(x, k, (1, 2)), torch.rot90(y, k, (0, 1))
                if gen.random() < 0.5:
                    x, y = x.flip(2), y.flip(1)
                xs.append(x); ys.append(y)
            x = (torch.stack(xs).to(device) - m_t) / s_t
            loss = ce(model(x), torch.stack(ys).to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        for e in test:
            ev = data[e]
            h = np.zeros((2, 2, BINS), np.int64)
            with torch.no_grad():
                for i in range(len(ev["raw"])):
                    p = model((torch.from_numpy(ev["raw"][i]).to(device)[None] - m_t) / s_t).softmax(1)[0, 1]
                    b = (p * BINS).long().clamp(0, BINS - 1)
                    for sub, key in [(0, "lab"), (1, "lab_alt")]:
                        if key in ev:
                            l = torch.from_numpy(ev[key][i].astype(np.int64)).to(device)
                            for c in (0, 1):
                                h[sub, c] += torch.bincount(b[l == c], minlength=BINS).cpu().numpy()
            done[f"ft|V0|{e}"] = h
            fnr, _, iou = curves(h[0])
            print(f"ft fold {fold} event {e}: miss rate at 0.5 {fnr[BINS // 2]:.3f}, IoU {iou[BINS // 2]:.3f}", flush=True)
        np.savez_compressed(path, **done)
        print(f"fold {fold} done in {time.time() - t0:.0f}s ({len(raw)} training chips)", flush=True)
    print("KS_FT_DONE", flush=True)


if __name__ == "__main__":
    main()
