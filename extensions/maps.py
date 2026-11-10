"""
Example flood maps for the paper (Kuro Siwo, U-Net model seed 0, radar-only V0).

For each chosen event: adapt with each method exactly as ks_eval.py does, keep the adapted model, predict the chip with
the most labelled water, and threshold it at the 20% guarantee threshold calibrated leave-one-event-out on the other
Kuro Siwo events (from the histograms in --hist). The maps are printed to the log (zlib + base64, in chunks) because
notebook outputs cannot be downloaded from this setup; a local script rebuilds the figure from the log.

Usage: python maps.py --data data --ks data_ks --runs runs --hist runs_ks/ks_hist_m0_v2.npz --events 324,147,1111010,118
"""
import argparse
import base64
import inspect
import json
import os
import zlib

import numpy as np

import tta_px1
from conformal_px1 import curves

# run_one returns a histogram; make it return the adapted model instead (exactly one match, checked)
src = inspect.getsource(tta_px1.run_one)
OLD = "    final.eval()\n    h = np.zeros((2, 2, BINS), np.int64)\n"
assert src.count(OLD) == 1, "run_one changed"
src = src.replace(OLD, "    final.eval()\n    return final\n").replace("def run_one(", "def run_one_model(", 1)
exec(compile(src, "run_one_model", "exec"), tta_px1.__dict__)
run_one_model = tta_px1.run_one_model

from ks_eval import events_of, load_ks_event  # noqa: E402

METHODS = ["none", "tent", "adabn", "adabn_physa"]


def guarantee_k(hist, method, event, alpha=0.2):
    evs = sorted({k.split("|")[-1] for k in hist if k.startswith(f"{method}|V0|")})
    cur = {e: curves(hist[f"{method}|V0|{e}"][0]) for e in evs if hist[f"{method}|V0|{e}"][0, 1].sum() > 0}
    others = np.mean([cur[o][0] for o in cur if o != event], axis=0)
    n = len(cur) - 1
    ok = np.where(n / (n + 1) * others + 1 / (n + 1) <= alpha)[0]
    return int(ok.max()) if len(ok) else 0


def enc(a):
    return base64.b64encode(zlib.compress(np.ascontiguousarray(a).tobytes(), 9)).decode()


def main():
    import torch
    from sen1floods11_unet_baseline import channel_stats, read_split
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ks", default="data_ks")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--hist", default="runs_ks/ks_hist_m0_v2.npz")
    ap.add_argument("--events", default="324,147,1111010,118")
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hist = dict(np.load(a.hist))
    mean, std = channel_stats(a.data, read_split(a.data, "train"))
    events = events_of(a.ks)
    ckpt = os.path.join(a.runs, f"V0_seed{a.model_seed}", "model.pt")
    m_t = torch.tensor(mean, device=device)[None, :, None, None]
    s_t = torch.tensor(std, device=device)[None, :, None, None]
    for row, e in enumerate(a.events.split(",")):
        ev = load_ks_event(events[e])
        water = [int((l == 1).sum()) for l in ev["lab"]]
        i = int(np.argmax(water))
        raw, lab = ev["raw"][i], ev["lab"][i]
        out = {"event": e, "chip": os.path.basename(events[e][i]), "shape": list(lab.shape),
               "vv": enc(np.clip((raw[0] + 25) * 10, 0, 250).astype(np.uint8)), "lab": enc(lab.astype(np.uint8)),
               "maps": {}, "k": {}, "iou": {}, "fnr": {}}
        for m in METHODS:
            k = guarantee_k(hist, m, e)
            model = run_one_model(m, "V0", e, ev, mean, std, a, device, ckpt=ckpt)
            with torch.no_grad():
                x = (torch.from_numpy(raw).to(device)[None] - m_t) / s_t
                p = model(x).softmax(1)[0, 1].cpu().numpy()
            pred = p >= k / tta_px1.BINS
            v = lab != 255
            tp = int((pred & (lab == 1)).sum()); fp = int((pred & (lab == 0)).sum()); fn = int((~pred & (lab == 1) & v).sum())
            out["maps"][m] = enc(np.packbits(pred.ravel()))
            out["k"][m], out["iou"][m], out["fnr"][m] = k, tp / max(1, tp + fp + fn), fn / max(1, tp + fn)
            print(f"map {e} {m}: threshold {k / 1000:.3f}, chip IoU {out['iou'][m]:.3f}, miss rate {out['fnr'][m]:.3f}", flush=True)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        js = json.dumps(out)
        parts = [js[j:j + 3000] for j in range(0, len(js), 3000)]
        for j, p in enumerate(parts):
            print(f"MAPROW {row} {j + 1}/{len(parts)} {p}", flush=True)
    print("MAPS_DONE", flush=True)


if __name__ == "__main__":
    main()
