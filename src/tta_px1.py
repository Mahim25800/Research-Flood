"""
PX1 stage 5: label-free adaptation to each unseen flood, combined with the miss-rate guarantee.

For every held-out flood E and variant (V0 radar only, V1 radar + elevation) we take the
leave-one-event-out model that never saw E and adapt it to E's chips WITHOUT labels:

  none   no adaptation (reproduces conformal_px1.py)
  tent   TENT: minimise prediction entropy, update batch-norm parameters only
  st     mean-teacher self-training (the recipe of Chen et al. 2026): EMA teacher makes
         confident pseudo-labels (p > 0.9 water, p < 0.1 dry) on a weak view, the student
         learns them on a strong view (speckle-like noise, +-1.5 dB radiometric shift)
  phys   st + physics-checked pseudo-labels (ours). Rules fixed in advance (same radar
         thresholds as the label audit):
           water pseudo-label dropped if the pixel is radar-bright (VV > -15 dB) or
           physically implausible for ponding (> 15 m above the chip's low ground, or slope > 8 deg)
           a pixel that is radar-dark (VV < -18 and VH < -25 dB), low (< 5 m) and flat (< 3 deg)
           becomes a water pseudo-label unless the teacher is confident it is dry (then ignored)
         Elevation is used here only to check pseudo-labels, so it also helps the radar-only model.

Labels of E are used only for scoring. Every setting is fixed before seeing any result
(200 steps, batch 8, 256 crops, Adam; TENT lr 1e-3, self-training lr 1e-4, EMA 0.99).

Then, as in conformal_px1.py, the threshold for E is calibrated on the OTHER floods, each
adapted the same way by its own held-out model, and scored on E.
A classical physics baseline (Otsu threshold on VV per flood, no model) is reported too.

Ablations (physics check with one rule only): --methods phys_drop,phys_add

Usage:  python tta_px1.py --data data --runs runs_loeo     (resumes from runs_loeo/tta_hist.npz)
        python tta_px1.py --analyse-only
"""
import argparse
import copy
import json
import os
import time

os.environ.setdefault("GDAL_CACHEMAX", "64")

import numpy as np

from conformal_px1 import ALPHAS, BINS, EVENTS, VARIANTS, curves

METHODS = ["none", "tent", "st", "phys"]
ABLATIONS = ["phys_drop", "phys_add"]  # physics check with only the drop rule / only the add rule
COMBOS = ["adabn", "adabn_phys", "tent_phys"]  # added after the first Kuro Siwo results: recalibrate, then physics-checked self-training
ADAPTIVE = ["physa", "adabn_physa"]  # per-flood radar thresholds + HAND gating (add rule only)
ALL_METHODS = METHODS + ABLATIONS + COMBOS + ADAPTIVE
STEPS, BATCH, CROP = 200, 8, 256
LR_TENT, LR_ST, EMA, P_HI, P_LO = 1e-3, 1e-4, 0.99, 0.9, 0.1
ADABN_BATCHES = 50
VV_BRIGHT, VV_DARK, VH_DARK = -15.0, -18.0, -25.0
RELH_HIGH, SLOPE_HIGH, RELH_LOW, SLOPE_LOW = 15.0, 8.0, 5.0, 3.0
# adaptive rule (fixed before any run): per flood, Otsu split of all VV (and VH) pixels; "dark" = below the
# midpoint between the dark-mode mean and the Otsu threshold, capped at -15 dB (VV) / -22 dB (VH) so that a
# flood with little water cannot turn dry land into water; low = HAND < 5 m; flat = slope < 3 deg
HAND_LOW, VV_CAP, VH_CAP = 5.0, -15.0, -22.0


def otsu_dark(values):
    """Otsu threshold of a 1-D sample and the mean of the class below it."""
    v = values[values > -49.9]
    hist, edges = np.histogram(v, bins=256)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    s0 = np.cumsum(hist * c)
    mu0 = s0 / np.maximum(w0, 1); mu1 = (s0[-1] - s0) / np.maximum(w1, 1)
    k = int(np.argmax(w0 * w1 * (mu0 - mu1) ** 2))
    return float(c[k]), float(mu0[k])


def adaptive_dark(raw_list):
    """Per-flood dark thresholds (VV, VH) from the flood's own backscatter, no labels."""
    out = []
    for ch, cap in [(0, VV_CAP), (1, VH_CAP)]:
        v = np.concatenate([r[ch].ravel()[::7] for r in raw_list])
        t, mu = otsu_dark(v)
        out.append(min(cap, (t + mu) / 2))
    return out


def hand_channel(hand, terr):
    """Third terrain channel: HAND / 50 m (falls back to the chip's relative height if HAND is missing)."""
    h = terr[0:1] if hand is None else np.clip(hand[None].astype(np.float32), 0, 100) / 50.0
    return np.concatenate([terr, h], 0).astype(np.float32)


def load_event(args, e, every, clean):
    from sen1floods11_unet_baseline import event_of, load_label, load_s1
    from train_px1 import load_dem, terrain_features
    ev = {"raw": [], "terr": [], "lab": [], "clean": []}
    for s, l in [p for p in every if event_of(p[0]) == e]:
        s1_path = os.path.join(args.data, "S1Hand", s)
        ev["raw"].append(load_s1(s1_path).astype(np.float32))
        hp = os.path.join(args.data, "HAND", s.replace("S1Hand", "HAND"))
        hand = None
        if os.path.exists(hp):
            import rasterio
            with rasterio.open(hp) as src:
                hand = src.read(1)
        ev["hand_ok"] = ev.get("hand_ok", 0) + (hand is not None)
        ev["terr"].append(hand_channel(hand, terrain_features(load_dem(args.data, s, s1_path), s1_path)))
        ev["lab"].append(load_label(os.path.join(args.data, "LabelHand", l)))
        ev["clean"].append(s in clean)
    return ev


def run_one(method, v, e, ev, mean, std, args, device, ckpt=None):
    import torch
    import torch.nn.functional as F
    from sen1floods11_unet_baseline import IGNORE, UNet

    use_dem = VARIANTS[v] == "s1dem"
    torch.manual_seed(args.seed)
    gen = np.random.default_rng(args.seed)
    m_t = torch.tensor(mean, device=device)[None, :, None, None]
    s_t = torch.tensor(std, device=device)[None, :, None, None]
    raw = [torch.from_numpy(r).to(device) for r in ev["raw"]]
    terr = [torch.from_numpy(t).to(device) for t in ev["terr"]]

    def inp(r, t):
        x = (r - m_t) / s_t
        return torch.cat([x, t[:, :2]], 1) if use_dem else x

    model = UNet(cin=4 if use_dem else 2).to(device)
    ckpt = ckpt or os.path.join(args.runs, f"{v}_hold{e}_seed{args.model_seed}", "model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    def crops():
        rs, ts = [], []
        for i in gen.integers(0, len(raw), BATCH):
            H, W = raw[i].shape[1:]
            c = min(CROP, H, W)
            r0, c0 = gen.integers(0, H - c + 1), gen.integers(0, W - c + 1)
            r, t = raw[i][:, r0:r0 + c, c0:c0 + c], terr[i][:, r0:r0 + c, c0:c0 + c]
            k = int(gen.integers(0, 4))
            r, t = torch.rot90(r, k, (1, 2)), torch.rot90(t, k, (1, 2))
            if gen.random() < 0.5:
                r, t = r.flip(2), t.flip(2)
            rs.append(r); ts.append(t)
        return torch.stack(rs), torch.stack(ts)

    # optional first stage: re-estimate batch-norm statistics (adabn) or TENT, then optional self-training
    pre, base = None, method
    if method in ("adabn", "tent"):
        pre, base = method, "none"
    elif method.startswith(("adabn_", "tent_")):
        pre, base = method.split("_", 1)
    if pre == "adabn":
        bns = [mod for mod in model.modules() if isinstance(mod, torch.nn.BatchNorm2d)]
        for b in bns:
            b.reset_running_stats(); b.momentum = None  # cumulative average over the target batches
        model.train()
        with torch.no_grad():
            for _ in range(ADABN_BATCHES):
                r, t = crops()
                model(inp(r, t))
        for b in bns:
            b.momentum = 0.1
    if pre == "tent":
        model.train()
        params = [p for mod in model.modules() if isinstance(mod, torch.nn.BatchNorm2d) for p in (mod.weight, mod.bias)]
        for p in model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=LR_TENT)
        for _ in range(args.steps):
            r, t = crops()
            p = model(inp(r, t)).softmax(1)
            loss = -(p * torch.log(p.clamp_min(1e-6))).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        for p in model.parameters():
            p.requires_grad_(True)
    final = model
    if base in ("st", "phys", "phys_drop", "phys_add", "physa"):
        method = base
        if method == "physa":
            tv, th = adaptive_dark(ev["raw"])
            print(f"    adaptive dark thresholds VV {tv:.1f} dB, VH {th:.1f} dB; HAND for {ev.get('hand_ok', 0)}/{len(raw)} chips", flush=True)
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=LR_ST)
        ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device=device), ignore_index=IGNORE)
        kept = []
        for _ in range(args.steps):
            r, t = crops()
            with torch.no_grad():
                pt = teacher(inp(r, t)).softmax(1)[:, 1]
                y = torch.full_like(pt, IGNORE, dtype=torch.long)
                y[pt > P_HI] = 1
                y[pt < P_LO] = 0
                relh, slope = t[:, 0] * 50, t[:, 1] * 10
                if method in ("phys", "phys_drop"):
                    implausible = (r[:, 0] > VV_BRIGHT) | (relh > RELH_HIGH) | (slope > SLOPE_HIGH)
                    y[(y == 1) & implausible] = IGNORE
                if method == "physa":
                    hand = t[:, 2] * 50
                    dark_low = (r[:, 0] < tv) & (r[:, 1] < th) & (hand < HAND_LOW) & (slope < SLOPE_LOW)
                    y[dark_low & (pt >= P_LO)] = 1
                    y[dark_low & (pt < P_LO)] = IGNORE
                if method in ("phys", "phys_add"):
                    dark_low = (r[:, 0] < VV_DARK) & (r[:, 1] < VH_DARK) & (relh < RELH_LOW) & (slope < SLOPE_LOW)
                    y[dark_low & (pt >= P_LO)] = 1
                    y[dark_low & (pt < P_LO)] = IGNORE
                kept.append(float((y != IGNORE).float().mean()))
                off = (torch.rand(r.shape[0], 1, 1, 1, device=device) * 3 - 1.5)
            xs = inp(r + off, t)
            xs = torch.cat([xs[:, :2] + 0.1 * torch.randn_like(xs[:, :2]), xs[:, 2:]], 1)
            loss = ce(model(xs), y)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for pt_, ps_ in zip(teacher.parameters(), model.parameters()):
                    pt_.mul_(EMA).add_(ps_.detach(), alpha=1 - EMA)
                for bt, bs in zip(teacher.buffers(), model.buffers()):
                    if bt.dtype.is_floating_point:
                        bt.mul_(EMA).add_(bs, alpha=1 - EMA)
                    else:
                        bt.copy_(bs)
        print(f"    pseudo-labelled share of pixels {np.mean(kept):.3f}", flush=True)
        final = teacher

    final.eval()
    h = np.zeros((2, 2, BINS), np.int64)
    with torch.no_grad():
        for i in range(len(raw)):
            p = final(inp(raw[i][None], terr[i][None])).softmax(1)[0, 1]
            b = (p * BINS).long().clamp(0, BINS - 1)
            lab = torch.from_numpy(ev["lab"][i].astype(np.int64)).to(device)
            for sub in ([0, 1] if ev["clean"][i] else [0]):
                for c in (0, 1):
                    h[sub, c] += torch.bincount(b[lab == c], minlength=BINS).cpu().numpy()
    return h


def otsu_counts(ev):
    """Classical physics baseline: one Otsu threshold on VV per flood; water = VV below it."""
    vv = np.concatenate([r[0].ravel() for r in ev["raw"]])
    hist, edges = np.histogram(vv[vv > -49.9], bins=256)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    mu0 = np.cumsum(hist * c) / np.maximum(w0, 1)
    mu1 = (np.sum(hist * c) - np.cumsum(hist * c)) / np.maximum(w1, 1)
    thr = c[np.argmax(w0 * w1 * (mu0 - mu1) ** 2)]
    out = np.zeros((2, 3), np.int64)  # [all/clean][tp, fp, fn]
    for r, lab, cl in zip(ev["raw"], ev["lab"], ev["clean"]):
        pred = r[0] < thr
        cnt = [np.sum(pred & (lab == 1)), np.sum(pred & (lab == 0)), np.sum(~pred & (lab == 1))]
        for sub in ([0, 1] if cl else [0]):
            out[sub] += cnt
    return out, float(thr)


def save(path, d):
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **d)
    os.replace(tmp, path)


def phase1(args):
    import torch
    from sen1floods11_unet_baseline import channel_stats, event_of, read_split
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = os.path.join(args.runs, f"tta_hist{tag(args)}.npz")
    done = dict(np.load(path)) if os.path.exists(path) else {}
    with open(os.path.join("audit", "clean_chips.txt")) as f:
        clean = {ln.strip() for ln in f if ln.strip()}
    splits = {k: read_split(args.data, k) for k in ["train", "valid", "test", "bolivia"]}
    every = splits["train"] + splits["valid"] + splits["test"] + splits["bolivia"]
    for e in EVENTS:
        todo = [(m, v) for v in VARIANTS for m in args.methods
                if f"{m}|{v}|{e}" not in done and os.path.exists(os.path.join(args.runs, f"{v}_hold{e}_seed{args.model_seed}", "model.pt"))]
        if not todo and f"otsu|{e}" in done:
            continue
        ev = load_event(args, e, every, clean)
        if not ev["raw"]:
            continue
        mean, std = channel_stats(args.data, [p for p in splits["train"] if event_of(p[0]) != e])
        if f"otsu|{e}" not in done:
            done[f"otsu|{e}"], thr = otsu_counts(ev)
            print(f"{e}: Otsu VV threshold {thr:.1f} dB", flush=True)
        for m, v in todo:
            t0 = time.time()
            h = run_one(m, v, e, ev, mean, std, args, device)
            done[f"{m}|{v}|{e}"] = h
            fnr, _, iou = curves(h[0])
            print(f"{m:9s} {v} hold {e:9s}: miss rate at 0.5 {fnr[BINS // 2]:.3f}, IoU {iou[BINS // 2]:.3f}, "
                  f"{time.time() - t0:.0f}s", flush=True)
            save(path, done)
        save(path, done)
        del ev
        if device == "cuda":
            torch.cuda.empty_cache()


def tag(args):
    return "" if args.seed == 0 and args.model_seed == 0 else f"_m{args.model_seed}_a{args.seed}"


def compute(data, events=None):
    events = events or EVENTS
    out = {}
    k05 = BINS // 2
    for v in VARIANTS:
        for sub, sub_name in [(0, "all"), (1, "clean")]:
            block = {}
            for m in [m for m in ALL_METHODS if any(k.startswith(m + "|") for k in data)]:
                cur = {e: curves(data[f"{m}|{v}|{e}"][sub]) for e in events
                       if f"{m}|{v}|{e}" in data and data[f"{m}|{v}|{e}"][sub, 1].sum() > 0}
                if len(cur) < 2:
                    continue
                per = {}
                for e in cur:
                    others = np.mean([cur[o][0] for o in cur if o != e], axis=0)
                    n = len(cur) - 1
                    fnr, prec, iou = cur[e]
                    per[e] = {"default": {"fnr": fnr[k05], "iou": iou[k05], "precision": prec[k05]}}
                    for a in ALPHAS:
                        ok = np.where(n / (n + 1) * others + 1 / (n + 1) <= a)[0]
                        k = int(ok.max()) if len(ok) else 0
                        per[e][f"crc_{a}"] = {"threshold": k / BINS, "fnr": fnr[k], "iou": iou[k], "precision": prec[k]}
                summ = {}
                for rule in ["default"] + [f"crc_{a}" for a in ALPHAS]:
                    f = np.array([per[e][rule]["fnr"] for e in per])
                    a = None if rule == "default" else float(rule.split("_")[1])
                    summ[rule] = {"mean_fnr": float(f.mean()), "max_fnr": float(f.max()), "sd_fnr": float(f.std()),
                                  "meeting": None if a is None else int((f <= a).sum()),
                                  "mean_iou": float(np.mean([per[e][rule]["iou"] for e in per]))}
                block[m] = {"n": len(per), "summary": summ,
                            "events": {e: {r: {k: round(float(x), 4) for k, x in d.items()} for r, d in pe.items()} for e, pe in per.items()}}
            if True:
                ot = [data[f"otsu|{e}"][sub] for e in events if f"otsu|{e}" in data]
                if ot:
                    ious = [c[0] / max(1, c.sum()) for c in ot]
                    fnrs = [c[2] / max(1, c[0] + c[2]) for c in ot]
                    block["otsu"] = {"n": len(ot), "summary": {"default": {"mean_fnr": float(np.mean(fnrs)), "max_fnr": float(np.max(fnrs)),
                                     "sd_fnr": float(np.std(fnrs)), "meeting": None, "mean_iou": float(np.mean(ious))}}}
            out[f"{v}|{sub_name}"] = block
    return out


def analyse(args):
    data = dict(np.load(os.path.join(args.runs, f"tta_hist{tag(args)}.npz")))
    report(compute(data), os.path.join(args.runs, f"tta_results{tag(args)}.json"),
           os.path.join(args.runs, f"tta_summary{tag(args)}.md"))


def report(out, json_path, md_path):
    lines = []
    for key, block in out.items():
            v, sub_name = key.split("|")
            lines += [f"### {v}, {sub_name} chips", "",
                      "| Adaptation | Rule | Mean miss rate | Worst flood | SD across floods | Floods meeting target | Mean IoU |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for m, b in block.items():
                for rule, d in b["summary"].items():
                    meet = "-" if d["meeting"] is None else f"{d['meeting']}/{b['n']}"
                    lines.append(f"| {m} | {rule} | {d['mean_fnr']:.3f} | {d['max_fnr']:.3f} | {d['sd_fnr']:.3f} | {meet} | {d['mean_iou']:.3f} |")
            lines.append("")
            if "phys" in block and "none" in block:
                lines += [f"Per flood at the 20% target ({v}, {sub_name}): miss rate / IoU", "",
                          "| Flood | " + " | ".join(m for m in ALL_METHODS if m in block) + " |",
                          "| --- |" + " --- |" * len([m for m in ALL_METHODS if m in block])]
                for e in block["none"]["events"]:
                    cells = []
                    for m in ALL_METHODS:
                        if m in block and e in block[m]["events"]:
                            d = block[m]["events"][e]["crc_0.2"]
                            cells.append(f"{d['fnr']:.3f} / {d['iou']:.3f}")
                        elif m in block:
                            cells.append("-")
                    lines.append(f"| {e} | " + " | ".join(cells) + " |")
                lines.append("")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=1)
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("TTA_DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", default="runs_loeo")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seed", type=int, default=0, help="adaptation seed")
    ap.add_argument("--model-seed", type=int, default=0, help="seed of the leave-one-event-out models to adapt")
    ap.add_argument("--methods", default=",".join(METHODS), help="comma-separated, from " + ",".join(ALL_METHODS))
    ap.add_argument("--no-analyse", action="store_true")
    a = ap.parse_args()
    a.methods = a.methods.split(",")
    if not a.analyse_only:
        phase1(a)
    if not a.no_analyse:
        analyse(a)
