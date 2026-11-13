"""
PX1 stage 4: a per-flood guarantee on missed flooding.

Phase 1 (GPU): for every held-out flood E and model variant (V0 radar only, V1 radar +
elevation), load the leave-one-event-out model that never saw E, predict water
probabilities on every chip of E, and store histograms of the probability for
labelled-water and labelled-dry pixels (all chips, and clean chips only).

Phase 2: for each E, pick a water threshold using only the OTHER floods' curves, each
measured by its own held-out model (so every calibration curve is an "unseen flood"
curve), then score that threshold on E.
  default  threshold 0.5 (argmax)
  naive    largest threshold whose mean calibration miss rate is <= alpha
  crc      conformal risk control (Angelopoulos et al.): largest t with
           n/(n+1) * mean_FNR(t) + 1/(n+1) <= alpha     (loss bounded by 1)
CRC guarantees that the expected miss rate on a new, exchangeable flood is <= alpha.
Miss rate (FNR) = labelled-water pixels predicted dry / labelled-water pixels.

Usage:  python conformal_px1.py --data data --runs runs_loeo
        python conformal_px1.py --analyse-only   (reuse runs_loeo/conformal_hist.npz)
"""
import argparse
import json
import os

import numpy as np

BINS = 1000
ALPHAS = [0.1, 0.2, 0.3]
EVENTS = ["Bolivia", "Ghana", "India", "Mekong", "Nigeria", "Pakistan", "Paraguay",
          "Somalia", "Spain", "Sri-Lanka", "USA"]
VARIANTS = {"V0": "s1", "V1": "s1dem"}


def phase1(args):
    import torch
    from sen1floods11_unet_baseline import IGNORE, UNet, channel_stats, event_of, read_split
    from train_px1 import Chips

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join("audit", "clean_chips.txt")) as f:
        clean = {ln.strip() for ln in f if ln.strip()}
    splits = {k: read_split(args.data, k) for k in ["train", "valid", "test", "bolivia"]}
    every = splits["train"] + splits["valid"] + splits["test"] + splits["bolivia"]
    hist = {}
    for v, inputs in VARIANTS.items():
        use_dem = inputs == "s1dem"
        for e in EVENTS:
            ckpt = os.path.join(args.runs, f"{v}_hold{e}_seed0", "model.pt")
            if not os.path.exists(ckpt):
                print(f"missing {ckpt}", flush=True)
                continue
            train = [p for p in splits["train"] if event_of(p[0]) != e]
            mean, std = channel_stats(args.data, train)  # same stats the model was trained with
            model = UNet(cin=4 if use_dem else 2).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            ds = Chips(args.data, [p for p in every if event_of(p[0]) == e], mean, std, False, use_dem)
            h = np.zeros((2, 2, BINS), np.int64)  # [subset all/clean][label dry/water][bin]
            with torch.no_grad():
                for i in range(len(ds)):
                    x, y, _, s = ds[i]
                    p = model(x[None].to(device)).softmax(1)[0, 1].cpu().numpy()
                    t = y.numpy()
                    m = t != IGNORE
                    b = np.clip((p * BINS).astype(int), 0, BINS - 1)
                    for sub in ([0, 1] if s in clean else [0]):
                        for lab in (0, 1):
                            h[sub, lab] += np.bincount(b[m & (t == lab)], minlength=BINS)
            hist[f"{v}|{e}"] = h
            print(f"{v} hold {e}: {len(ds)} chips, water px {h[0, 1].sum()}", flush=True)
    np.savez_compressed(os.path.join(args.runs, "conformal_hist.npz"), **hist)


def curves(h):
    """For thresholds t_k = k/BINS: predicted water iff p >= t_k. Returns FNR, precision, IoU."""
    dry, wat = h[0].astype(float), h[1].astype(float)
    # pixels with bin >= k are predicted water at threshold k/BINS
    wat_pos = np.cumsum(wat[::-1])[::-1]
    dry_pos = np.cumsum(dry[::-1])[::-1]
    W = wat.sum()
    tp, fp, fn = wat_pos, dry_pos, W - wat_pos
    fnr = fn / max(W, 1)
    prec = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
    iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)
    return fnr, prec, iou


def phase2(args):
    data = np.load(os.path.join(args.runs, "conformal_hist.npz"))
    out = {}
    for v in VARIANTS:
        for sub, sub_name in [(0, "all"), (1, "clean")]:
            cur = {}
            for e in EVENTS:
                key = f"{v}|{e}"
                if key in data and data[key][sub, 1].sum() > 0:
                    cur[e] = curves(data[key][sub])
            rows = []
            for e in cur:
                others = [cur[o][0] for o in cur if o != e]
                n = len(others)
                mean_fnr = np.mean(others, axis=0)  # mean calibration miss rate per threshold
                fnr, prec, iou = cur[e]
                k05 = BINS // 2
                row = {"event": e, "default": {"fnr": fnr[k05], "precision": prec[k05], "iou": iou[k05]}}
                for a in ALPHAS:
                    ok_naive = np.where(mean_fnr <= a)[0]
                    ok_crc = np.where(n / (n + 1) * mean_fnr + 1 / (n + 1) <= a)[0]
                    for name, ok in [("naive", ok_naive), ("crc", ok_crc)]:
                        k = int(ok.max()) if len(ok) else 0  # largest threshold meeting the bound
                        row[f"{name}_{a}"] = {"threshold": k / BINS, "fnr": fnr[k], "precision": prec[k], "iou": iou[k]}
                rows.append(row)
            summ = {"n_events": len(rows)}
            for m in ["default"] + [f"{nm}_{a}" for a in ALPHAS for nm in ["naive", "crc"]]:
                f = np.array([r[m]["fnr"] for r in rows])
                target = None if m == "default" else float(m.split("_")[1])
                summ[m] = {"mean_fnr": round(float(f.mean()), 4), "max_fnr": round(float(f.max()), 4),
                           "events_meeting_target": None if target is None else int((f <= target).sum()),
                           "mean_precision": round(float(np.mean([r[m]["precision"] for r in rows])), 4),
                           "mean_iou": round(float(np.mean([r[m]["iou"] for r in rows])), 4)}
            for r in rows:
                for m, d in r.items():
                    if isinstance(d, dict):
                        r[m] = {k: round(float(x), 4) for k, x in d.items()}
            out[f"{v}|{sub_name}"] = {"summary": summ, "events": rows}
    with open(os.path.join(args.runs, "conformal_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    lines = []
    for key, res in out.items():
        s = res["summary"]
        lines += [f"### {key} ({s['n_events']} held-out floods)", "",
                  "| Threshold rule | Mean miss rate | Worst flood | Floods meeting target | Mean precision | Mean IoU |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for m in ["default"] + [f"{nm}_{a}" for a in ALPHAS for nm in ["naive", "crc"]]:
            d = s[m]
            meet = "-" if d["events_meeting_target"] is None else f"{d['events_meeting_target']}/{s['n_events']}"
            lines.append(f"| {m} | {d['mean_fnr']:.3f} | {d['max_fnr']:.3f} | {meet} | {d['mean_precision']:.3f} | {d['mean_iou']:.3f} |")
        lines.append("")
    with open(os.path.join(args.runs, "conformal_summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("CONFORMAL_DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", default="runs_loeo")
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    if not a.analyse_only:
        phase1(a)
    phase2(a)
