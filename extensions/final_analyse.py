"""
Analysis for the last Kaggle run: (1) Kuro Siwo 10% guarantee column for the adaptive variants (three models),
(2) sensitivity of the proposed method to its settings (model seed 0, one change at a time).
Same leave-one-event-out conformal rule as everywhere else. Prints markdown tables and one FINAL_JSON line.

Usage: python final_analyse.py --ks runs_ks --sens runs_sens --out OUT
"""
import argparse
import glob
import json
import os

import numpy as np

from conformal_px1 import curves


def per_event(d, m, alpha=None):
    evs = sorted({k.split("|")[-1] for k in d if k.startswith(f"{m}|V0|")})
    cur = {e: curves(d[f"{m}|V0|{e}"][0]) for e in evs if d[f"{m}|V0|{e}"][0, 1].sum() > 0}
    res = {}
    for e in cur:
        if alpha is None:
            k = 500
        else:
            others = np.mean([cur[o][0] for o in cur if o != e], axis=0)
            n = len(cur) - 1
            ok = np.where(n / (n + 1) * others + 1 / (n + 1) <= alpha)[0]
            k = int(ok.max()) if len(ok) else 0
        res[e] = (cur[e][2][k], cur[e][0][k])
    return res


def summary(ds, m):
    out = {}
    for name, a in [("t0.5", None), ("crc_0.1", 0.1), ("crc_0.2", 0.2), ("crc_0.3", 0.3)]:
        per = [per_event(d, m, a) for d in ds]
        evs = sorted(set.intersection(*[set(p) for p in per]))
        iou = np.mean([np.mean([p[e][0] for p in per]) for e in evs])
        fnr = np.mean([np.mean([p[e][1] for p in per]) for e in evs])
        meet = np.mean([np.sum([p[e][1] <= a for e in evs]) for p in per]) if a else None
        out[name] = {"iou": float(iou), "fnr": float(fnr), "meeting": None if meet is None else float(meet), "n": len(evs)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="runs_ks")
    ap.add_argument("--sens", default="runs_sens")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    res = {"ks": {}, "sens": {}}
    ds = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(a.ks, "ks_hist_m[0-9]_v2.npz")))]
    print(f"Kuro Siwo histograms: {len(ds)} models", flush=True)
    lines = ["| Method | 0.5 | 10% guar. | 20% guar. | 30% guar. | 20%: mean miss rate | 20%: events meeting |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for m in ["none", "tent", "adabn", "physa", "adabn_physa"]:
        use = [d for d in ds if any(k.startswith(f"{m}|V0|") for k in d)]
        if not use:
            continue
        s = summary(use, m)
        res["ks"][m] = dict(s, models=len(use))
        lines.append(f"| {m} ({len(use)} models) | {s['t0.5']['iou']:.3f} | {s['crc_0.1']['iou']:.3f} | {s['crc_0.2']['iou']:.3f} | "
                     f"{s['crc_0.3']['iou']:.3f} | {s['crc_0.2']['fnr']:.3f} | {s['crc_0.2']['meeting']:.1f}/{s['crc_0.2']['n']} |")
    print("\n".join(lines), flush=True)

    base = [d for d in ds[:1]]
    rows = [("default (cap -15/-22 dB, HAND 5 m, 200 steps, conf 0.9/0.1)", base)]
    for cfg in ["cap_m2", "cap_p2", "hand3", "hand10", "steps100", "steps400", "conf80", "conf95"]:
        f = glob.glob(os.path.join(a.sens, cfg, "ks_hist_m0_v2.npz"))
        if f:
            rows.append((cfg, [dict(np.load(f[0]))]))
    lines = ["| Setting | 0.5 | 10% guar. | 20% guar. | 30% guar. | 20%: mean miss rate | 20%: events meeting |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, d in rows:
        if not d or not any(k.startswith("adabn_physa|V0|") for k in d[0]):
            continue
        s = summary(d, "adabn_physa")
        res["sens"][name] = s
        lines.append(f"| {name} | {s['t0.5']['iou']:.3f} | {s['crc_0.1']['iou']:.3f} | {s['crc_0.2']['iou']:.3f} | {s['crc_0.3']['iou']:.3f} | "
                     f"{s['crc_0.2']['fnr']:.3f} | {s['crc_0.2']['meeting']:.0f}/{s['crc_0.2']['n']} |")
    print("\nSensitivity, AdaBN + adaptive physics, Kuro Siwo, model seed 0:\n" + "\n".join(lines), flush=True)
    json.dump(res, open(os.path.join(a.out, "final_analysis.json"), "w"), indent=1)
    print("FINAL_JSON " + json.dumps(res), flush=True)
    print("FINAL_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
