"""
Model-free change-detection reference on Kuro Siwo, using the pre-flood image that the network methods do not use.

Per event (no labels): a pixel is mapped as flood water if it is dark after the flood (VV below the event's adaptive
dark threshold, the same rule as in the physics check) AND its VV dropped by more than a change threshold relative
to the pre-flood image. The change threshold is the Otsu threshold of the VV difference over the event, capped at -3 dB.
Scores are given against flood water only (permanent water ignored) and against all water.

Usage: python ks_change.py --ks data_ks --out out
"""
import argparse
import json
import os

import numpy as np

from ks_eval import events_of
from tta_px1 import VV_CAP, otsu_dark


def score(pred, lab):
    v = lab != 255
    tp = int((pred & (lab == 1)).sum()); fp = int((pred & (lab == 0)).sum()); fn = int((~pred & (lab == 1) & v).sum())
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="data_ks")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    res = {}
    for e, files in events_of(a.ks).items():
        chips = []
        for f in files:
            z = np.load(f)
            if "vv_pre" not in z:
                continue
            vv, pre, m = z["vv"].astype(np.float32), z["vv_pre"].astype(np.float32), z["mask"]
            ok = (z["valid"] != 0) & np.isfinite(vv) & np.isfinite(pre) & (pre > -49.9) & (vv > -49.9)
            allw = np.full(m.shape, 255, np.uint8); allw[m == 0] = 0; allw[(m == 1) | (m == 2)] = 1; allw[~ok] = 255
            flood = allw.copy(); flood[(m == 1) & ok] = 255
            chips.append((np.clip(vv, -50, 5), np.clip(pre, -50, 5), allw, flood))
        if len(chips) < 8:
            continue
        vv_all = np.concatenate([c[0][c[2] != 255].ravel()[::7] for c in chips])
        d_all = np.concatenate([(c[0] - c[1])[c[2] != 255].ravel()[::7] for c in chips])
        t, mu = otsu_dark(vv_all)
        t_dark = min(VV_CAP, (t + mu) / 2)
        t_chg = min(-3.0, otsu_dark(d_all)[0])
        out = {}
        for name, rule in [("dark", lambda c: c[0] < t_dark), ("change", lambda c: (c[0] < t_dark) & (c[0] - c[1] < t_chg))]:
            for lab_name, li in [("all", 2), ("flood", 3)]:
                s = np.sum([score(rule(c), c[li]) for c in chips], axis=0)
                out[f"{name}|{lab_name}"] = [round(s[0] / max(1, s.sum()), 4), round(s[2] / max(1, s[0] + s[2]), 4)]
        res[e] = out
        print(f"event {e} ({len(chips)} chips): dark < {t_dark:.1f} dB, change < {t_chg:.1f} dB; " +
              "; ".join(f"{k} IoU {v[0]:.3f} miss {v[1]:.3f}" for k, v in out.items()), flush=True)
    keys = sorted(next(iter(res.values()))) if res else []
    summ = {k: [round(float(np.mean([r[k][i] for r in res.values()])), 4) for i in (0, 1)] for k in keys}
    print(f"KS_CHANGE mean over {len(res)} events (IoU, miss rate): " + json.dumps(summ), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"events": res, "mean": summ}, open(os.path.join(a.out, "ks_change.json"), "w"))
    print("KS_CHANGE_JSON " + json.dumps({"events": res, "mean": summ}, separators=(",", ":")), flush=True)
    print("KS_CHANGE_DONE", flush=True)


if __name__ == "__main__":
    main()
