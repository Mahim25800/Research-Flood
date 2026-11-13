"""
Revision analysis from stored probability histograms (no GPU).

For every dataset (Sen1Floods11 = S1F, Kuro Siwo = KS, Kuro Siwo scored on flood water only = KSf, MMFlood
activations = MM), method and model, each flood is thresholded under several rules:

  t0.5            fixed threshold 0.5
  loeo_A          conformal risk control (mean miss rate <= A), calibrated on the other floods of the same dataset
  x_S1F_A         the same rule calibrated ONLY on the 11 Sen1Floods11 floods (no label from the target dataset)
  x_S1FKS_A       MMFlood only: calibrated on Sen1Floods11 + Kuro Siwo floods
  q_A_D           per-flood rule: threshold = conformal quantile of the per-flood thresholds, so that a new flood
                  misses at most A of its water with probability >= 1 - D (same-dataset calibration)
  xq_S1F_A_D      per-flood rule calibrated only on Sen1Floods11

Prints markdown tables, paired tests and one compact JSON (REV_JSON lines) used for the figures and tables.

Usage: python rev_analyse.py --root /kaggle/input --extra /kaggle/working --out OUT
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

BINS = 1000
ALPHAS = [0.1, 0.2, 0.3]
QRULES = [(0.2, 0.2), (0.2, 0.1), (0.3, 0.2)]
ORDER = ["none", "st", "tent", "tent_batch", "eata", "sar", "cotta", "adabn", "phys", "physa", "adabn_phys", "adabn_physa", "ft"]
GRID = [round(x, 2) for x in np.arange(0.04, 0.501, 0.02)]


def curves(h):
    dry, wat = h[0].astype(float), h[1].astype(float)
    wat_pos = np.cumsum(wat[::-1])[::-1]
    dry_pos = np.cumsum(dry[::-1])[::-1]
    W = wat.sum()
    tp, fp, fn = wat_pos, dry_pos, W - wat_pos
    fnr = fn / max(W, 1)
    iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)
    prec = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
    return fnr, iou, prec, fp / max(dry.sum(), 1)


def k_crc(cal, alpha):
    n = len(cal)
    ok = np.where(n / (n + 1) * np.mean(cal, axis=0) + 1 / (n + 1) <= alpha)[0]
    return int(ok.max()) if len(ok) else 0


def k_quant(cal, alpha, delta):
    ks = sorted(int(np.where(f <= alpha)[0].max()) for f in cal)
    j = int(np.floor(delta * (len(ks) + 1)))
    return ks[j - 1] if j >= 1 else 0


def find(roots, pattern, exclude=()):
    out = []
    for r in roots:
        out += [p for p in glob.glob(os.path.join(r, "**", pattern), recursive=True) if not any(x in p for x in exclude)]
    return sorted(set(out))


def load(roots):
    sets, src = {}, {}
    f = find(roots, "tta_hist.npz") + find(roots, "tta_hist_local.npz")
    f = sorted(f, key=lambda p: ("pc_hist" not in p and "_local" not in p, p))  # PC first, Kaggle overrides
    d = {}
    for p in f:
        d.update({k: v for k, v in np.load(p).items() if k.count("|") == 2})
    if d:
        sets["S1F"] = [d]; src["S1F"] = f
    ks, ksf = [], []
    for s in range(3):
        f = find(roots, f"ks_hist_m{s}_v2*.npz", exclude=("r34", "sens", "r50", "_ft"))
        f = sorted(f, key=lambda p: ("pc_hist" not in p and "_local" not in p, "rev" in p, p))
        d = {}
        for p in f:
            for k, v in np.load(p).items():
                if k.count("|") != 2:
                    continue
                if k in d and "rev" in p:  # keep the stored scores; the re-run only adds the flood-water-only slot
                    v = np.stack([d[k][0], v[1]])
                d[k] = v
        for p in find(roots, f"ks_hist_m{s}_v2.npz", exclude=("r34", "sens", "r50", "rev_p")):
            if "_ft" in p:
                d.update({k: v for k, v in np.load(p).items() if k.startswith("ft|")})
        if d:
            ks.append({k: v for k, v in d.items() if k.split("|")[1] == "V0"})
            alt = {k: np.stack([v[1], v[1]]) for k, v in ks[-1].items() if v[1, 1].sum() > 0}
            if alt:
                ksf.append(alt)
            src[f"KS{s}"] = f
    if ks:
        sets["KS"] = ks
    if ksf:
        sets["KSf"] = ksf
    mm = []
    for s in range(3):
        f, seen, d = find(roots, f"mm_hist_m{s}_p*.npz", exclude=("r34", "r50", "_act")), set(), {}
        for p in f:
            for k, v in np.load(p).items():
                if k.count("|") != 2:
                    continue
                if (os.path.basename(p), k) in seen:
                    # the same file found twice (copied into two outputs) or re-run with more methods
                    continue
                seen.add((os.path.basename(p), k))
                parts = k.split("|"); parts[-1] = parts[-1].split("-")[0]
                kk = "|".join(parts)
                d[kk] = d[kk] + v if kk in d else v.copy()
        if d:
            mm.append(d); src[f"MM{s}"] = f
    if mm:
        sets["MM"] = mm
    return sets, src


def load_otsu(roots, sets):
    """Per flood: counts of the model-free Otsu split on VV [tp, fp, fn] and the number of labelled pixels.
    This is darkness restricted to label-valid pixels, NOT a fully label-free
    deployment statistic: both the support mask and denominator use labels."""
    out = {}
    pats = {"S1F": ("tta_hist*.npz", ()), "KS": ("ks_hist_m0_v2*.npz", ("r34", "sens", "r50", "_ft")),
            "MM": ("mm_hist_m0_p*.npz", ("r34", "r50", "_act"))}
    for D, (pat, excl) in pats.items():
        if D not in sets:
            continue
        seen, acc = set(), {}
        for p in find(roots, pat, exclude=excl):
            for k, v in np.load(p).items():
                if not k.startswith("otsu|"):
                    continue
                tag = (os.path.basename(p), k) if D == "MM" else k
                if tag in seen:
                    continue
                seen.add(tag)
                e = k.split("|")[1]
                e = e.split("-")[0] if D == "MM" else e
                acc[e] = acc.get(e, 0) + np.asarray(v)[0].astype(np.int64)
        d0 = sets[D][0]
        out[D] = {}
        for e, c in acc.items():
            h = d0.get(f"none|V0|{e}")
            if h is not None and h[0].sum() > 0:
                out[D][e] = [int(c[0]), int(c[1]), int(c[2]), int(h[0].sum())]
    return out


def per_model(d, m):
    return {k.split("|")[2]: curves(v[0]) for k, v in d.items() if k.startswith(m + "|V0|") and v[0, 1].sum() > 0}


def evaluate(sets):
    """res[D][m][rule] = {event: (iou, fnr)} averaged over models."""
    res = {}
    for D, models in sets.items():
        res[D] = {}
        methods = [m for m in ORDER if any(k.startswith(m + "|V0|") for k in models[0])]
        for m in methods:
            acc = {}
            use = [d for d in models if any(k.startswith(m + "|V0|") for k in d)]
            use = [d for d in use if len(per_model(d, m)) >= 3]
            if not use:
                continue
            for si, d in enumerate(use):
                cur = per_model(d, m)
                s1f = list(per_model(sets["S1F"][0], m).values()) if D != "S1F" and "S1F" in sets else []
                kscal = []
                if D == "MM" and "KS" in sets:
                    kk = [x for x in sets["KS"] if any(k.startswith(m + "|V0|") for k in x)]
                    if kk:
                        kscal = list(per_model(kk[min(si, len(kk) - 1)], m).values())
                rules = {}
                for e, (fnr, iou, prec, far) in cur.items():
                    others = [cur[o][0] for o in cur if o != e]
                    r = {"t0.5": BINS // 2}
                    for a in ALPHAS:
                        r[f"loeo_{a}"] = k_crc(others, a)
                        if s1f:
                            r[f"x_S1F_{a}"] = k_crc([c[0] for c in s1f], a)
                        if s1f and kscal:
                            r[f"x_S1FKS_{a}"] = k_crc([c[0] for c in s1f] + [c[0] for c in kscal], a)
                    for a, dl in QRULES:
                        r[f"q_{a}_{dl}"] = k_quant(others, a, dl)
                        if s1f:
                            r[f"xq_S1F_{a}_{dl}"] = k_quant([c[0] for c in s1f], a, dl)
                        if s1f and kscal:
                            r[f"xq_S1FKS_{a}_{dl}"] = k_quant([c[0] for c in s1f] + [c[0] for c in kscal], a, dl)
                    for g in GRID:
                        r[f"g_{g}"] = k_crc(others, g)
                    for name, k in r.items():
                        acc.setdefault(name, {}).setdefault(e, []).append((iou[k], fnr[k], k / BINS, prec[k], far[k]))
            n_models = len(use)
            res[D][m] = {name: {e: tuple(np.mean(v, axis=0)) for e, v in ev.items() if len(v) == n_models} for name, ev in acc.items()}
            res[D][m]["_models"] = n_models
            # share of floods meeting the target, per model then averaged
            res[D][m]["_meet"] = {}
            for name, ev in acc.items():
                if name.startswith("g_") or name == "t0.5":
                    continue
                a = float(name.split("_")[-2] if name.startswith(("q_", "xq_")) else name.split("_")[-1])
                res[D][m]["_meet"][name] = float(np.mean([np.sum([v[i][1] <= a + 1e-12 for v in ev.values() if len(v) == n_models]) for i in range(n_models)]))
    return res


def holm(ps):
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kaggle/input")
    ap.add_argument("--extra", default="")
    ap.add_argument("--extra2", default="")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    roots = [a.root] + [x for x in (a.extra, a.extra2) if x]
    sets, src = load(roots)
    for k, v in src.items():
        print(f"{k}: {len(v)} files: " + "; ".join(p.replace(a.root, "")[-70:] for p in v), flush=True)
    for D, models in sets.items():
        print(f"{D}: {len(models)} models; methods " + ", ".join(sorted({k.split('|')[0] for d in models for k in d})), flush=True)
    res = evaluate(sets)
    out = {"summary": {}, "tests": {}, "curves": {}, "events": {}, "otsu": load_otsu(roots, sets)}
    for D, v in out["otsu"].items():
        print(f"{D}: Otsu counts for {len(v)} floods", flush=True)
    L = []
    for D in res:
        rules = [r for r in ["t0.5"] + [f"loeo_{x}" for x in ALPHAS] + [f"x_S1F_{x}" for x in ALPHAS] + [f"x_S1FKS_{x}" for x in ALPHAS]
                 + [f"{p}_{x}_{d}" for p in ("q", "xq_S1F", "xq_S1FKS") for x, d in QRULES]
                 if any(r in res[D][m] for m in res[D])]
        L += [f"\n### {D}: mean IoU / mean miss rate / max miss rate / floods meeting the target", "",
              "| Method (models, floods) | " + " | ".join(rules) + " |", "| --- |" + " --- |" * len(rules)]
        out["summary"][D] = {}
        for m in res[D]:
            cells, out["summary"][D][m] = [], {"models": res[D][m]["_models"]}
            n = 0
            for r in rules:
                ev = res[D][m].get(r)
                if not ev:
                    cells.append("-"); continue
                v = np.array(list(ev.values())); n = len(v)
                meet = res[D][m]["_meet"].get(r)
                out["summary"][D][m][r] = [round(float(v[:, 0].mean()), 5), round(float(v[:, 1].mean()), 5), round(float(v[:, 1].max()), 5),
                                           None if meet is None else round(meet, 2), n, round(float(v[:, 2].mean()), 5),
                                           round(float(v[:, 3].mean()), 5), round(float(v[:, 4].mean()), 5)]
                cells.append(f"{v[:, 0].mean():.3f} / {v[:, 1].mean():.3f} / {v[:, 1].max():.2f}" + ("" if meet is None else f" / {meet:.1f}"))
            L.append(f"| {m} ({res[D][m]['_models']}, {n}) | " + " | ".join(cells) + " |")
        out["curves"][D] = {m: [[round(float(np.mean([v[i] for v in res[D][m][f"g_{g}"].values()])), 5) for g in GRID] for i in (0, 1)] for m in res[D]}
        for m in res[D]:
            keep = [r for r in ["t0.5", "loeo_0.2", "x_S1F_0.2", "q_0.2_0.2"] if r in res[D][m]]
            out["events"].setdefault(D, {})[m] = {r: {e: [round(float(x), 3) for x in v[:2]] for e, v in res[D][m][r].items()} for r in keep}
        # paired tests
        pairs = [("adabn_physa", o) for o in res[D] if o != "adabn_physa"] + [("none", o) for o in ("tent", "eata", "sar", "cotta", "adabn") if o in res[D]]
        if "adabn_physa" not in res[D]:
            continue
        for r in [x for x in ["t0.5", "loeo_0.2", "x_S1F_0.2", "x_S1FKS_0.2", "q_0.2_0.2"] if x in res[D]["adabn_physa"]]:
            rows = []
            for p, o in pairs:
                if r not in res[D].get(p, {}) or r not in res[D].get(o, {}):
                    continue
                evs = sorted(set(res[D][p][r]) & set(res[D][o][r]))
                dlt = np.array([res[D][p][r][e][0] - res[D][o][r][e][0] for e in evs])
                try:
                    pv = float(wilcoxon(dlt).pvalue)
                except Exception:
                    pv = float("nan")
                rows.append([p, o, float(dlt.mean()), int((dlt > 0).sum()), len(dlt), pv])
            if not rows:
                continue
            adj = holm(np.nan_to_num(np.array([x[5] for x in rows]), nan=1.0))
            L += [f"\n{D}, rule {r}: paired IoU differences over floods (two-sided Wilcoxon, Holm over the {len(rows)} tests)", "",
                  "| A | B | mean A - B | A better on | p | p (Holm) |", "| --- | --- | --- | --- | --- | --- |"]
            for x, pa in zip(rows, adj):
                L.append(f"| {x[0]} | {x[1]} | {x[2]:+.3f} | {x[3]}/{x[4]} | {x[5]:.2g} | {pa:.2g} |")
            out["tests"].setdefault(D, {})[r] = [[x[0], x[1], round(x[2], 5), x[3], x[4], float(f"{x[5]:.3g}"), float(f"{pa:.3g}")] for x, pa in zip(rows, adj)]
    txt = "\n".join(L)
    print(txt, flush=True)
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "rev_analysis.md"), "w").write(txt)
    js = json.dumps(out, separators=(",", ":"))
    open(os.path.join(a.out, "rev_analysis.json"), "w").write(js)
    parts = [js[i:i + 4000] for i in range(0, len(js), 4000)]
    for i, p in enumerate(parts):
        print(f"REV_JSON {i + 1}/{len(parts)} {p}", flush=True)
    print("REV_ANALYSIS_DONE", flush=True)


if __name__ == "__main__":
    main()
