import os, sys, glob, time, shutil, hashlib, subprocess
py = sys.executable
W, OUT, ST = '/tmp/px1', '/kaggle/working/out', '/tmp/px1_stage'
for d in [OUT + '/logs', ST]:
    os.makedirs(d, exist_ok=True)
FILES = {}
FILES['r34_patch.py'] = r'''import os
import sys
BUILD = '\n\ndef build_model(cin=2, pretrained=False):\n    """PX1_ARCH=unet (default): the original U-Net. PX1_ARCH=r34: U-Net with a ResNet-34 encoder (ImageNet weights when training)."""\n    arch = os.environ.get("PX1_ARCH", "unet")\n    if arch == "unet":\n        return UNet(cin=cin)\n    import segmentation_models_pytorch as smp\n    enc = {"r34": "resnet34", "r50": "resnet50"}[arch]\n    return smp.Unet(enc, encoder_weights="imagenet" if pretrained else None, in_channels=cin, classes=2)\n\n'
PATCHES = {'sen1floods11_unet_baseline.py': [('\n\n# ---------------------------------------------------------------- evaluation\n', BUILD + '\n# ---------------------------------------------------------------- evaluation\n')], 'train_px1.py': [('from sen1floods11_unet_baseline import IGNORE, UNet, channel_stats,', 'from sen1floods11_unet_baseline import IGNORE, UNet, build_model, channel_stats,'), ('    model = UNet(cin=4 if use_dem else 2).to(device)\n', '    model = build_model(cin=4 if use_dem else 2, pretrained=True).to(device)\n')], 'tta_px1.py': [('    from sen1floods11_unet_baseline import IGNORE, UNet\n', '    from sen1floods11_unet_baseline import IGNORE, UNet, build_model\n'), ('    model = UNet(cin=4 if use_dem else 2).to(device)\n', '    model = build_model(cin=4 if use_dem else 2).to(device)\n')], 'ks_eval.py': [('    events = events_of(args.ks)\n', '    events = dict(list(events_of(args.ks).items())[args.part::args.nparts])\n'), ('        todo = [(m, v) for v in VARIANTS for m in args.methods', '        todo = [(m, v) for v in args.variants for m in args.methods'), ('    a.methods = a.methods.split(",")\n', '    a.methods = a.methods.split(",")\n    a.variants = a.variants.split(",")\n'), ('    ap.add_argument("--analyse-only", action="store_true")\n', '    ap.add_argument("--analyse-only", action="store_true")\n    ap.add_argument("--variants", default="V0,V1")\n    ap.add_argument("--part", type=int, default=0)\n    ap.add_argument("--nparts", type=int, default=1)\n')]}

def main():
    code, check = (sys.argv[1], '--check' in sys.argv)
    for f, ps in PATCHES.items():
        p = os.path.join(code, f)
        s = open(p).read()
        for old, new in ps:
            n = s.count(old)
            if n != 1:
                raise SystemExit(f'patch mismatch in {f}: {old.strip()[:60]!r} found {n} times')
            s = s.replace(old, new)
        if not check:
            open(p, 'w').write(s)
        print(f"{('checked' if check else 'patched')} {f} ({len(ps)} changes)", flush=True)
    print('PATCH_OK', flush=True)
if __name__ == '__main__':
    main()
'''
FILES['sens_eval.py'] = r'''"""
Sensitivity of the proposed method (AdaBN + adaptive physics-checked self-training) to its settings, on Kuro Siwo.

One setting is changed at a time from the defaults fixed before the MMFlood test:
  cap   : dark-threshold caps VV/VH (default -15/-22 dB)          -> -13/-20 and -17/-24
  hand  : HAND limit for the add rule (default 5 m)               -> 3 and 10
  steps : self-training steps (default 200)                       -> 100 and 400 (passed as --steps)
  conf  : teacher confidence thresholds P_HI/P_LO (0.9/0.1)       -> 0.8/0.2 and 0.95/0.05

The module constants of tta_px1 are overridden before ks_eval runs, so the code path is exactly the one used for
the reported results. Usage: python sens_eval.py --config hand3 --model-seed 0 --out runs_sens/hand3 [ks_eval args]
"""
import sys

import tta_px1

CONFIGS = {
    "cap_m2": {"VV_CAP": -17.0, "VH_CAP": -24.0},
    "cap_p2": {"VV_CAP": -13.0, "VH_CAP": -20.0},
    "hand3": {"HAND_LOW": 3.0},
    "hand10": {"HAND_LOW": 10.0},
    "steps100": {},
    "steps400": {},
    "conf80": {"P_HI": 0.8, "P_LO": 0.2},
    "conf95": {"P_HI": 0.95, "P_LO": 0.05},
}

if __name__ == "__main__":
    i = sys.argv.index("--config")
    cfg = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    for k, v in CONFIGS[cfg].items():
        assert hasattr(tta_px1, k), k
        setattr(tta_px1, k, v)
    print(f"sensitivity config {cfg}: {CONFIGS[cfg]} (HAND_LOW {tta_px1.HAND_LOW}, caps {tta_px1.VV_CAP}/{tta_px1.VH_CAP}, "
          f"P_HI/P_LO {tta_px1.P_HI}/{tta_px1.P_LO})", flush=True)
    import runpy
    sys.argv[0] = "ks_eval.py"
    runpy.run_path("ks_eval.py", run_name="__main__")
'''
FILES['maps.py'] = r'''"""
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
'''
FILES['final_analyse.py'] = r'''"""
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
'''
MD5 = {'r34_patch.py': '739506c5ec2c5ba451137f7c82e6da6c', 'sens_eval.py': 'a4a7b10167278ca5b1869cdfd33e64e7', 'maps.py': 'c825b3db09d11638f45491a3a5d1e86c', 'final_analyse.py': '87cedc58c7fa9f368e8c1d5bec7dbe97'}
for k, v in FILES.items():
    assert hashlib.md5(v.encode()).hexdigest() == MD5[k], 'typo in ' + k
    open(os.path.join(ST, k), 'w').write(v)
code = os.path.dirname(glob.glob('/kaggle/input/**/run_kaggle.py', recursive=True)[0])
print('code ok', sorted(FILES), code, flush=True)
assert subprocess.call([py, os.path.join(ST, 'r34_patch.py'), code, '--check']) == 0
assert subprocess.call([py, '-c', f"import sys; sys.path.insert(0, {code!r}); import run_kaggle; run_kaggle.setup()"]) == 0
for k in FILES:
    shutil.copy(os.path.join(ST, k), W)
assert subprocess.call([py, 'r34_patch.py', '.'], cwd=W) == 0
print('Kuro Siwo chips:', len(glob.glob(W + '/data_ks/*_GRD/*/*.npz')), '| HAND:', len(glob.glob(W + '/data_ks_hand/**/*.npy', recursive=True)), flush=True)
sys.path.insert(0, W)
os.chdir(W)
import run_kaggle
base = f'{py} -u ks_eval.py --data data --ks data_ks --runs runs --out runs_ks --variants V0'
jobs = [('ks_m0', base + ' --model-seed 0 --methods none,tent,adabn,physa,adabn_physa')]
jobs += [(f'ks_m{s}', base + f' --model-seed {s} --methods physa,adabn_physa') for s in (1, 2)]
for cfg in ['cap_m2', 'cap_p2', 'hand3', 'hand10', 'steps100', 'steps400', 'conf80', 'conf95']:
    extra = ' --steps 100' if cfg == 'steps100' else (' --steps 400' if cfg == 'steps400' else '')
    jobs.append((f'sens_{cfg}', f'{py} -u sens_eval.py --config {cfg} --data data --ks data_ks --runs runs --model-seed 0 --out runs_sens/{cfg} --methods adabn_physa --variants V0' + extra))
status = run_kaggle.run_parallel(jobs, per_gpu=2)
print(status, flush=True)
os.makedirs(OUT + '/runs_ks', exist_ok=True)
for f in glob.glob(W + '/runs_ks/ks_*'):
    shutil.copy(f, OUT + '/runs_ks')
shutil.copytree(W + '/runs_sens', OUT + '/runs_sens', dirs_exist_ok=True)
subprocess.call([py, '-u', 'final_analyse.py', '--ks', 'runs_ks', '--sens', 'runs_sens', '--out', OUT], cwd=W)
subprocess.call([py, '-u', 'maps.py', '--data', 'data', '--ks', 'data_ks', '--runs', 'runs', '--hist', 'runs_ks/ks_hist_m0_v2.npz', '--events', '324,147,1111010,118'], cwd=W, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'))
print('EXTRAS_DONE')
