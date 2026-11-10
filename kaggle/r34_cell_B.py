import os, sys, glob, time, shutil, hashlib, subprocess
py = sys.executable
W, OUT, ST = '/tmp/px1', '/kaggle/working/out', '/tmp/px1_stage'
for d in [OUT + '/logs', ST]:
    os.makedirs(d, exist_ok=True)
FILES = {}
FILES['mm_prep.py'] = r'''import argparse
import csv
import glob
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault('GDAL_CACHEMAX', '256')
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform_bounds
URL = 'https://zenodo.org/records/6534637/files/mmflood.zip?download=1'
CHIP, MAX_CHIPS, MIN_CHIPS, MIN_VALID = (512, 48, 8, 0.5)
S1F_YEARS = {'Bolivia': 2018, 'Ghana': 2018, 'India': 2016, 'Mekong': 2018, 'Nigeria': 2018, 'Pakistan': 2017, 'Paraguay': 2018, 'Somalia': 2018, 'Spain': 2019, 'Sri-Lanka': 2017, 'USA': 2019}

def km(lon1, lat1, lon2, lat2):
    p1, p2 = (math.radians(lat1), math.radians(lat2))
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(min(1, a)))

def s1f_centroids(data):
    pts = {}
    for f in glob.glob(os.path.join(data, 'S1Hand', '*.tif')):
        e = os.path.basename(f).split('_')[0]
        with rasterio.open(f) as src:
            b = src.bounds if src.crs.is_geographic else transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
        pts.setdefault(e, []).append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    return {e: (np.mean([p[0] for p in v]), np.mean([p[1] for p in v])) for e, v in pts.items()}

def download(root):
    z = os.path.join(root, 'mmflood.zip')
    if not os.path.exists(os.path.join(root, 'extracted.ok')):
        os.makedirs(root, exist_ok=True)
        for t in range(4):
            if subprocess.call(['wget', '-q', '-c', '-O', z, URL]) == 0:
                break
        print(f'downloaded {os.path.getsize(z) / 2 ** 30:.1f} GB', flush=True)
        subprocess.check_call(['unzip', '-q', '-o', z, '-d', root])
        os.remove(z)
        open(os.path.join(root, 'extracted.ok'), 'w').close()
    for p in sorted(glob.glob(os.path.join(root, '*')) + glob.glob(os.path.join(root, '*', '*')))[:20]:
        print(f'  layout: {p}', flush=True)
    if not glob.glob(os.path.join(root, '**', 's1_raw'), recursive=True):
        for z in glob.glob(os.path.join(root, '**', '*.zip'), recursive=True) + glob.glob(os.path.join(root, '**', '*.tar*'), recursive=True):
            print(f'  extracting inner archive {z}', flush=True)
            if z.endswith('.zip'):
                subprocess.call(['unzip', '-q', '-o', z, '-d', os.path.dirname(z)])
            else:
                subprocess.call(['tar', '-xf', z, '-C', os.path.dirname(z)])
    acts = glob.glob(os.path.join(root, '**', 'activations.json'), recursive=True)
    dirs = {os.path.basename(os.path.dirname(p)): os.path.dirname(p) for p in glob.glob(os.path.join(root, '**', 's1_raw'), recursive=True) if os.path.isdir(p)}
    print(f'activations.json: {acts[:2]}; {len(dirs)} event folders', flush=True)
    if not dirs:
        raise SystemExit('no s1_raw folders found in the extracted archive')
    return (acts[0] if acts else None, dirs)

def to_db(a):
    a = a.astype(np.float32)
    v = a[np.isfinite(a) & (a != 0)]
    if v.size and np.median(v) > 0:
        a = np.where(a > 0, 10 * np.log10(np.maximum(a, 1e-06)), np.nan)
    return a

def resample(src, band, transform, crs, shape, method):
    out = np.full(shape, np.nan, np.float32)
    reproject(src.read(band).astype(np.float32), out, src_transform=src.transform, src_crs=src.crs, dst_transform=transform, dst_crs=crs, src_nodata=src.nodata, dst_nodata=np.nan, resampling=method)
    return out

def image_chips(base, name):
    with rasterio.open(os.path.join(base, 's1_raw', name)) as s1:
        crs, t = (s1.crs, s1.transform)
        lat = (s1.bounds.top + s1.bounds.bottom) / 2
        res_m = abs(t.a) * (111320 * math.cos(math.radians(lat)) if crs.is_geographic else 1)
        f = max(1.0, res_m / 10.0)
        shape = (int(s1.height * f), int(s1.width * f))
        tt = t * Affine.scale(1 / f)
        vv = to_db(resample(s1, 1, tt, crs, shape, Resampling.bilinear))
        vh = to_db(resample(s1, 2, tt, crs, shape, Resampling.bilinear))
    with rasterio.open(os.path.join(base, 'mask', name)) as m:
        mask = resample(m, 1, tt, crs, shape, Resampling.nearest)
    with rasterio.open(os.path.join(base, 'DEM', name)) as d:
        dem = resample(d, 1, tt, crs, shape, Resampling.bilinear)
    hp = os.path.join(base, 'hydro', name)
    hydro = None
    if os.path.exists(hp):
        with rasterio.open(hp) as h:
            hydro = resample(h, 1, tt, crs, shape, Resampling.nearest)
    valid = np.isfinite(vv) & np.isfinite(vh) & (vv > -49) & (vh > -49)
    vals = set(np.unique(mask[np.isfinite(mask)]).astype(int).tolist())
    water_val = 255 if vals <= {0, 255} and 255 in vals else 1
    lab = np.full(shape, 255, np.uint8)
    lab[(mask == 0) & valid] = 0
    lab[(mask == water_val) & valid] = 1
    if hydro is not None:
        lab[(hydro > 0) & (lab == 0)] = 255
    chips = []
    for r in range(0, shape[0] - CHIP + 1, CHIP):
        for c in range(0, shape[1] - CHIP + 1, CHIP):
            v = valid[r:r + CHIP, c:c + CHIP]
            if v.mean() < MIN_VALID:
                continue
            d = dem[r:r + CHIP, c:c + CHIP]
            d = np.nan_to_num(d, nan=float(np.nanmedian(d)) if np.isfinite(d).any() else 0.0)
            chips.append({'vv': np.nan_to_num(vv[r:r + CHIP, c:c + CHIP], nan=-50.0), 'vh': np.nan_to_num(vh[r:r + CHIP, c:c + CHIP], nan=-50.0), 'dem': d, 'lab': lab[r:r + CHIP, c:c + CHIP], 'transform': tuple(tt * Affine.translation(c, r))[:6], 'crs': crs.to_wkt()})
    return chips

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='mmflood_raw')
    ap.add_argument('--out', default='data_mm')
    ap.add_argument('--data', default='data')
    ap.add_argument('--ks-events', default='ks_events.json')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    meta_path, dirs = download(a.raw)
    meta = {}
    if meta_path:
        txt = open(meta_path).read()
        try:
            meta = json.loads(txt)
        except json.JSONDecodeError:
            import re
            meta = json.loads(re.sub(',\\s*([}\\]])', '\\1', txt))
    if isinstance(meta, list):
        meta = {m.get('code') or m.get('activation') or str(i): m for i, m in enumerate(meta)}
    acts = {}
    for d in sorted(dirs):
        info = meta.get(d) or meta.get(d.split('-')[0]) or {}
        if 'lon' not in info or 'lat' not in info:
            tifs = sorted(glob.glob(os.path.join(dirs[d], 's1_raw', '*.tif')))
            if tifs:
                with rasterio.open(tifs[0]) as src:
                    b = src.bounds if src.crs.is_geographic else transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
                info = dict(info, lon=(b[0] + b[2]) / 2, lat=(b[1] + b[3]) / 2)
        acts[d] = {'lon': info.get('lon', 0), 'lat': info.get('lat', 0), 'start': str(info.get('start') or info.get('end') or '2000-01-01'), 'country': info.get('country'), 'subset': info.get('subset')}
    print(f'metadata example: {next(iter(meta.items()), None)}', flush=True)
    ks = json.load(open(a.ks_events))
    s1f = s1f_centroids(a.data)
    keep, excl = ([], [])
    for code, info in sorted(acts.items()):
        lon, lat, date = (float(info['lon']), float(info['lat']), info['start'][:10])
        why = []
        for k, e in ks.items():
            if km(lon, lat, e['lon'], e['lat']) <= 300:
                dd = abs((np.datetime64(date) - np.datetime64(e['date'])).astype(int))
                if dd <= 365:
                    why.append(f'KuroSiwo {k} ({dd} days)')
        for e, (elon, elat) in s1f.items():
            if km(lon, lat, elon, elat) <= 300 and abs(int(date[:4]) - S1F_YEARS.get(e, 0)) <= 1:
                why.append(f'Sen1Floods11 {e}')
        (excl if why else keep).append((code, info, '; '.join(why)))
    print(f'{len(acts)} activations; {len(excl)} excluded for overlap; {len(keep)} kept', flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'excluded.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['event', 'country', 'start', 'reason'])
        for code, info, why in excl:
            w.writerow([code, info.get('country'), info['start'][:10], why])
    import hand_fetch
    hand_fetch.probe()
    rows = []

    def do_event(item):
        code, info, _ = item
        base = dirs[code]
        if not os.path.isdir(base):
            return []
        evdir = os.path.join(a.out, code)
        if os.path.exists(os.path.join(evdir, 'done.ok')):
            return [r for r in csv.DictReader(open(os.path.join(evdir, 'chips.csv')))]
        chips = []
        for name in sorted(os.listdir(os.path.join(base, 's1_raw'))):
            try:
                chips += [(name, ch) for ch in image_chips(base, name)]
            except Exception as e:
                print(f'  {code}/{name}: {type(e).__name__} {e}', flush=True)
        if len(chips) < MIN_CHIPS:
            return []
        if len(chips) > MAX_CHIPS:
            idx = np.linspace(0, len(chips) - 1, MAX_CHIPS).round().astype(int)
            chips = [chips[i] for i in sorted(set(idx))]
        os.makedirs(evdir, exist_ok=True)
        out = []
        for i, (name, ch) in enumerate(chips):
            tr = Affine(*ch['transform'])
            hand = hand_fetch.hand_on_grid(tr, rasterio.crs.CRS.from_wkt(ch['crs']), (CHIP, CHIP))
            fn = os.path.join(evdir, f'{i:03d}.npz')
            np.savez_compressed(fn, vv=ch['vv'].astype(np.float16), vh=ch['vh'].astype(np.float16), dem=ch['dem'].astype(np.float32), hand=hand.astype(np.float32), lab=ch['lab'])
            out.append({'event': code, 'file': fn, 'image': name, 'country': info.get('country'), 'start': info['start'][:10], 'water_px': int((ch['lab'] == 1).sum()), 'dry_px': int((ch['lab'] == 0).sum())})
        with open(os.path.join(evdir, 'chips.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        open(os.path.join(evdir, 'done.ok'), 'w').close()
        print(f"  {code} ({info.get('country')}): {len(out)} chips, water px {sum((r['water_px'] for r in out))}, dry px {sum((r['dry_px'] for r in out))}", flush=True)
        return out
    with ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(do_event, keep):
            rows += r
    with open(os.path.join(a.out, 'index.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['event', 'file', 'image', 'country', 'start', 'water_px', 'dry_px'])
        w.writeheader()
        w.writerows(rows)
    n_ev = len({r['event'] for r in rows})
    print(f'MMFlood: {n_ev} events, {len(rows)} chips', flush=True)
    print('MM_PREP_DONE', flush=True)
if __name__ == '__main__':
    main()
'''
FILES['mm_eval.py'] = r'''import argparse
import csv
import os
import time
os.environ.setdefault('GDAL_CACHEMAX', '64')
import numpy as np
from ks_eval import terrain_10m
from tta_px1 import BINS, curves, hand_channel, otsu_counts, run_one, save
METHODS_MM = ['none', 'tent', 'adabn', 'adabn_phys', 'adabn_physa']
MIN_CHIPS = 8

def load_mm_event(files):
    ev = {'raw': [], 'terr': [], 'lab': [], 'clean': []}
    for f in files:
        z = np.load(f)
        ev['raw'].append(np.clip(np.stack([z['vv'], z['vh']]).astype(np.float32), -50, 5))
        hand = z['hand']
        ev['hand_ok'] = ev.get('hand_ok', 0) + int(np.any(hand != 0))
        ev['terr'].append(hand_channel(hand, terrain_10m(z['dem'])))
        ev['lab'].append(z['lab'])
        ev['clean'].append(False)
    return ev

def events_of(mm):
    ev = {}
    for r in csv.DictReader(open(os.path.join(mm, 'index.csv'))):
        ev.setdefault(r['event'], []).append(r['file'])
    return {e: sorted(f) for e, f in sorted(ev.items()) if len(f) >= MIN_CHIPS}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--mm', default='data_mm')
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--out', default='runs_mm')
    ap.add_argument('--model-seed', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1)
    ap.add_argument('--variants', default='V0')
    ap.add_argument('--methods', default=','.join(METHODS_MM))
    a = ap.parse_args()
    a.methods, variants = (a.methods.split(','), a.variants.split(','))
    import torch
    from sen1floods11_unet_baseline import channel_stats, read_split
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f'mm_hist_m{a.model_seed}_p{a.part}.npz')
    done = dict(np.load(path)) if os.path.exists(path) else {}
    mean, std = channel_stats(a.data, read_split(a.data, 'train'))
    events = list(events_of(a.mm).items())[a.part::a.nparts]
    print(f'part {a.part}/{a.nparts}: {len(events)} MMFlood events', flush=True)
    for e, files in events:
        todo = [(m, v) for v in variants for m in a.methods if f'{m}|{v}|{e}' not in done]
        if not todo and f'otsu|{e}' in done:
            continue
        ev = load_mm_event(files)
        if f'otsu|{e}' not in done:
            done[f'otsu|{e}'], _ = otsu_counts(ev)
        for m, v in todo:
            t0 = time.time()
            ckpt = os.path.join(a.runs, f'{v}_seed{a.model_seed}', 'model.pt')
            h = run_one(m, v, e, ev, mean, std, a, device, ckpt=ckpt)
            done[f'{m}|{v}|{e}'] = h
            fnr, _, iou = curves(h[0])
            print(f'{m:11s} {v} event {e} ({len(files)} chips): miss rate at 0.5 {fnr[BINS // 2]:.3f}, IoU {iou[BINS // 2]:.3f}, {time.time() - t0:.0f}s', flush=True)
            save(path, done)
        save(path, done)
        if device == 'cuda':
            torch.cuda.empty_cache()
    print('MM_EVAL_DONE', flush=True)
if __name__ == '__main__':
    main()
'''
FILES['ks_events.json'] = r'''{"118":{"lon":-2.333,"lat":42.358,"date":"2015-02-02"},"130":{"lon":94.896,"lat":17.072,"date":"2015-08-01"},"147":{"lon":-2.821,"lat":54.647,"date":"2015-12-05"},"267":{"lon":21.491,"lat":55.287,"date":"2018-01-31"},"273":{"lon":19.546,"lat":42.01,"date":"2018-03-10"},"275":{"lon":16.674,"lat":45.387,"date":"2018-03-21"},"277":{"lon":26.462,"lat":41.291,"date":"2018-03-27"},"279":{"lon":-0.248,"lat":41.203,"date":"2018-04-12"},"321":{"lon":-86.991,"lat":13.049,"date":"2018-10-06"},"324":{"lon":2.674,"lat":43.226,"date":"2018-10-15"},"417":{"lon":-8.638,"lat":40.189,"date":"2019-12-19"},"421":{"lon":2.96,"lat":42.721,"date":"2020-01-22"},"427":{"lon":13.141,"lat":58.173,"date":"2020-02-18"},"437":{"lon":-0.093,"lat":43.971,"date":"2020-05-10"},"445":{"lon":26.705,"lat":48.199,"date":"2020-06-26"},"470":{"lon":0.573,"lat":10.409,"date":"2020-10-05"},"492":{"lon":-0.982,"lat":43.705,"date":"2021-01-01"},"497":{"lon":6.651,"lat":51.446,"date":"2021-02-01"},"498":{"lon":1.842,"lat":45.011,"date":"2021-02-02"},"502":{"lon":-8.449,"lat":52.029,"date":"2021-02-23"},"518":{"lon":5.632,"lat":50.571,"date":"2021-07-14"},"555":{"lon":-1.848,"lat":42.449,"date":"2021-12-10"},"567":{"lon":152.999,"lat":-29.618,"date":"2022-02-26"},"1111004":{"lon":-96.354,"lat":29.119,"date":"2017-08-28"},"1111005":{"lon":48.402,"lat":-17.628,"date":"2020-01-25"},"1111006":{"lon":46.6,"lat":-16.461,"date":"2020-01-30"},"1111007":{"lon":85.67,"lat":25.55,"date":"2019-09-17"},"1111008":{"lon":-83.538,"lat":14.072,"date":"2020-08-15"},"1111009":{"lon":68.295,"lat":27.36,"date":"2022-09-10"},"1111010":{"lon":-80.712,"lat":-5.568,"date":"2017-03-10"},"1111011":{"lon":121.698,"lat":17.584,"date":"2020-11-12"},"1111013":{"lon":-90.109,"lat":38.386,"date":"2017-05-03"},"554":{"lon":151.171,"lat":-28.431,"date":"2021-11-30"}}'''
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
FILES['bb_analyse.py'] = r'''import argparse
import glob
import json
import os
import numpy as np
from scipy.stats import wilcoxon
from tta_px1 import compute
RULES = ['default', 'crc_0.1', 'crc_0.2', 'crc_0.3']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hist', required=True, help='glob with {s} for the model seed')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--methods', default='none,tent,adabn,adabn_physa')
    ap.add_argument('--primary', default='adabn_physa')
    ap.add_argument('--variant', default='V0')
    ap.add_argument('--activation', action='store_true', help='merge MMFlood region folders (EMSRxxx-N) into activations')
    ap.add_argument('--title', default='')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    methods = a.methods.split(',')
    res = []
    for s in a.seeds.split(','):
        files = sorted(glob.glob(a.hist.format(s=s)))
        if not files:
            print(f'no files for seed {s}', flush=True)
            continue
        data = {}
        for p in files:
            for k, v in np.load(p).items():
                if a.activation:
                    parts = k.split('|')
                    parts[-1] = parts[-1].split('-')[0]
                    k = '|'.join(parts)
                data[k] = data[k] + v if k in data else v.copy()
        events = sorted({k.split('|')[-1] for k in data if k.count('|') == 2})
        res.append(compute(data, events)[f'{a.variant}|all'])
    methods = [m for m in methods if all((m in r for r in res))]
    evs = sorted(set.intersection(*[set(r[m]['events']) for r in res for m in methods]))
    unit = 'activations' if a.activation else 'events'
    L = [f'## {a.title}', '', f'Models: {len(res)}; {unit} analysed: {len(evs)}. Exploratory (not pre-registered).', '', '| Method | ' + ' | '.join((f'{r} IoU' for r in RULES)) + f' | 20%: mean miss rate | 20%: {unit} meeting |', '| --- |' + ' --- |' * (len(RULES) + 2)]
    per = {}
    for m in methods:
        for r in RULES:
            per[m, r] = np.array([np.mean([x[m]['events'][e][r]['iou'] for x in res]) for e in evs])
        fnr = [np.array([x[m]['events'][e]['crc_0.2']['fnr'] for e in evs]) for x in res]
        L.append(f'| {m} | ' + ' | '.join((f'{per[m, r].mean():.3f}' for r in RULES)) + f' | {np.mean([f.mean() for f in fnr]):.3f} | {np.mean([(f <= 0.2).sum() for f in fnr]):.1f}/{len(evs)} |')
    L.append('')
    tests = [(ref, r) for ref, r in [('tent', 'crc_0.2'), ('adabn', 'crc_0.2'), ('none', 'default')] if ref in methods]
    out, ps = ({}, [])
    for ref, r in tests:
        d = per[a.primary, r] - per[ref, r]
        p = float(wilcoxon(d, alternative='greater').pvalue) if np.any(d != 0) else 1.0
        ps.append(p)
        out[f'{ref}|{r}'] = {'mean_gain': float(d.mean()), 'wins': int((d > 0).sum()), 'n': len(d), 'p': p}
    running = 0.0
    for rank, i in enumerate(np.argsort(ps)):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[f'{tests[i][0]}|{tests[i][1]}']['p_holm'] = running
    for ref, r in tests:
        o = out[f'{ref}|{r}']
        L.append(f"- {a.primary} vs {ref} ({r}): mean gain {o['mean_gain']:+.3f}, better on {o['wins']}/{o['n']}, p {o['p']:.3g}, Holm p {o['p_holm']:.3g}")
    txt = '\n'.join(L)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    open(a.out, 'w').write(txt)
    json.dump({'tests': out, 'table': {f'{m}|{r}': float(per[m, r].mean()) for m in methods for r in RULES}}, open(a.out[:-3] + '.json', 'w'), indent=1)
    print(txt)
    print('BB_ANALYSIS_DONE', flush=True)
if __name__ == '__main__':
    main()
'''
MD5 = {'mm_prep.py': 'f17430d9393438bbc2e9f379ff98ee22', 'mm_eval.py': '588f8fea6adfbf83d7f20b09d46851ec', 'ks_events.json': '166be54a6e14ff3aca8ed49cdf902d69', 'r34_patch.py': '739506c5ec2c5ba451137f7c82e6da6c', 'bb_analyse.py': '65111eea70b09d52f69a049765cce7d4'}
for k, v in FILES.items():
    assert hashlib.md5(v.encode()).hexdigest() == MD5[k], 'typo in ' + k
    open(os.path.join(ST, k), 'w').write(v)
code = os.path.dirname(glob.glob('/kaggle/input/**/run_kaggle.py', recursive=True)[0])
models = sorted(glob.glob('/kaggle/input/**/runs_r34/V0_seed*/model.pt', recursive=True))
print('code ok', sorted(FILES), code, models, flush=True)
assert len(models) == 3
assert subprocess.call([py, os.path.join(ST, 'r34_patch.py'), code, '--check']) == 0
subprocess.call([py, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch'])
assert subprocess.call([py, '-c', f"import sys; sys.path.insert(0, {code!r}); import run_kaggle; run_kaggle.setup()"]) == 0
for k in FILES:
    shutil.copy(os.path.join(ST, k), W)
assert subprocess.call([py, 'r34_patch.py', '.'], cwd=W) == 0
for m in models:
    d = os.path.join(W, 'runs_r34', os.path.basename(os.path.dirname(m)))
    os.makedirs(d, exist_ok=True)
    shutil.copy(m, d)
assert subprocess.call([py, '-u', 'mm_prep.py', '--raw', 'mmflood_raw', '--out', 'data_mm', '--data', 'data', '--ks-events', 'ks_events.json', '--workers', '4'], cwd=W) == 0
ENV = dict(os.environ, PX1_ARCH='r34')
jobs = [(f'mm_r34_m{s}_p{p}', [py, '-u', 'mm_eval.py', '--data', 'data', '--mm', 'data_mm', '--runs', 'runs_r34', '--model-seed', str(s), '--part', str(p), '--nparts', '2', '--out', 'runs_mm_r34', '--methods', 'none,tent,adabn,adabn_physa']) for p in range(2) for s in range(3)]
def start(i):
    name, cmd = jobs[i]
    return subprocess.Popen(cmd, cwd=W, env=dict(ENV, CUDA_VISIBLE_DEVICES=str(i % 2)), stdout=open(f'{OUT}/logs/{name}.txt', 'a'), stderr=subprocess.STDOUT)
procs, retried = {i: start(i) for i in range(len(jobs))}, set()
while procs:
    time.sleep(300)
    for i, pr in list(procs.items()):
        name = jobs[i][0]
        lines = [l for l in open(f'{OUT}/logs/{name}.txt').read().splitlines() if 'event' in l]
        print(time.strftime('%H:%M'), name, pr.poll(), lines[-1][:110] if lines else '', flush=True)
        if pr.poll() is not None:
            del procs[i]
            if pr.returncode != 0 and i not in retried:
                retried.add(i)
                procs[i] = start(i)
shutil.copytree(f'{W}/runs_mm_r34', f'{OUT}/runs_mm_r34', dirs_exist_ok=True)
for f in glob.glob(W + '/data_mm/*.csv'):
    shutil.copy(f, OUT)
subprocess.call([py, '-u', 'bb_analyse.py', '--hist', 'runs_mm_r34/mm_hist_m{s}_p*.npz', '--activation', '--title', 'MMFlood (activations), ResNet-34 U-Net', '--out', OUT + '/mm_r34_act.md'], cwd=W)
subprocess.call([py, '-u', 'bb_analyse.py', '--hist', 'runs_mm_r34/mm_hist_m{s}_p*.npz', '--title', 'MMFlood (region folders), ResNet-34 U-Net', '--out', OUT + '/mm_r34_folder.md'], cwd=W)
print('R34_MM_DONE')
