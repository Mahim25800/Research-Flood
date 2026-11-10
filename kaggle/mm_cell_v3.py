import os, sys, glob, time, shutil, hashlib, subprocess
k = glob.glob('/kaggle/input/**/run_kaggle.py', recursive=True)[0]
print(k)
assert subprocess.call([sys.executable, '-u', k, '--task', 'setup']) == 0
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
FILES['mm_analyse.py'] = r'''import argparse
import csv
import glob
import json
import os
import numpy as np
from scipy.stats import wilcoxon
from tta_px1 import compute
METHODS = ['none', 'tent', 'adabn', 'adabn_phys', 'adabn_physa']
RULES = ['default', 'crc_0.1', 'crc_0.2', 'crc_0.3']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--primary', required=True, choices=['adabn_physa', 'adabn_phys'])
    ap.add_argument('--runs', default='runs_mm')
    ap.add_argument('--mm', default='data_mm')
    ap.add_argument('--out', default='out')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = []
    for s in range(3):
        parts = sorted(glob.glob(os.path.join(a.runs, f'mm_hist_m{s}_p*.npz')))
        if not parts:
            continue
        data = {}
        for p in parts:
            data.update(dict(np.load(p)))
        events = sorted({k.split('|')[2] for k in data if k.count('|') == 2})
        res.append(compute(data, events)['V0|all'])
    secondary = 'adabn_phys' if a.primary == 'adabn_physa' else 'adabn_physa'
    L = ['# MMFlood final test (pre-registered)', '', f'Primary method: {a.primary} (chosen by the pre-registered rule from the Kuro Siwo results). Secondary: {secondary}.', '']
    ex = os.path.join(a.mm, 'excluded.csv')
    if os.path.exists(ex):
        rows = list(csv.DictReader(open(ex)))
        L.append(f"Excluded for overlap with Kuro Siwo or Sen1Floods11: {len(rows)} events ({', '.join((r['event'] for r in rows))}).")
    methods = [m for m in METHODS if all((m in r for r in res))]
    evs = sorted(set.intersection(*[set(r[m]['events']) for r in res for m in methods]))
    L += [f'Models: {len(res)}; events with labelled water analysed: {len(evs)}.', '', '| Method | ' + ' | '.join((f'{r} IoU' for r in RULES)) + ' | 20%: mean miss rate | 20%: events meeting |', '| --- |' + ' --- |' * (len(RULES) + 2)]
    per = {}
    for m in methods:
        for r in RULES:
            per[m, r] = np.array([np.mean([x[m]['events'][e][r]['iou'] for x in res]) for e in evs])
        fn = np.mean([x[m]['summary']['crc_0.2']['mean_fnr'] for x in res])
        meet = np.mean([x[m]['summary']['crc_0.2']['meeting'] for x in res])
        L.append(f'| {m} | ' + ' | '.join((f'{per[m, r].mean():.3f}' for r in RULES)) + f' | {fn:.3f} | {meet:.1f}/{len(evs)} |')
    L.append('')
    tests = [('H1', 'tent', 'crc_0.2'), ('H2', 'adabn', 'crc_0.2'), ('H3', 'none', 'default')]
    ps, out = ([], {})
    for h, ref, r in tests:
        d = per[a.primary, r] - per[ref, r]
        p = float(wilcoxon(d, alternative='greater').pvalue) if np.any(d != 0) else 1.0
        ps.append(p)
        out[h] = {'ref': ref, 'rule': r, 'mean_gain': float(d.mean()), 'wins': int((d > 0).sum()), 'n': len(d), 'p': p}
    order = np.argsort(ps)
    adj, running = ([0.0] * 3, 0.0)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (3 - rank) * ps[i]))
        adj[i] = running
    for (h, ref, r), pa in zip(tests, adj):
        o = out[h]
        o['p_holm'], o['supported'] = (pa, bool(pa < 0.05 and o['mean_gain'] > 0))
        L.append(f"- {h}: {a.primary} vs {ref} ({r}): mean gain {o['mean_gain']:+.3f}, better on {o['wins']}/{o['n']} events, p {o['p']:.3g}, Holm p {pa:.3g} -> {('SUPPORTED' if o['supported'] else 'not supported')}")
    fn = float(np.mean([x[a.primary]['summary']['crc_0.2']['mean_fnr'] for x in res]))
    out['H4'] = {'mean_miss_rate': fn, 'supported': fn <= 0.2}
    L.append(f"- H4: mean miss rate under the 20% guarantee {fn:.3f} -> {('SUPPORTED' if fn <= 0.2 else 'not supported')}")
    d = per[a.primary, 'crc_0.2'] - per[secondary, 'crc_0.2']
    L.append(f'- (not tested) {a.primary} vs {secondary} under the 20% guarantee: {d.mean():+.3f}, better on {int((d > 0).sum())}/{len(d)}')
    txt = '\n'.join(L)
    open(os.path.join(a.out, 'analysis_mmflood.md'), 'w').write(txt)
    json.dump(out, open(os.path.join(a.out, 'analysis_mmflood.json'), 'w'), indent=1)
    print(txt)
    print('MM_ANALYSIS_DONE')
if __name__ == '__main__':
    main()
'''
FILES['ks_events.json'] = r'''{"118":{"lon":-2.333,"lat":42.358,"date":"2015-02-02"},"130":{"lon":94.896,"lat":17.072,"date":"2015-08-01"},"147":{"lon":-2.821,"lat":54.647,"date":"2015-12-05"},"267":{"lon":21.491,"lat":55.287,"date":"2018-01-31"},"273":{"lon":19.546,"lat":42.01,"date":"2018-03-10"},"275":{"lon":16.674,"lat":45.387,"date":"2018-03-21"},"277":{"lon":26.462,"lat":41.291,"date":"2018-03-27"},"279":{"lon":-0.248,"lat":41.203,"date":"2018-04-12"},"321":{"lon":-86.991,"lat":13.049,"date":"2018-10-06"},"324":{"lon":2.674,"lat":43.226,"date":"2018-10-15"},"417":{"lon":-8.638,"lat":40.189,"date":"2019-12-19"},"421":{"lon":2.96,"lat":42.721,"date":"2020-01-22"},"427":{"lon":13.141,"lat":58.173,"date":"2020-02-18"},"437":{"lon":-0.093,"lat":43.971,"date":"2020-05-10"},"445":{"lon":26.705,"lat":48.199,"date":"2020-06-26"},"470":{"lon":0.573,"lat":10.409,"date":"2020-10-05"},"492":{"lon":-0.982,"lat":43.705,"date":"2021-01-01"},"497":{"lon":6.651,"lat":51.446,"date":"2021-02-01"},"498":{"lon":1.842,"lat":45.011,"date":"2021-02-02"},"502":{"lon":-8.449,"lat":52.029,"date":"2021-02-23"},"518":{"lon":5.632,"lat":50.571,"date":"2021-07-14"},"555":{"lon":-1.848,"lat":42.449,"date":"2021-12-10"},"567":{"lon":152.999,"lat":-29.618,"date":"2022-02-26"},"1111004":{"lon":-96.354,"lat":29.119,"date":"2017-08-28"},"1111005":{"lon":48.402,"lat":-17.628,"date":"2020-01-25"},"1111006":{"lon":46.6,"lat":-16.461,"date":"2020-01-30"},"1111007":{"lon":85.67,"lat":25.55,"date":"2019-09-17"},"1111008":{"lon":-83.538,"lat":14.072,"date":"2020-08-15"},"1111009":{"lon":68.295,"lat":27.36,"date":"2022-09-10"},"1111010":{"lon":-80.712,"lat":-5.568,"date":"2017-03-10"},"1111011":{"lon":121.698,"lat":17.584,"date":"2020-11-12"},"1111013":{"lon":-90.109,"lat":38.386,"date":"2017-05-03"},"554":{"lon":151.171,"lat":-28.431,"date":"2021-11-30"}}'''
MD5 = {"mm_prep.py": "f17430d9393438bbc2e9f379ff98ee22", "mm_eval.py": "588f8fea6adfbf83d7f20b09d46851ec", "mm_analyse.py": "6ea18eab68af78d7e2f7f2897249a817", "ks_events.json": "166be54a6e14ff3aca8ed49cdf902d69"}
W = '/tmp/px1'
OUT = '/kaggle/working/out'
os.makedirs(OUT + '/logs', exist_ok=True)
for k, v in FILES.items():
    assert hashlib.md5(v.encode()).hexdigest() == MD5[k], 'typo in ' + k
    open(os.path.join(W, k), 'w').write(v)
print('code ok', sorted(FILES))
py = sys.executable
assert subprocess.call([py, '-u', 'mm_prep.py', '--raw', 'mmflood_raw', '--out', 'data_mm', '--data', 'data', '--ks-events', 'ks_events.json', '--workers', '4'], cwd=W) == 0
jobs = [(s, p) for p in range(2) for s in range(3)]
def start(i, s, p):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % 2))
    log = open(f'{OUT}/logs/mm_m{s}_p{p}.txt', 'a')
    return subprocess.Popen([py, '-u', 'mm_eval.py', '--data', 'data', '--mm', 'data_mm', '--runs', 'runs', '--model-seed', str(s), '--part', str(p), '--nparts', '2', '--out', 'runs_mm'], cwd=W, env=env, stdout=log, stderr=subprocess.STDOUT)
procs = {(s, p): start(i, s, p) for i, (s, p) in enumerate(jobs)}
retried = set()
while procs:
    time.sleep(300)
    for (s, p), pr in list(procs.items()):
        lines = [l for l in open(f'{OUT}/logs/mm_m{s}_p{p}.txt').read().splitlines() if 'event' in l]
        print(time.strftime('%H:%M'), f'mm_m{s}_p{p}', pr.poll(), lines[-1][:100] if lines else '', flush=True)
        if pr.poll() is not None:
            del procs[(s, p)]
            if pr.returncode != 0 and (s, p) not in retried:
                retried.add((s, p))
                procs[(s, p)] = start(jobs.index((s, p)), s, p)
for f in glob.glob(W + '/runs_mm/*') + glob.glob(W + '/data_mm/*.csv'):
    shutil.copy(f, OUT)
subprocess.call([py, '-u', 'mm_analyse.py', '--primary', 'adabn_physa', '--runs', 'runs_mm', '--mm', 'data_mm', '--out', OUT], cwd=W)
print('MMFLOOD_TASK_DONE')
