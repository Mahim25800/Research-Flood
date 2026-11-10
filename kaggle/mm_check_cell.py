import os, sys, hashlib, subprocess
py = sys.executable
W, OUT = '/tmp/chk', '/kaggle/working/out_check'
os.makedirs(W, exist_ok=True)
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
FILES['mm_check.py'] = r'''import argparse
import glob
import json
import os
import numpy as np

def otsu_thr(v):
    hist, edges = np.histogram(v, bins=256)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    mu0 = np.cumsum(hist * c) / np.maximum(w0, 1)
    mu1 = (np.sum(hist * c) - np.cumsum(hist * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (mu0 - mu1) ** 2)])

def iou(pred, lab):
    tp = np.sum(pred & (lab == 1))
    fp = np.sum(pred & (lab == 0))
    fn = np.sum(~pred & (lab == 1))
    return float(tp / max(1, tp + fp + fn))

def part_a(hist_dir, out):
    import csv
    files = sorted(glob.glob(os.path.join(hist_dir, '**', 'mm_hist_m0_p*.npz'), recursive=True))
    cnt = {}
    for f in files:
        z = np.load(f)
        for k in z.files:
            if k.startswith('otsu|'):
                a = k.split('|')[1].split('-')[0]
                cnt[a] = cnt.get(a, 0) + z[k][0]
    rows = {a: {'otsu_iou': float(c[0] / max(1, c.sum()))} for a, c in sorted(cnt.items())}
    idx = glob.glob(os.path.join(hist_dir, '**', 'index.csv'), recursive=True)
    if idx:
        w, d = ({}, {})
        for r in csv.DictReader(open(idx[0])):
            a = r['event'].split('-')[0]
            w[a] = w.get(a, 0) + int(r['water_px'])
            d[a] = d.get(a, 0) + int(r['dry_px'])
        for a in rows:
            rows[a]['water_fraction'] = w.get(a, 0) / max(1, w.get(a, 0) + d.get(a, 0))
    ious = [r['otsu_iou'] for r in rows.values()]
    pooled = np.sum(list(cnt.values()), axis=0)
    print(f'A. Otsu on VV (no model), {len(rows)} activations: mean IoU {np.mean(ious):.3f}, median {np.median(ious):.3f}, pixel-pooled IoU {pooled[0] / max(1, pooled.sum()):.3f}', flush=True)
    for a, r in rows.items():
        print(f"   {a}: Otsu IoU {r['otsu_iou']:.3f}, labelled water {100 * r.get('water_fraction', 0):.1f}% of valid pixels", flush=True)
    json.dump(rows, open(os.path.join(out, 'otsu_per_activation.json'), 'w'), indent=1)
    return rows

def part_b(raw, out, folders):
    import rasterio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import mm_prep
    _, dirs = mm_prep.download(raw)
    res, panels = ({}, [])
    for fo in folders:
        if fo not in dirs:
            print(f'   {fo}: not found', flush=True)
            continue
        base = dirs[fo]
        name = sorted(os.listdir(os.path.join(base, 's1_raw')))[0]
        with rasterio.open(os.path.join(base, 's1_raw', name)) as s:
            a = s.read().astype(np.float32)
            nod = s.nodata
        with rasterio.open(os.path.join(base, 'mask', name)) as m:
            mk = m.read(1)
        ok = np.isfinite(a[0]) & (a[0] > 0) & np.isfinite(a[1]) & (a[1] > 0)
        vv = np.where(ok, 10 * np.log10(np.maximum(a[0], 1e-06)), np.nan)
        lab = np.full(mk.shape, 255, np.uint8)
        lab[(mk == 0) & ok] = 0
        lab[(mk == 1) & ok] = 1
        w, d = (vv[lab == 1], vv[lab == 0])
        thr = otsu_thr(vv[ok & (vv > -49)])
        r = {'image': name, 'raw_vv_median': float(np.nanmedian(a[0][ok])), 'raw_vh_median': float(np.nanmedian(a[1][ok])), 'nodata': None if nod is None else float(nod), 'mask_values': sorted(set(np.unique(mk).tolist())), 'water_frac': float(np.mean(lab[lab != 255] == 1)), 'vv_db_median_water': float(np.median(w)) if w.size else None, 'vv_db_median_dry': float(np.median(d)) if d.size else None, 'water_dark_share': float(np.mean(w < -15)) if w.size else None, 'otsu_thr_db': thr, 'otsu_iou_native': iou((vv < thr) & ok, lab)}
        chips = mm_prep.image_chips(base, name)
        if chips:
            cv = np.concatenate([c['vv'].ravel() for c in chips])
            cl = np.concatenate([c['lab'].ravel() for c in chips])
            r['otsu_iou_chips_same_thr'] = iou(cv < thr, cl)
            r['chips_water_frac'] = float(np.mean(cl[cl != 255] == 1))
        res[fo] = r
        print(f'   {fo}: {json.dumps(r)}', flush=True)
        S = min(1024, *vv.shape)
        wy, wx = np.where(lab == 1)
        cy, cx = (int(np.median(wy)), int(np.median(wx))) if wy.size else (vv.shape[0] // 2, vv.shape[1] // 2)
        y0, x0 = (max(0, min(cy - S // 2, vv.shape[0] - S)), max(0, min(cx - S // 2, vv.shape[1] - S)))
        panels.append((fo, vv[y0:y0 + S, x0:x0 + S], lab[y0:y0 + S, x0:x0 + S]))
    if panels:
        fig, ax = plt.subplots(2, len(panels), figsize=(3.2 * len(panels), 6.6), squeeze=False)
        for i, (fo, v, l) in enumerate(panels):
            ax[0, i].imshow(np.clip(v, -25, 0), cmap='gray')
            ax[0, i].set_title(f'{fo} VV (dB)', fontsize=8)
            ax[1, i].imshow(np.clip(v, -25, 0), cmap='gray')
            ax[1, i].contour(l == 1, levels=[0.5], colors='cyan', linewidths=0.5)
            ax[1, i].set_title('label outline', fontsize=8)
            for a_ in ax[:, i]:
                a_.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'mmflood_check.png'), dpi=150)
    json.dump(res, open(os.path.join(out, 'raw_checks.json'), 'w'), indent=1)
    return res
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--hist-dir', required=True)
    ap.add_argument('--raw', default='mmflood_raw')
    ap.add_argument('--out', default='out_check')
    ap.add_argument('--folders', default='EMSR107-5,EMSR167-2,EMSR184-4,EMSR203-3,EMSR265-14,EMSR438-5,EMSR479-0,EMSR514-0')
    ap.add_argument('--skip-raw', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    part_a(a.hist_dir, a.out)
    if not a.skip_raw:
        print('B. raw-data checks', flush=True)
        part_b(a.raw, a.out, a.folders.split(','))
    print('MM_CHECK_DONE', flush=True)
'''
MD5 = {'mm_prep.py': 'f17430d9393438bbc2e9f379ff98ee22', 'mm_check.py': 'aa1657e7c9d30b0aac4aec27f0f3a68a'}
for k, v in FILES.items():
    assert hashlib.md5(v.encode()).hexdigest() == MD5[k], 'typo in ' + k
    open(os.path.join(W, k), 'w').write(v)
print('code ok', flush=True)
subprocess.call([py, '-m', 'pip', 'install', '-q', 'rasterio'])
subprocess.call([py, '-u', 'mm_check.py', '--hist-dir', '/kaggle/input', '--raw', '/tmp/mmraw', '--out', OUT], cwd=W)
print('CHECK_TASK_DONE')
