import os, sys, glob, shutil, hashlib, subprocess
INP = '/kaggle/input'
W = '/tmp/act'
OUT = '/kaggle/working/out_act'
os.makedirs(W + '/runs_mm', exist_ok=True)
os.makedirs(W + '/data_mm', exist_ok=True)
code = os.path.dirname(glob.glob(INP + '/**/tta_px1.py', recursive=True)[0])
for f in ['tta_px1.py', 'conformal_px1.py']:
    shutil.copy(os.path.join(code, f), W)
hist = sorted(glob.glob(INP + '/**/mm_hist_m*_p*.npz', recursive=True))
print(code, len(hist), hist)
assert len(hist) == 6
for f in hist:
    shutil.copy(f, W + '/runs_mm')
for n in ['excluded.csv', 'index.csv']:
    shutil.copy(os.path.join(os.path.dirname(hist[0]), n), W + '/data_mm')
FILES = {}
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
FILES['mm_analyse_act.py'] = r'''import argparse
import glob
import os
import subprocess
import sys
import numpy as np

def merge(runs, dst):
    os.makedirs(dst, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(runs, 'mm_hist_m*_p*.npz'))):
        s = os.path.basename(p).split('_')[2]
        out_p = os.path.join(dst, f'mm_hist_{s}_p0.npz')
        acc = dict(np.load(out_p)) if os.path.exists(out_p) else {}
        for k, v in np.load(p).items():
            parts = k.split('|')
            parts[-1] = parts[-1].split('-')[0]
            kk = '|'.join(parts)
            acc[kk] = acc[kk] + v if kk in acc else v.copy()
        np.savez(out_p, **acc)
    n = {k.split('|')[-1] for p in glob.glob(os.path.join(dst, '*.npz')) for k in np.load(p).keys()}
    print(f'merged into {len(n)} activations', flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--primary', default='adabn_physa')
    ap.add_argument('--runs', default='runs_mm')
    ap.add_argument('--mm', default='data_mm')
    ap.add_argument('--out', default='out_act')
    a = ap.parse_args()
    merged = a.runs.rstrip('/') + '_act'
    if os.path.isdir(merged):
        raise SystemExit(f'{merged} exists; remove it first so histograms are not added twice')
    merge(a.runs, merged)
    sys.exit(subprocess.call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mm_analyse.py'), '--primary', a.primary, '--runs', merged, '--mm', a.mm, '--out', a.out]))
'''
FILES['mm_detail.py'] = r'''import csv
import glob
import os
import sys
import numpy as np
from tta_px1 import compute
METHODS = ['none', 'tent', 'adabn', 'adabn_phys', 'adabn_physa']
runs, index, out = sys.argv[1:4]
country, folders, chips = ({}, {}, {})
for r in csv.DictReader(open(index)):
    a = r['event'].split('-')[0]
    country[a] = r['country']
    folders.setdefault(a, set()).add(r['event'])
    chips[a] = chips.get(a, 0) + 1
res = []
for p in sorted(glob.glob(os.path.join(runs, 'mm_hist_m*_p0.npz'))):
    d = dict(np.load(p))
    res.append(compute(d, sorted({k.split('|')[-1] for k in d if k.count('|') == 2}))['V0|all'])
evs = sorted(set.intersection(*[set(r[m]['events']) for r in res for m in METHODS]))
L = [f'Activations analysed: {len(evs)} (of {len(folders)} with chips); models: {len(res)}', '', '| Activation | Country | Folders | Chips | ' + ' | '.join((f'{m} 0.5 / 20%' for m in METHODS)) + ' |', '| --- ' * (4 + len(METHODS)) + '|']
for e in evs:
    cells = []
    for m in METHODS:
        d0 = np.mean([r[m]['events'][e]['default']['iou'] for r in res])
        d2 = np.mean([r[m]['events'][e]['crc_0.2']['iou'] for r in res])
        cells.append(f'{d0:.2f} / {d2:.2f}')
    L.append(f'| {e} | {country.get(e)} | {len(folders.get(e, []))} | {chips.get(e, 0)} | ' + ' | '.join(cells) + ' |')
L.append('')
for m in METHODS:
    f = [np.array([r[m]['events'][e]['crc_0.2']['fnr'] for e in evs]) for r in res]
    L.append(f'{m}: 20% guarantee, per-activation miss rate mean {np.mean([x.mean() for x in f]):.3f}, activations meeting it {np.mean([(x <= 0.2).sum() for x in f]):.1f}/{len(evs)}, worst {np.mean([x.max() for x in f]):.3f}')
txt = '\n'.join(L)
os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
open(out, 'w').write(txt)
print(txt)
print('MM_DETAIL_DONE')
'''
MD5 = {'mm_analyse.py': '6ea18eab68af78d7e2f7f2897249a817', 'mm_analyse_act.py': '2434c4f3fd256813833d5db344363cbf', 'mm_detail.py': 'bb5bcac2c90e3caf05d2edefd631f441'}
for k, v in FILES.items():
    assert hashlib.md5(v.encode()).hexdigest() == MD5[k], 'typo in ' + k
    open(os.path.join(W, k), 'w').write(v)
print('code ok', sorted(FILES))
py = sys.executable
print('--- per-folder analysis (secondary; should match the final-test log) ---', flush=True)
subprocess.call([py, '-u', 'mm_analyse.py', '--primary', 'adabn_physa', '--runs', 'runs_mm', '--mm', 'data_mm', '--out', OUT + '/folder'], cwd=W)
print('--- activation-level analysis (primary) ---', flush=True)
assert subprocess.call([py, '-u', 'mm_analyse_act.py', '--primary', 'adabn_physa', '--runs', 'runs_mm', '--mm', 'data_mm', '--out', OUT], cwd=W) == 0
subprocess.call([py, '-u', 'mm_detail.py', 'runs_mm_act', 'data_mm/index.csv', OUT + '/detail.md'], cwd=W)
print('ACT_TASK_DONE')
