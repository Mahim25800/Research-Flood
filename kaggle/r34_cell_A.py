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
MD5 = {'r34_patch.py': '739506c5ec2c5ba451137f7c82e6da6c', 'bb_analyse.py': '65111eea70b09d52f69a049765cce7d4'}
for k, v in FILES.items():
    assert hashlib.md5(v.encode()).hexdigest() == MD5[k], 'typo in ' + k
    open(os.path.join(ST, k), 'w').write(v)
code = os.path.dirname(glob.glob('/kaggle/input/**/run_kaggle.py', recursive=True)[0])
print('code ok', sorted(FILES), code, flush=True)
assert subprocess.call([py, os.path.join(ST, 'r34_patch.py'), code, '--check']) == 0
subprocess.call([py, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch'])
assert subprocess.call([py, '-c', "import segmentation_models_pytorch as smp, torch; m = smp.Unet('resnet34', encoder_weights='imagenet', in_channels=2, classes=2); print('r34 ok', m(torch.zeros(1, 2, 224, 224)).shape)"]) == 0
assert subprocess.call([py, '-c', f"import sys; sys.path.insert(0, {code!r}); import run_kaggle; run_kaggle.setup()"]) == 0
for k in FILES:
    shutil.copy(os.path.join(ST, k), W)
assert subprocess.call([py, 'r34_patch.py', '.'], cwd=W) == 0
ENV = dict(os.environ, PX1_ARCH='r34')

def run_jobs(jobs, key):
    def start(i):
        name, cmd = jobs[i]
        env = dict(ENV, CUDA_VISIBLE_DEVICES=str(i % 2))
        return subprocess.Popen(cmd, cwd=W, env=env, stdout=open(f'{OUT}/logs/{name}.txt', 'a'), stderr=subprocess.STDOUT)
    procs, retried = {i: start(i) for i in range(len(jobs))}, set()
    while procs:
        time.sleep(300)
        for i, pr in list(procs.items()):
            name = jobs[i][0]
            lines = [l for l in open(f'{OUT}/logs/{name}.txt').read().splitlines() if key in l]
            print(time.strftime('%H:%M'), name, pr.poll(), lines[-1][:110] if lines else '', flush=True)
            if pr.poll() is not None:
                del procs[i]
                if pr.returncode != 0 and i not in retried:
                    retried.add(i)
                    procs[i] = start(i)
train = [(f'train_r34_s{s}', [py, '-u', 'train_px1.py', '--data', 'data', '--variant', 'V0', '--inputs', 's1', '--epochs', '60', '--lr', '3e-4', '--seed', str(s), '--out', 'runs_r34', '--workers', '1']) for s in range(3)]
run_jobs(train, 'epoch')
for s in range(3):
    d = f'{W}/runs_r34/V0_seed{s}'
    os.makedirs(f'{OUT}/runs_r34/V0_seed{s}', exist_ok=True)
    for f in ['model.pt', 'results.json']:
        shutil.copy(f'{d}/{f}', f'{OUT}/runs_r34/V0_seed{s}/{f}')
    print(open(f'{d}/results.json').read()[:400], flush=True)
ks = [(f'ks_r34_m{s}_p{p}', [py, '-u', 'ks_eval.py', '--data', 'data', '--ks', 'data_ks', '--runs', 'runs_r34', '--model-seed', str(s), '--out', f'runs_ks_r34_p{p}', '--methods', 'none,tent,adabn,adabn_physa', '--variants', 'V0', '--part', str(p), '--nparts', '2']) for p in range(2) for s in range(3)]
run_jobs(ks, 'event')
for p in range(2):
    shutil.copytree(f'{W}/runs_ks_r34_p{p}', f'{OUT}/runs_ks_r34_p{p}', dirs_exist_ok=True)
subprocess.call([py, '-u', 'bb_analyse.py', '--hist', 'runs_ks_r34_p*/ks_hist_m{s}_v2.npz', '--title', 'Kuro Siwo, ResNet-34 U-Net', '--out', OUT + '/ks_r34.md'], cwd=W)
print('R34_KS_DONE')
