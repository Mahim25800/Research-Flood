"""Kaggle runner B (GPU): EATA/SAR/CoTTA on Sen1Floods11 (leave one flood out) and a Sentinel-1-pretrained backbone."""
exec(open('/tmp/px1_stage/rev_common.py').read())
subprocess.call([py, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch', 'torchgeo'])
assert subprocess.call([py, '-c', "from torchgeo.models import ResNet50_Weights as R; sd = R.SENTINEL1_ALL_MOCO.get_state_dict(progress=False); print('weights ok', sd['conv1.weight'].shape)"]) == 0
assert subprocess.call([py, '-c', f"import sys; sys.path.insert(0, {code!r}); import run_kaggle; run_kaggle.setup()"]) == 0
patch_all()
prep = subprocess.Popen([py, '-u', 'mm_prep.py', '--raw', 'mmflood_raw', '--out', 'data_mm', '--data', 'data', '--ks-events', 'ks_events.json', '--workers', '2'],
                        cwd=W, stdout=open(OUT + '/logs/mm_prep.txt', 'w'), stderr=subprocess.STDOUT)
A = {'PX1_ARCH': 'r50s1'}
jobs = [('s1f_tta', [py, '-u', 'tta_px1.py', '--data', 'data', '--runs', 'runs_loeo', '--seed', '0', '--model-seed', '0', '--methods', 'eata,sar,cotta'], {})]
jobs += [(f'train_r50s1_s{s}', [py, '-u', 'train_px1.py', '--data', 'data', '--variant', 'V0', '--inputs', 's1', '--epochs', '60', '--lr', '3e-4',
                                '--seed', str(s), '--out', 'runs_r50s1', '--workers', '1'], A) for s in range(3)]
run_jobs(jobs, 'e', 3.0, per_gpu=2)
for f in glob.glob(W + '/runs_loeo/tta_*'):
    shutil.copy(f, OUT)
for s in range(3):
    d = f'{W}/runs_r50s1/V0_seed{s}'
    os.makedirs(f'{OUT}/runs_r50s1/V0_seed{s}', exist_ok=True)
    for f in ['model.pt', 'results.json']:
        shutil.copy(f'{d}/{f}', f'{OUT}/runs_r50s1/V0_seed{s}/{f}')
    print(open(f'{d}/results.json').read()[:300], flush=True)
M = 'none,tent,adabn,adabn_physa'
ks = [(f'ks_r50s1_m{s}_p{p}', [py, '-u', 'ks_eval.py', '--data', 'data', '--ks', 'data_ks', '--runs', 'runs_r50s1', '--model-seed', str(s), '--out', f'runs_ks_r50s1_p{p}',
                              '--methods', M, '--variants', 'V0', '--part', str(p), '--nparts', '2'], A) for p in range(2) for s in range(3)]
run_jobs(ks, 'event', 6.0)
keep('runs_ks_r50s1_p0', 'runs_ks_r50s1_p1')
subprocess.call([py, '-u', 'bb_analyse.py', '--hist', 'runs_ks_r50s1_p*/ks_hist_m{s}_v2.npz', '--title', 'Kuro Siwo, ResNet-50 U-Net (SSL4EO-S12 Sentinel-1 weights)', '--out', OUT + '/ks_r50s1.md'], cwd=W)
rc = prep.wait()
print('mm_prep exit', rc, open(OUT + '/logs/mm_prep.txt').read()[-400:], flush=True)
if rc == 0:
    mm = [(f'mm_r50s1_m{s}_p{p}', [py, '-u', 'mm_eval.py', '--data', 'data', '--mm', 'data_mm', '--runs', 'runs_r50s1', '--model-seed', str(s), '--part', str(p),
                                  '--nparts', '2', '--out', 'runs_mm_r50s1', '--methods', M], A) for p in range(2) for s in range(3)]
    run_jobs(mm, 'event', 11.0)
    keep('runs_mm_r50s1')
    subprocess.call([py, '-u', 'bb_analyse.py', '--hist', 'runs_mm_r50s1/mm_hist_m{s}_p*.npz', '--activation', '--title', 'MMFlood (activations), ResNet-50 U-Net (SSL4EO-S12 Sentinel-1 weights)', '--out', OUT + '/mm_r50s1_act.md'], cwd=W)
print('REV_GPUB_DONE, min', round((time.time() - T0) / 60))
