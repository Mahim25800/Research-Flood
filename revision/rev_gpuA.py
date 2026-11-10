"""Kaggle runner A (GPU): EATA, SAR and CoTTA on MMFlood with the three U-Net models."""
exec(open('/tmp/px1_stage/rev_common.py').read())
assert subprocess.call([py, '-c', f"import sys; sys.path.insert(0, {code!r}); import run_kaggle; run_kaggle.setup('mmflood')"]) == 0
patch_all()
print('MMFlood chips:', len(glob.glob(W + '/data_mm/**/*.npz', recursive=True)), '| setup min', round((time.time() - T0) / 60), flush=True)
jobs = [(f'mm_m{s}_p{p}', [py, '-u', 'mm_eval.py', '--data', 'data', '--mm', 'data_mm', '--runs', 'runs', '--model-seed', str(s), '--part', str(p),
                          '--nparts', '2', '--out', 'runs_mm_rev', '--methods', 'eata,sar,cotta'], {}) for p in range(2) for s in range(3)]
run_jobs(jobs, 'event', 10.6)
keep('runs_mm_rev')
for f in glob.glob(W + '/data_mm/*.csv'):
    shutil.copy(f, OUT)
print('REV_GPUA_DONE, min', round((time.time() - T0) / 60))
