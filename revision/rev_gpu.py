"""Kaggle runner of the revision experiments (GPU part); started by the notebook cell in revision/README.md."""
import os, sys, glob, time, shutil, subprocess, zipfile
py = sys.executable
ST = '/tmp/px1_stage'
FILES = [f for f in os.listdir(ST) if f.endswith('.py') and not f.startswith('rev_cpu') and not f.startswith('rev_gpu')]
for z in glob.glob('/kaggle/input/**/px1_part*.zip', recursive=True):
    with zipfile.ZipFile(z) as zf:
        zf.extractall('/tmp/px1_hist', [n for n in zf.namelist() if 'pc_hist' in n])
print('inputs:', sorted(os.listdir('/kaggle/input')), 'files:', FILES, flush=True)
T0 = time.time()
W, OUT = '/tmp/px1', '/kaggle/working/out'
os.makedirs(OUT + '/logs', exist_ok=True)
code = os.path.dirname(glob.glob('/kaggle/input/**/run_kaggle.py', recursive=True)[0])
assert subprocess.call([py, os.path.join(ST, 'r34_patch.py'), code, '--check']) == 0
assert subprocess.call([py, os.path.join(ST, 'rev_patch.py'), code, '--check']) == 0
assert subprocess.call([py, '-c', f"import sys; sys.path.insert(0, {code!r}); import run_kaggle; run_kaggle.setup()"]) == 0
for k in FILES:
    shutil.copy(os.path.join(ST, k), W)
assert subprocess.call([py, 'r34_patch.py', '.'], cwd=W) == 0
assert subprocess.call([py, 'rev_patch.py', '.'], cwd=W) == 0
print('Kuro Siwo chips:', len(glob.glob(W + '/data_ks/*_GRD/*/*.npz')), '| setup min', round((time.time() - T0) / 60), flush=True)

def run_jobs(jobs, key, deadline):
    def start(i):
        name, cmd = jobs[i]
        return subprocess.Popen(cmd, cwd=W, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % 2)), stdout=open(f'{OUT}/logs/{name}.txt', 'a'), stderr=subprocess.STDOUT)
    procs, retried = {i: start(i) for i in range(len(jobs))}, set()
    while procs:
        time.sleep(240)
        late = time.time() - T0 > deadline
        for i, pr in list(procs.items()):
            name = jobs[i][0]
            lines = [l for l in open(f'{OUT}/logs/{name}.txt').read().splitlines() if key in l]
            print(time.strftime('%H:%M'), name, pr.poll(), len(lines), lines[-1][:110] if lines else '', flush=True)
            if pr.poll() is not None:
                del procs[i]
                if pr.returncode != 0 and i not in retried and not late:
                    retried.add(i)
                    print(open(f'{OUT}/logs/{name}.txt').read()[-1500:], flush=True)
                    procs[i] = start(i)
            elif late:
                pr.terminate(); del procs[i]
                print('stopped at the time limit:', name, flush=True)

M = 'eata,sar,cotta,none,tent,adabn,adabn_physa'
ks = [(f'ks_m{s}_p{p}', [py, '-u', 'ks_eval.py', '--data', 'data', '--ks', 'data_ks', '--runs', 'runs', '--model-seed', str(s), '--out', f'runs_ks_rev_p{p}', '--methods', M, '--variants', 'V0', '--part', str(p), '--nparts', '2']) for p in range(2) for s in range(3)]
run_jobs(ks, 'event', 4.1 * 3600)
ft = [(f'ft_m{s}', [py, '-u', 'ks_finetune.py', '--data', 'data', '--ks', 'data_ks', '--runs', 'runs', '--model-seed', str(s), '--out', 'runs_ks_ft']) for s in range(3)]
run_jobs(ft, 'fold', 4.7 * 3600)
for d in ['runs_ks_rev_p0', 'runs_ks_rev_p1', 'runs_ks_ft']:
    if os.path.isdir(f'{W}/{d}'):
        shutil.copytree(f'{W}/{d}', f'{OUT}/{d}', dirs_exist_ok=True)
print('GPU part done, min', round((time.time() - T0) / 60), flush=True)
subprocess.call([py, '-u', os.path.join(ST, 'rev_analyse.py'), '--root', '/kaggle/input', '--extra', '/tmp/px1_hist', '--extra2', OUT, '--out', OUT + '/rev'])
print('REV_GPU_DONE')
