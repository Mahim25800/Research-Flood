"""Builds the two Kaggle notebooks of the revision run (analysis on CPU, experiments on GPU)."""
import base64, json, os, sys, zlib
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.dirname(HERE)

def pack(names):
    return {n: base64.b64encode(zlib.compress(open(os.path.join(P, n), "rb").read(), 9)).decode() for n in names}

HEAD = '''import os, sys, glob, time, shutil, subprocess, zipfile, base64, zlib
py = sys.executable
ST = '/tmp/px1_stage'
os.makedirs(ST, exist_ok=True)
FILES = %s
for k, v in FILES.items():
    open(os.path.join(ST, k), 'wb').write(zlib.decompress(base64.b64decode(v)))
for z in glob.glob('/kaggle/input/**/px1_part*.zip', recursive=True):
    with zipfile.ZipFile(z) as zf:
        zf.extractall('/tmp/px1_hist', [n for n in zf.namelist() if 'pc_hist' in n])
print('inputs:', sorted(os.listdir('/kaggle/input')), flush=True)
'''
CPU = HEAD + '''subprocess.call([py, '-u', os.path.join(ST, 'rev_analyse.py'), '--root', '/kaggle/input', '--extra', '/tmp/px1_hist', '--out', '/kaggle/working/out_rev'])
print('REV_CPU_DONE')
'''
GPU = HEAD + '''T0 = time.time()
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
'''

def nb(src, gpu, sources):
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4" if gpu else "none", "dataSources": sources, "isInternetEnabled": True,
                       "language": "python", "sourceType": "notebook", "isGpuEnabled": gpu}}
    return {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}], "metadata": meta, "nbformat": 4, "nbformat_minor": 5}

SRC = [{"sourceId": 21857658, "sourceType": "datasetVersion"}, {"sourceId": 367062493, "sourceType": "kernelVersion"},
       {"sourceId": 366960729, "sourceType": "kernelVersion"}, {"sourceId": 367828676, "sourceType": "kernelVersion"}]
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
cpu = CPU.replace('%s', repr(pack(["rev_analyse.py"])), 1)
gpu = GPU.replace('%s', repr(pack(["rev_analyse.py", "rev_patch.py", "r34_patch.py", "ks_finetune.py"])), 1)
for name, src, g in [("rev_cpu", cpu, False), ("rev_gpu", gpu, True)]:
    compile(src, name, "exec")
    open(os.path.join(out, name + ".py"), "w").write(src)
    js = json.dumps(nb(src, g, SRC))
    open(os.path.join(out, name + ".ipynb"), "w").write(js)
    open(os.path.join(out, name + "_js.txt"), "w").write("window.__nb=" + json.dumps(js) + ";window.confirm=()=>true;window.__nb.length")
    print(name, len(src), len(js))
