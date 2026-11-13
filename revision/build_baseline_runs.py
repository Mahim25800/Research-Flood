"""Package reviewed code into six self-contained Kaggle notebooks, without launching.

Every full notebook runs CPU mechanism tests and a real-data smoke test first.
No existing output directory may be reused. No held-out external dataset is supported.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def payload():
    names = ['conformal_px1.py', 'tta_px1.py', 'ks_eval.py', 'sen1floods11_unet_baseline.py',
             'train_px1.py', 'download_sen1floods11.py', 'dem_fetch.py', 'hand_fetch.py', 'ks_download.py']
    files = {name: (ROOT / 'src' / name).read_text(encoding='utf-8') for name in names}
    for name in ['baselines_v2.py', 'eval_baselines_v2.py', 'test_baselines_v2.py',
                 'verify_baseline_outputs.py', 'mm_eval.py', 'mm_prep.py', 'mm_range_zip.py', 'ks_events.json']:
        files[name] = (ROOT / 'revision' / name).read_text(encoding='utf-8')
    files['run_kaggle.py'] = (ROOT / 'kaggle/run_kaggle.py').read_text(encoding='utf-8')
    # Add only alternate-label scoring to the staged legacy control, not old baseline patches.
    sys.path.insert(0, str(ROOT / 'revision'))
    from rev_patch import HIST_OLD, HIST_NEW
    if files['tta_px1.py'].count(HIST_OLD) != 1:
        raise ValueError('Legacy control histogram patch mismatch')
    files['tta_px1.py'] = files['tta_px1.py'].replace(HIST_OLD, HIST_NEW)
    for name, content in files.items():
        if name.endswith('.py'):
            compile(content, name, 'exec')
    return files


BOOT = r'''
import os, sys, json, hashlib, subprocess, time, uuid, shutil
from pathlib import Path
import torch
if not Path('/kaggle/input').is_dir() or not torch.cuda.is_available():
    raise RuntimeError('Run this notebook on Kaggle GPU; it is not a PC training launcher')
START = time.monotonic()
RUN_ID = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
OUT = Path('/kaggle/working/baselines_v2') / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)
ST = OUT / 'code'
ST.mkdir()
for name, content in FILES.items():
    (ST / name).write_text(content, encoding='utf-8')
fingerprints = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in FILES.items()}
manifest = dict(schema='px1-baselines-v2-run', dataset=DATASET, mode=MODE, status='started',
                run_id=RUN_ID, files=fingerprints, jobs=[], cuda=torch.version.cuda, torch=torch.__version__)
def record():
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
record()
W = Path('/tmp/px1')
active = []
logs = []
def execute_group(jobs):
    global active, logs
    active, logs = [], []
    try:
        for name, args, gpu in jobs:
            log = open(OUT / (name + '.log'), 'w')
            logs.append(log)
            command = [sys.executable, '-u', str(W / 'eval_baselines_v2.py'), '--dataset', DATASET,
                       '--work', str(W), '--out', str(OUT / name)] + args
            proc = subprocess.Popen(command, cwd=W, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'), stdout=log, stderr=subprocess.STDOUT)
            active.append((name, proc))
        while any(proc.poll() is None for _, proc in active):
            failures = [(name, proc.returncode) for name, proc in active if proc.poll() not in (None, 0)]
            if failures:
                raise RuntimeError(f'Failed jobs: {failures}')
            if time.monotonic() - START > 10.5 * 3600:
                raise TimeoutError('Notebook budget exceeded; partial results retained')
            time.sleep(10)
        failures = [(name, proc.returncode) for name, proc in active if proc.returncode != 0]
        if failures:
            raise RuntimeError(f'Failed jobs: {failures}')
        for name, _ in active:
            child = json.loads((OUT / name / 'manifest.json').read_text())
            if child['status'] != 'completed' or len(child['records']) != len(child['events']):
                raise RuntimeError(f'Incomplete output: {name}')
            verify_command = [sys.executable, str(W / 'verify_baseline_outputs.py'), str(OUT / name)]
            if name.startswith('smoke'):
                verify_command.append('--allow-smoke')
            subprocess.run(verify_command, cwd=W, check=True)
            manifest['jobs'].append(dict(name=name, status='completed'))
            print('Completed', name, flush=True)
        record()
    finally:
        for _, proc in active:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        for log in logs:
            log.close()

try:
    subprocess.run([sys.executable, str(ST / 'test_baselines_v2.py')], cwd=ST, check=True,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    if W.exists() or Path('/tmp/px1_override').exists():
        raise RuntimeError('Use a fresh Kaggle session: refusing to overwrite a prior PX1 workspace')
    override = Path('/tmp/px1_override')
    override.mkdir()
    for name in FILES:
        shutil.copy2(ST / name, override / name)
    sys.path.insert(0, str(ST))
    import run_kaggle
    run_kaggle.OUT = str(OUT / 'setup')
    # Only existing project datasets. S1F additionally needs terrain for proposed-method controls.
    if DATASET == 'ks':
        run_kaggle.setup('physa')
    elif DATASET == 'mm' and MODE == 'smoke':
        run_kaggle.setup('setup')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'inflate64==1.0.4'], cwd=W, check=True)
        with (OUT / 'setup/logs/mm_smoke.txt').open('w') as log:
            subprocess.run([sys.executable, '-u', 'mm_prep.py', '--smoke'], cwd=W,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    elif DATASET == 'mm':
        run_kaggle.setup('mmflood')
    else:
        run_kaggle.setup('setup')
        subprocess.run([sys.executable, 'dem_fetch.py', '--data', 'data', '--workers', '8'], cwd=W, check=True)
        subprocess.run([sys.executable, 'hand_fetch.py', '--data', 'data', '--sen1floods11', '--workers', '8'], cwd=W, check=True)
    if DATASET == 'ks':
        # setup('physa') also prepares source terrain; no future dataset access.
        pass
    # Freeze exact runtime packages for this notebook output, not just requested requirements.
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (OUT / 'environment.txt').write_text(freeze)
    execute_group([('smoke_seed0', ['--seed', '0', '--smoke'], 0)])
    if MODE == 'full':
        seeds = [0] if DATASET == 's1f' else [0, 1, 2]
        devices = torch.cuda.device_count()
        for offset in range(0, len(seeds), devices):
            execute_group([(f'full_seed{s}', ['--seed', str(s)], gpu)
                           for gpu, s in enumerate(seeds[offset:offset+devices])])
    manifest['status'] = 'completed'
except BaseException as exc:
    manifest['status'], manifest['error'] = 'failed', repr(exc)
    raise
finally:
    manifest['seconds'] = time.monotonic() - START
    record()
print('BASELINES_V2_NOTEBOOK_DONE', OUT, flush=True)
'''


def build():
    files = payload()
    output = ROOT / 'revision/generated/baselines_v2'
    output.mkdir(parents=True, exist_ok=True)
    for dataset in ('ks', 's1f', 'mm'):
        for mode in ('smoke', 'full'):
            source = f'DATASET = {dataset!r}\nMODE = {mode!r}\nFILES = {files!r}\n' + BOOT
            compile(source, f'{dataset}_{mode}', 'exec')
            notebook = dict(nbformat=4, nbformat_minor=5,
                cells=[dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=source)],
                metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                    language_info=dict(name='python'), kaggle=dict(accelerator='nvidiaTeslaT4',
                        isGpuEnabled=True, isInternetEnabled=True, language='python', sourceType='notebook',
                        dataSources=[dict(sourceId=21857658, sourceType='datasetVersion')])))
            path = output / f'{dataset}_{mode}.ipynb'
            path.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
            print(path)
    (output / 'sources.json').write_text(json.dumps({name: hashlib.sha256(content.encode()).hexdigest()
        for name, content in files.items()}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    build()
