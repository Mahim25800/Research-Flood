"""Build an offline-packaged Kaggle diagnostic; does not launch or download data.

Run from any directory: python revision/build_tent_audit.py
Choose existing-data dataset with PX1_AUDIT_DATASET=ks|s1f|mm inside the notebook.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    name: (ROOT / path).read_text(encoding='utf-8') for name, path in {
        'tta_px1.py': 'src/tta_px1.py',
        'ks_eval.py': 'src/ks_eval.py',
        'r34_patch.py': 'revision/r34_patch.py',
        'rev_patch.py': 'revision/rev_patch.py',
        'mm_eval.py': 'revision/mm_eval.py',
        'mm_prep.py': 'revision/mm_prep.py',
        'ks_events.json': 'revision/ks_events.json',
    }.items()
}
RUNNER = r'''
import os, sys, glob, time, json, hashlib, shutil, subprocess
from pathlib import Path
py = sys.executable
dataset = os.environ.get('PX1_AUDIT_DATASET', 'ks')
if dataset not in ('ks', 's1f', 'mm'):
    raise ValueError('PX1_AUDIT_DATASET must be ks, s1f or mm')
W = Path('/tmp/px1')
OUT = Path('/kaggle/working/tent_batch_audit') / dataset
OUT.mkdir(parents=True, exist_ok=True)
manifest = dict(dataset=dataset, method='tent_batch', status='started',
                schedule='event-wise 200 steps; Adam 1e-3; final per-chip batch statistics',
                files={k: hashlib.sha256(v.encode()).hexdigest() for k,v in FILES.items()})
def record():
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
record()
bundle = glob.glob('/kaggle/input/**/run_kaggle.py', recursive=True)
if len(bundle) != 1:
    raise RuntimeError(f'Expected exactly one PX1 code bundle, found {len(bundle)}')
sys.path.insert(0, str(Path(bundle[0]).parent))
import run_kaggle
processes = []
handles = []
try:
    run_kaggle.setup('mmflood') if dataset == 'mm' else run_kaggle.setup()
    for name, content in FILES.items():
        (W / name).write_text(content)
    for name in ('r34_patch.py', 'rev_patch.py'):
        subprocess.run([py, name, '.'], cwd=W, check=True)
    jobs = []
    if dataset == 'ks':
        for seed in range(3):
            for part in range(2):
                jobs.append((f'ks_s{seed}_p{part}', part, [py, '-u', 'ks_eval.py', '--data', 'data',
                    '--ks', 'data_ks', '--runs', 'runs', '--model-seed', str(seed), '--out', f'runs_ks_tentbatch_p{part}',
                    '--methods', 'tent_batch', '--variants', 'V0', '--part', str(part), '--nparts', '2']))
    elif dataset == 'mm':
        for seed in range(3):
            for part in range(2):
                jobs.append((f'mm_s{seed}_p{part}', part, [py, '-u', 'mm_eval.py', '--data', 'data',
                    '--mm', 'data_mm', '--runs', 'runs', '--model-seed', str(seed), '--out', 'runs_mm_tentbatch',
                    '--methods', 'tent_batch', '--part', str(part), '--nparts', '2']))
    else:
        jobs.append(('s1f', 0, [py, '-u', 'tta_px1.py', '--data', 'data', '--runs', 'runs_loeo',
                              '--methods', 'tent_batch', '--model-seed', '0']))
    # At most one process per GPU. A failed job fails the notebook; preserve partial histograms.
    for offset in range(0, len(jobs), 2):
        processes = []
        handles = []
        for name, gpu, command in jobs[offset:offset+2]:
            handle = open(OUT / (name + '.log'), 'w')
            handles.append(handle)
            processes.append((name, subprocess.Popen(command, cwd=W,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=handle, stderr=subprocess.STDOUT)))
        failed = [(name, proc.wait()) for name, proc in processes]
        for handle in handles:
            handle.close()
        if any(code != 0 for _, code in failed):
            raise RuntimeError(f'Failed diagnostic jobs: {failed}')
        print('Completed:', failed, flush=True)
    manifest['status'] = 'completed'
except BaseException as exc:
    manifest['status'] = 'failed'
    manifest['error'] = repr(exc)
    raise
finally:
    for _, proc in processes:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
    for handle in handles:
        handle.close()
    patterns = {'ks': 'runs_ks_tentbatch_p*', 'mm': 'runs_mm_tentbatch', 's1f': 'runs_loeo'}
    for directory in W.glob(patterns[dataset]):
        for file in directory.glob('*.npz'):
            destination = OUT / directory.name
            destination.mkdir(exist_ok=True)
            shutil.copy2(file, destination / file.name)
    record()
print('TENT_BATCH_AUDIT_DONE', flush=True)
'''


if __name__ == '__main__':
    source = 'FILES = ' + repr(FILES) + '\n' + RUNNER
    compile(source, 'tent_batch_audit', 'exec')
    out = ROOT / 'revision/generated'
    out.mkdir(exist_ok=True)
    notebook = dict(nbformat=4, nbformat_minor=5, cells=[dict(cell_type='code', metadata={},
        execution_count=None, outputs=[], source=source)], metadata={
        'kernelspec': dict(name='python3', display_name='Python 3', language='python'),
        'language_info': dict(name='python'),
        'kaggle': dict(accelerator='nvidiaTeslaT4', isGpuEnabled=True, isInternetEnabled=True,
            language='python', sourceType='notebook', dataSources=[dict(sourceId=21857658, sourceType='datasetVersion')])})
    path = out / 'tent_batch_audit.ipynb'
    path.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    (out / 'tent_batch_audit_sources.json').write_text(json.dumps({name: hashlib.sha256(content.encode()).hexdigest()
        for name, content in FILES.items()}, indent=2), encoding='utf-8')
    print(path)
