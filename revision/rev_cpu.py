"""Kaggle runner of the revision experiments (CPU part); started by the notebook cell in revision/README.md."""
import os, sys, glob, time, shutil, subprocess, zipfile
py = sys.executable
ST = '/tmp/px1_stage'
FILES = [f for f in os.listdir(ST) if f.endswith('.py') and not f.startswith('rev_cpu') and not f.startswith('rev_gpu')]
for z in glob.glob('/kaggle/input/**/px1_part*.zip', recursive=True):
    with zipfile.ZipFile(z) as zf:
        zf.extractall('/tmp/px1_hist', [n for n in zf.namelist() if 'pc_hist' in n])
print('inputs:', sorted(os.listdir('/kaggle/input')), 'files:', FILES, flush=True)
subprocess.call([py, '-u', os.path.join(ST, 'rev_analyse.py'), '--root', '/kaggle/input', '--extra', '/tmp/px1_hist', '--out', '/kaggle/working/out_rev'])
print('REV_CPU_DONE')
