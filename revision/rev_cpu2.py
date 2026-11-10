"""Kaggle runner (CPU): change-detection reference on Kuro Siwo with the pre-flood images."""
import os, sys, glob, shutil, subprocess
py = sys.executable
ST, W = '/tmp/px1_stage', '/tmp/px1'
code = os.path.dirname(glob.glob('/kaggle/input/**/run_kaggle.py', recursive=True)[0])
os.makedirs(W, exist_ok=True)
for f in glob.glob(code + '/*.py'):
    shutil.copy(f, W)
shutil.copy(ST + '/ks_change.py', W)
for attempt in range(3):
    if subprocess.call([py, '-u', 'ks_download.py', '--splits', 'test_GRD', 'train_GRD', '--step-mb', '512', '--k', '6'], cwd=W, stdout=open('/kaggle/working/ks_download.txt', 'a'), stderr=subprocess.STDOUT) == 0:
        break
print(open('/kaggle/working/ks_download.txt').read()[-600:], flush=True)
print('Kuro Siwo chips:', len(glob.glob(W + '/data_ks/*_GRD/*/*.npz')), flush=True)
subprocess.call([py, '-u', 'ks_change.py', '--ks', 'data_ks', '--out', '/kaggle/working/out_change'], cwd=W)
print('REV_CPU2_DONE')
