"""Shared helpers of the second-round Kaggle runners (exec'd by them)."""
import os, sys, glob, time, shutil, subprocess
py = sys.executable
ST, W, OUT = '/tmp/px1_stage', '/tmp/px1', '/kaggle/working/out'
T0 = time.time()
os.makedirs(OUT + '/logs', exist_ok=True)
os.makedirs('/tmp/px1_override', exist_ok=True)
code = os.path.dirname(glob.glob('/kaggle/input/**/run_kaggle.py', recursive=True)[0])
assert subprocess.call([py, os.path.join(ST, 'r34_patch.py'), code, '--check']) == 0
assert subprocess.call([py, os.path.join(ST, 'rev_patch.py'), code, '--check']) == 0
for f in ['mm_prep.py', 'mm_eval.py', 'ks_events.json', 'bb_analyse.py', 'rev_analyse.py', 'r34_patch.py', 'rev_patch.py', 'rev_patch_b.py']:
    shutil.copy(os.path.join(ST, f), '/tmp/px1_override')


def patch_all():
    for p in ['r34_patch.py', 'rev_patch.py', 'rev_patch_b.py']:
        assert subprocess.call([py, p, '.'], cwd=W) == 0


def run_jobs(jobs, key, deadline_h, per_gpu=3):
    """jobs: (name, command, extra environment). At most per_gpu jobs per GPU; stops everything at the deadline."""
    queue, procs, retried = list(range(len(jobs))), {}, set()

    def start(i, gpu):
        name, cmd, env = jobs[i]
        return subprocess.Popen(cmd, cwd=W, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), **env),
                                stdout=open(f'{OUT}/logs/{name}.txt', 'a'), stderr=subprocess.STDOUT), gpu
    while queue or procs:
        late = time.time() - T0 > deadline_h * 3600
        for gpu in (0, 1):
            while queue and not late and sum(1 for p, g in procs.values() if g == gpu) < per_gpu:
                i = queue.pop(0)
                procs[i] = start(i, gpu)
        if late:
            queue = []
        time.sleep(240)
        for i, (pr, gpu) in list(procs.items()):
            name = jobs[i][0]
            lines = [l for l in open(f'{OUT}/logs/{name}.txt').read().splitlines() if key in l]
            print(time.strftime('%H:%M'), name, pr.poll(), len(lines), lines[-1][:110] if lines else '', flush=True)
            if pr.poll() is not None:
                del procs[i]
                if pr.returncode != 0:
                    print(open(f'{OUT}/logs/{name}.txt').read()[-1200:], flush=True)
                    if i not in retried and not late:
                        retried.add(i)
                        procs[i] = start(i, gpu)
            elif late:
                pr.terminate(); del procs[i]
                print('stopped at the time limit:', name, flush=True)


def keep(*dirs):
    for d in dirs:
        if os.path.isdir(f'{W}/{d}'):
            shutil.copytree(f'{W}/{d}', f'{OUT}/{d}', dirs_exist_ok=True)
