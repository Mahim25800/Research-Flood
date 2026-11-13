"""
PX1 on Kaggle: one entry point for every remaining experiment.

  python run_kaggle.py --task physa      adaptive physics rule (per-flood radar thresholds + HAND)
                                         on Sen1Floods11 (11 held-out floods) and Kuro Siwo (3 models x 26 events)

Layout (everything big lives in /tmp, only results go to /kaggle/working/out):
  bundle  = the uploaded dataset (code, model weights, PC histograms), found anywhere under /kaggle/input
  WORK    = /tmp/px1   code + data + runs
  OUT     = /kaggle/working/out   logs, histograms, summaries (the notebook's saved output)

Data are downloaded here, not uploaded: Sen1Floods11 hand-labelled chips (Google Cloud), the Copernicus DEM and
GLO-30 HAND (AWS), and the same evenly spaced Kuro Siwo sample as on the PC (Hugging Face, step 512 MB, 6 per window).
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import threading
import time

WORK, OUT = "/tmp/px1", "/kaggle/working/out"


def sh(cmd, log=None, env=None, cwd=WORK):
    print(f"$ {cmd}", flush=True)
    if log:
        with open(log, "a") as f:
            return subprocess.call(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd)
    return subprocess.call(cmd, shell=True, env=env, cwd=cwd)


ROOTS = ["/kaggle/input", "/tmp/px1_bundle"]


def unpack_zips():
    """Kaggle may keep uploaded zips packed: extract any px1 zips into /tmp/px1_bundle."""
    import zipfile
    for z in glob.glob("/kaggle/input/**/px1_part*.zip", recursive=True):
        dst = os.path.join("/tmp/px1_bundle", os.path.basename(z)[:-4])
        if not os.path.exists(dst):
            print(f"extracting {z}", flush=True)
            with zipfile.ZipFile(z) as f:
                f.extractall(dst)


def rglob(pattern):
    return [p for r in ROOTS for p in glob.glob(os.path.join(r, "**", pattern), recursive=True)]


def find_bundle():
    unpack_zips()
    hits = rglob("run_kaggle.py")
    if not hits:
        raise SystemExit("bundle not found under /kaggle/input (add the px1 dataset to the notebook)")
    return os.path.dirname(hits[0])


def setup(task="physa"):
    t0 = time.time()
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    sh(f"{sys.executable} -m pip install -q rasterio", cwd="/tmp")
    code = find_bundle()
    for f in glob.glob(os.path.join(code, "*.py")):
        shutil.copy(f, WORK)
    for f in glob.glob("/tmp/px1_override/*"):  # newer code typed into the notebook overrides the bundle
        shutil.copy(f, WORK)
        print(f"override: {os.path.basename(f)}", flush=True)
    # model weights and PC histograms, wherever the zips were extracted
    for p in rglob("model.pt"):
        run, group = os.path.basename(os.path.dirname(p)), os.path.basename(os.path.dirname(os.path.dirname(p)))
        if group in ("runs", "runs_loeo"):
            d = os.path.join(WORK, group, run)
            os.makedirs(d, exist_ok=True)
            shutil.copy(p, os.path.join(d, "model.pt"))
    for p in rglob("clean_chips.txt"):
        os.makedirs(os.path.join(WORK, "audit"), exist_ok=True)
        shutil.copy(p, os.path.join(WORK, "audit", "clean_chips.txt"))
    for p in [p for p in rglob("*.npz") if os.path.basename(os.path.dirname(p)) == "pc_hist"]:
        name = os.path.basename(p)
        dst = os.path.join(WORK, "runs_loeo" if name.startswith("tta_hist") else "runs_ks", name.replace(".npz", "_local.npz"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(p, dst)
    n_models = len(glob.glob(os.path.join(WORK, "runs*", "*", "model.pt")))
    print(f"bundle: {code}; {n_models} models", flush=True)

    L = os.path.join(OUT, "logs", "setup.txt")
    py = sys.executable
    steps = [f"{py} -u download_sen1floods11.py --out data --workers 16"]
    if task == "physa":
        steps += [f"{py} -u dem_fetch.py --data data --workers 16",
                  f"{py} -u ks_download.py --splits test_GRD train_GRD --step-mb 512 --k 6",
                  f"{py} -u hand_fetch.py --data data --ks data_ks --workers 16"]
    if task == "mmflood":
        steps += [f"{py} -u mm_prep.py --raw mmflood_raw --out data_mm --data data --ks-events ks_events.json --workers 4"]
    for s in steps:
        for attempt in range(3):
            if sh(s, log=L) == 0:
                break
            print(f"  retry {attempt + 1}: {s}", flush=True)
        else:
            raise SystemExit(f"setup step failed: {s} (see {L})")
    print(f"setup done in {(time.time() - t0) / 60:.0f} min", flush=True)


def run_parallel(jobs, per_gpu=2):
    """jobs: list of (name, command). Runs per_gpu jobs at a time on each visible GPU."""
    import torch
    n_gpu = max(1, torch.cuda.device_count())
    # each GPU gets its own queue, jobs dealt round-robin, so no GPU sits idle while another has two jobs
    queues = {g: [j for i, j in enumerate(jobs) if i % n_gpu == g] for g in range(n_gpu)}
    slots = [g for g in range(n_gpu) for _ in range(per_gpu)]
    lock, status = threading.Lock(), {}

    def worker(gpu):
        while True:
            with lock:
                if not queues[gpu]:
                    return
                name, cmd = queues[gpu].pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            status[name] = "running"
            rc = sh(cmd, log=os.path.join(OUT, "logs", f"{name}.txt"), env=env)
            if rc != 0:  # resume once (the scripts skip finished work)
                rc = sh(cmd, log=os.path.join(OUT, "logs", f"{name}.txt"), env=env)
            status[name] = f"exit {rc}"

    threads = [threading.Thread(target=worker, args=(g,)) for g in slots]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(300)
        for name, _ in jobs:
            log = os.path.join(OUT, "logs", f"{name}.txt")
            last = ""
            if os.path.exists(log):
                with open(log) as f:
                    tail = [ln for ln in f.read().splitlines() if "event" in ln or "hold" in ln]
                last = tail[-1][:90] if tail else ""
            print(f"[{time.strftime('%H:%M')}] {name}: {status.get(name, 'queued')} | {last}", flush=True)
    for t in threads:
        t.join()
    return status


def task_physa():
    py = sys.executable
    m = "none,adabn_phys,physa,adabn_physa"  # none and adabn_phys re-computed as a reproducibility check
    jobs = [(f"ks_m{s}", f"{py} -u ks_eval.py --data data --ks data_ks --runs runs --model-seed {s} --out runs_ks --methods {m}")
            for s in range(3)]
    jobs.append(("s1f", f"{py} -u tta_px1.py --data data --runs runs_loeo --seed 0 --model-seed 0 --methods {m}"))
    status = run_parallel(jobs)
    print(status, flush=True)
    for p in glob.glob(os.path.join(WORK, "runs_loeo", "tta_*")) + glob.glob(os.path.join(WORK, "runs_ks", "ks_*")):
        shutil.copy(p, OUT)
    sh(f"{py} -u merge_analyse.py runs_loeo runs_ks {OUT}")


def task_mmflood(primary):
    py = sys.executable
    jobs = [(f"mm_m{s}_p{p}", f"{py} -u mm_eval.py --data data --mm data_mm --runs runs --model-seed {s} --part {p} --nparts 2 --out runs_mm")
            for p in range(2) for s in range(3)]
    status = run_parallel(jobs, per_gpu=3)
    print(status, flush=True)
    for p in glob.glob(os.path.join(WORK, "runs_mm", "*")) + glob.glob(os.path.join(WORK, "data_mm", "*.csv")):
        shutil.copy(p, OUT)
    sh(f"{py} -u mm_analyse.py --primary {primary} --runs runs_mm --mm data_mm --out {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["setup", "physa", "mmflood"])
    ap.add_argument("--primary", default="adabn_physa", choices=["adabn_physa", "adabn_phys"])
    a = ap.parse_args()
    setup(a.task)
    if a.task == "physa":
        task_physa()
    if a.task == "mmflood":
        task_mmflood(a.primary)
    print("KAGGLE_TASK_DONE", flush=True)
