"""
Probe the Kuro Siwo GRD webdataset before committing to a large download.

Streams the first ~400 MB of one shard (nothing is saved except a few sample arrays),
reports download speed, free disk space, the sample format (arrays, dtypes, value ranges,
info.json) and whether samples inside a shard are grouped by flood event.
Usage:  python probe_kurosiwo.py [--split test_GRD] [--shard 0] [--mb 400]
"""
import argparse
import io
import json
import os
import shutil
import socket
import tarfile
import time
import urllib.request
from collections import Counter, OrderedDict

import numpy as np

socket.setdefaulttimeout(60)
BASE = "https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-Webdataset/resolve/main"


class Limited(io.RawIOBase):
    def __init__(self, resp, limit):
        self.resp, self.left, self.read_total = resp, limit, 0

    def readable(self):
        return True

    def readinto(self, b):
        if self.left <= 0:
            return 0
        data = self.resp.read(min(len(b), self.left, 1 << 20))
        n = len(data)
        b[:n] = data
        self.left -= n
        self.read_total += n
        return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_GRD")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--mb", type=int, default=400)
    ap.add_argument("--out", default="kurosiwo_probe")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for d in ["F:\\", "C:\\"]:
        try:
            u = shutil.disk_usage(d)
            print(f"{d} free {u.free / 2**30:.1f} GB of {u.total / 2**30:.1f} GB", flush=True)
        except Exception as e:
            print(d, e)
    url = f"{BASE}/{a.split}/shard-{a.shard:05d}.tar"
    req = urllib.request.Request(url, headers={"User-Agent": "px1-research-probe"})
    t0 = time.time()
    resp = urllib.request.urlopen(req)
    print("url", resp.geturl()[:120], "length", resp.headers.get("Content-Length"), flush=True)
    raw = Limited(resp, a.mb << 20)
    tf = tarfile.open(fileobj=io.BufferedReader(raw, 1 << 20), mode="r|")
    samples, order, shown, infos = OrderedDict(), [], 0, []
    try:
        for m in tf:
            if not m.isfile():
                continue
            key, _, field = m.name.partition(".")
            data = tf.extractfile(m).read()
            s = samples.setdefault(key, {})
            s[field] = data
            if "info.json" in s and key not in order:
                order.append(key)
                info = json.loads(s["info.json"])
                infos.append(info)
                if shown < 3:
                    print("sample", key, "info:", json.dumps(info)[:800], flush=True)
                    shown += 1
            if len(samples) > 50:
                samples.popitem(last=False)
            if len(order) <= 3 and len(s) >= 10:
                for f, b in s.items():
                    if f.endswith(".npy"):
                        arr = np.load(io.BytesIO(b))
                        fin = arr[np.isfinite(arr)]
                        print(f"  {key} {f}: {arr.dtype} {arr.shape} min {fin.min() if fin.size else 'nan'} "
                              f"max {fin.max() if fin.size else 'nan'} mean {fin.mean() if fin.size else 'nan'} "
                              f"uniq {np.unique(arr)[:6] if 'mask' in f else ''}", flush=True)
                np.savez_compressed(os.path.join(a.out, key.replace('/', '_') + ".npz"),
                                    **{f.replace('.npy', ''): np.load(io.BytesIO(b)) for f, b in s.items() if f.endswith('.npy')})
    except Exception as e:
        print("stream stopped:", type(e).__name__, e, flush=True)
    dt = time.time() - t0
    print(f"read {raw.read_total / 2**20:.0f} MB in {dt:.0f}s = {raw.read_total / 2**20 / max(dt, 1e-6):.1f} MB/s, "
          f"{len(order)} complete samples", flush=True)
    print("first keys:", order[:5], "last keys:", order[-5:], flush=True)
    for k in sorted({k for i in infos for k in i}):
        vals = [json.dumps(i.get(k))[:60] for i in infos]
        c = Counter(vals)
        print(f"info field {k}: {len(c)} distinct; top {c.most_common(5)}", flush=True)
        if 1 < len(c) <= 60:
            runs, prev = [], None
            for v in vals:
                if v != prev:
                    runs.append([v, 0]); prev = v
                runs[-1][1] += 1
            print(f"   run-lengths in stream order: {runs[:40]}", flush=True)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
