"""
Download a systematic subsample of the Kuro Siwo GRD webdataset (43 flood events) without
pulling the full 0.5+ TB.

Inside each tar shard, samples are stored event by event and are nearly fixed-size
(about 1.74 MB), so we jump through every shard with HTTP range requests: every STEP_MB
we read a short window and keep the first K complete samples found there. This gives an
evenly spread subsample of every event (roughly 1 in 50 patches), about 10 GB of traffic.

Kept per sample (data_ks/<split>/<actid>/<shard>_<offsetMB>_<key>.npz, compressed):
  vv, vh            post-flood backscatter, dB, float16
  vv_pre, vh_pre    first pre-flood image (sec1), dB, float16
  dem               float32 metres
  mask              uint8: 0 no water, 1 permanent water, 2 flood, 3 invalid
  valid             uint8
and info fields (actid, aoiid, flood_date, pflood, pwater, geom) in data_ks/index.csv.
Resumable: finished windows are listed in data_ks/progress.json.

Usage:  python ks_download.py [--splits test_GRD train_GRD] [--step-mb 512] [--k 6]
"""
import argparse
import csv
import io
import json
import os
import socket
import tarfile
import time
import urllib.request

import numpy as np

socket.setdefaulttimeout(60)
REPO = "orion-ai-lab/Kuro-Siwo-Webdataset"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
UA = {"User-Agent": "px1-research-download"}
FIELDS = ["flood_vv", "flood_vh", "sec1_vv", "sec1_vh", "dem", "mask", "valid_mask", "info"]


def get(url, rng=None, tries=4):
    for t in range(tries):
        try:
            h = dict(UA)
            if rng:
                h["Range"] = f"bytes={rng[0]}-{rng[1]}"
            with urllib.request.urlopen(urllib.request.Request(url, headers=h)) as r:
                return r.read()
        except Exception as e:
            if t == tries - 1:
                raise
            print(f"    retry {t + 1}: {type(e).__name__} {e}", flush=True)
            time.sleep(5 * (t + 1))


def list_shards(split):
    items = json.loads(get(f"{API}/{split}"))
    return sorted((it["path"], it["size"]) for it in items if it.get("type") == "file" and it["path"].endswith(".tar"))


def first_header(buf):
    for i in range(0, len(buf) - 512, 512):
        blk = buf[i:i + 512]
        if blk[257:262] != b"ustar":
            continue
        try:
            tarfile.TarInfo.frombuf(blk, tarfile.ENCODING, "surrogateescape")
            return i
        except Exception:
            continue
    return None


def to_db(x):
    return (10 * np.log10(np.clip(x, 1e-5, None))).astype(np.float16)


def parse_window(buf, k):
    i = first_header(buf)
    if i is None:
        return []
    out, cur, cur_key = [], {}, None
    try:
        tf = tarfile.open(fileobj=io.BytesIO(buf[i:]), mode="r|")
        for m in tf:
            if not m.isfile():
                continue
            key, _, field = m.name.partition(".")
            field = field.replace(".npy", "").replace(".json", "")
            if key != cur_key:
                if cur_key is not None and all(f in cur for f in FIELDS):
                    out.append((cur_key, cur))
                    if len(out) >= k:
                        return out
                cur, cur_key = {}, key
            if field in FIELDS:
                cur[field] = tf.extractfile(m).read()
    except (tarfile.ReadError, EOFError, OSError):
        pass  # window ends mid-member
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test_GRD", "train_GRD"])
    ap.add_argument("--step-mb", type=int, default=512)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", default="data_ks")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prog_path = os.path.join(a.out, "progress.json")
    done = set(json.load(open(prog_path))) if os.path.exists(prog_path) else set()
    idx_path = os.path.join(a.out, "index.csv")
    new_idx = not os.path.exists(idx_path)
    idx_f = open(idx_path, "a", newline="")
    idx = csv.writer(idx_f)
    if new_idx:
        idx.writerow(["split", "actid", "aoiid", "file", "flood_date", "pflood", "pwater", "water_px", "valid_px", "geom"])
    win = (a.k + 1) * int(1.8 * 2**20) + 2**20
    t0, total = time.time(), 0
    for split in a.splits:
        shards = list_shards(split)
        print(f"{split}: {len(shards)} shards, {sum(s for _, s in shards) / 2**30:.0f} GB", flush=True)
        for path, size in shards:
            shard = os.path.basename(path)[:-4]
            events = {}
            for off in range(0, size, a.step_mb << 20):
                tag = f"{split}/{shard}/{off >> 20}"
                if tag in done:
                    continue
                buf = get(f"{BASE}/{path}", (off, min(size - 1, off + win)))
                total += len(buf)
                for key, s in parse_window(buf, a.k):
                    info = json.loads(s["info"])
                    act = str(info.get("actid"))
                    arr = {f: np.load(io.BytesIO(s[f]))[0] for f in FIELDS if f != "info"}
                    mask = arr["mask"].astype(np.uint8)
                    d = os.path.join(a.out, split, act)
                    os.makedirs(d, exist_ok=True)
                    fn = f"{shard}_{off >> 20}_{key}.npz"
                    np.savez_compressed(os.path.join(d, fn), vv=to_db(arr["flood_vv"]), vh=to_db(arr["flood_vh"]),
                                        vv_pre=to_db(arr["sec1_vv"]), vh_pre=to_db(arr["sec1_vh"]),
                                        dem=arr["dem"].astype(np.float32), mask=mask,
                                        valid=arr["valid_mask"].astype(np.uint8))
                    idx.writerow([split, act, info.get("aoiid"), f"{split}/{act}/{fn}", info.get("flood_date"),
                                  info.get("pflood"), info.get("pwater"), int(((mask == 1) | (mask == 2)).sum()),
                                  int((mask != 3).sum()), info.get("geom")])
                    events[act] = events.get(act, 0) + 1
                idx_f.flush()
                done.add(tag)
                with open(prog_path + ".tmp", "w") as f:
                    json.dump(sorted(done), f)
                os.replace(prog_path + ".tmp", prog_path)
            el = time.time() - t0
            print(f"  {shard}: events {events}  | {total / 2**30:.2f} GB in {el / 60:.0f} min "
                  f"({total / 2**20 / max(el, 1):.1f} MB/s)", flush=True)
    idx_f.close()
    print("KS_DOWNLOAD_DONE", flush=True)


if __name__ == "__main__":
    main()
