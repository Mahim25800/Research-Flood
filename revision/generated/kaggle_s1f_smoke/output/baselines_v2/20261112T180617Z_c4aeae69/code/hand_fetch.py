"""
Fetch HAND (height above nearest drainage, GLO-30 HAND v1/2021, CC0, ASF/AWS) onto every chip grid.

  Sen1Floods11: data/HAND/<chip>_HAND.tif on the S1 chip grid (same layout as data/DEM)
  Kuro Siwo:    data_ks_hand/<split>/<actid>/<chip>.npy, float32 224x224, chip grid taken from the
                footprint in data_ks/index.csv (EPSG:3857, 224 px of 10 m)

HAND is the vertical distance (m) from a pixel to the nearest drainage cell it flows to: water
ponds near HAND 0, so it is the physically meaningful version of our "height above the chip's
low ground". Pixels without a tile (open sea) are set to 0.

Usage:  python hand_fetch.py [--sen1floods11] [--kurosiwo] [--workers 8]
Prints HAND_DONE at the end; resumable (existing outputs are skipped).
"""
import argparse
import csv
import math
import os
import re
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import threading

socket.setdefaulttimeout(120)
_locks, _locks_guard = {}, threading.Lock()
_stats = {"chips_without_hand": 0}
TILE_DIR = os.path.join("data", "HAND_tiles")

os.environ.setdefault("GDAL_CACHEMAX", "64")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

BASE = "https://glo-30-hand.s3.us-west-2.amazonaws.com/v1/2021/"
NAME = "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_HAND.tif"
PATTERNS = [BASE + NAME, BASE + NAME.replace("_HAND.tif", "_HAND/") + NAME]  # flat, or one folder per tile
NAME_ONLY = NAME
_pattern = [None]


def tile_url(lat, lon, pattern):
    return pattern.format(ns="N" if lat >= 0 else "S", lat=abs(lat), ew="E" if lon >= 0 else "W", lon=abs(lon))


def download_tile(lat, lon):
    """Download one HAND tile in full (cached, one download per tile even with many threads)."""
    name = tile_url(lat, lon, NAME)
    with _locks_guard:
        lock = _locks.setdefault(name, threading.Lock())
    with lock:
        return _download_tile(lat, lon, name)


def _download_tile(lat, lon, name):
    dst = os.path.join(TILE_DIR, name)
    if os.path.exists(dst):
        return dst
    if os.path.exists(dst + ".none"):
        return None
    os.makedirs(TILE_DIR, exist_ok=True)
    url = tile_url(lat, lon, _pattern[0])
    for t in range(4):
        try:
            urllib.request.urlretrieve(url, dst + ".part")
            os.replace(dst + ".part", dst)
            return dst
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):  # S3 answers 403/404 for keys that do not exist (open sea)
                open(dst + ".none", "w").close()
                return None
            err = e
        except Exception as e:
            err = e
        time.sleep(5 * (t + 1))
    print(f"  tile {name} failed: {type(err).__name__} {err}", flush=True)
    return None


def probe():
    """Find which URL layout the bucket uses, by downloading a land tile in Bangladesh (N23 E090)."""
    for p in PATTERNS:
        _pattern[0] = p
        url = tile_url(23, 90, p)
        try:
            path = download_tile(23, 90)
            if path is None:
                raise IOError("missing")
            with rasterio.open(path) as src:
                v = src.read(1, out_shape=(1, 64, 64))
            print(f"HAND layout ok: {url}  mean {float(np.nanmean(v)):.2f} m, size {os.path.getsize(path) / 2**20:.0f} MB", flush=True)
            return True
        except Exception as e:
            print(f"  not {url}: {type(e).__name__} {str(e)[:200]}", flush=True)
            for ext in ("", ".none", ".part"):
                f = os.path.join(TILE_DIR, tile_url(23, 90, NAME)) + ext
                if os.path.exists(f):
                    os.remove(f)
    return False


def hand_on_grid(transform, crs, shape):
    left, top = transform.c, transform.f
    right, bottom = left + transform.a * shape[1], top + transform.e * shape[0]
    b = transform_bounds(crs, "EPSG:4326", left, bottom, right, top) if not crs.is_geographic else (left, bottom, right, top)
    out = np.full(shape, np.nan, np.float32)
    for lat in range(math.floor(b[1]), math.floor(b[3]) + 1):
        for lon in range(math.floor(b[0]), math.floor(b[2]) + 1):
            path = download_tile(lat, lon)
            if path is None:
                continue
            try:
                with rasterio.open(path) as src:
                    part = np.full(shape, np.nan, np.float32)
                    reproject(rasterio.band(src, 1), part, dst_transform=transform, dst_crs=crs,
                              dst_nodata=np.nan, resampling=Resampling.bilinear)
                    out = np.where(np.isnan(out), part, out)
            except Exception as e:
                print(f"  reading {path} failed: {e}", flush=True)
                continue
    if np.all(np.isnan(out)):
        _stats["chips_without_hand"] += 1
    return np.nan_to_num(out, nan=0.0)


def do_s1f(args):
    from sen1floods11_unet_baseline import read_split
    chips = [s for k in ["train", "valid", "test", "bolivia"] for s, _ in read_split(args.data, k)]
    os.makedirs(os.path.join(args.data, "HAND"), exist_ok=True)

    def one(s):
        dst = os.path.join(args.data, "HAND", s.replace("S1Hand", "HAND"))
        if os.path.exists(dst):
            if not args.verify:
                return 0
            with rasterio.open(dst) as d:
                if np.any(d.read(1) != 0):
                    return 0
            dem_p = os.path.join(args.data, "DEM", s.replace("S1Hand", "DEM"))
            if os.path.exists(dem_p):
                with rasterio.open(dem_p) as d:
                    if float(np.nanstd(d.read(1))) < 2.0:
                        return 0  # flat (sea or delta): an all-zero HAND is plausible
            print(f"  re-fetching blank HAND for {s}", flush=True)
        with rasterio.open(os.path.join(args.data, "S1Hand", s)) as ref:
            prof, tr, crs, shape = ref.profile, ref.transform, ref.crs, (ref.height, ref.width)
        h = hand_on_grid(tr, crs, shape)
        prof.update(count=1, dtype="float32", nodata=None, compress="deflate")
        with rasterio.open(dst + ".part.tif", "w", **prof) as d:
            d.write(h, 1)
        os.replace(dst + ".part.tif", dst)
        return 1

    with ThreadPoolExecutor(args.workers) as ex:
        n = sum(ex.map(one, chips))
    print(f"Sen1Floods11: {n} new HAND chips ({len(chips)} total)", flush=True)


def do_ks(args):
    rows = list(csv.DictReader(open(os.path.join(args.ks, "index.csv"))))
    crs = rasterio.crs.CRS.from_epsg(3857)

    def one(r):
        dst = os.path.join(args.ks_hand, r["file"].replace(".npz", ".npy"))
        if os.path.exists(dst):
            if not args.verify or np.any(np.load(dst) != 0):
                return 0
            z = np.load(os.path.join(args.ks, r["file"]))
            if float(np.nanstd(z["dem"])) < 2.0:
                return 0
            print(f"  re-fetching blank HAND for {r['file']}", flush=True)
        nums = [float(v) for v in re.findall(r"-?\d+\.?\d*", r["geom"])]
        xs, ys = nums[0::2], nums[1::2]
        tr = from_bounds(min(xs), min(ys), max(xs), max(ys), 224, 224)
        h = hand_on_grid(tr, crs, (224, 224))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.save(dst + ".part.npy", h)
        os.replace(dst + ".part.npy", dst)
        return 1

    with ThreadPoolExecutor(args.workers) as ex:
        n = sum(ex.map(one, rows))
    print(f"Kuro Siwo: {n} new HAND chips ({len(rows)} total)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ks", default="data_ks")
    ap.add_argument("--ks-hand", default="data_ks_hand")
    ap.add_argument("--sen1floods11", action="store_true")
    ap.add_argument("--kurosiwo", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify", action="store_true", help="re-fetch chips whose HAND is all zero over land with relief")
    a = ap.parse_args()
    if not (a.sen1floods11 or a.kurosiwo):
        a.sen1floods11 = a.kurosiwo = True
    if not probe():
        raise SystemExit("HAND bucket not reachable with the expected file names")
    if a.sen1floods11:
        do_s1f(a)
    if a.kurosiwo:
        do_ks(a)
    print(f"chips with no HAND coverage (all sea or failed): {_stats['chips_without_hand']}", flush=True)
    print("HAND_DONE", flush=True)
