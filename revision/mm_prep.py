"""
MMFlood final-test preparation (pre-registered 9 Nov 2026; see the PX1 doc).

  1. download mmflood.zip (Zenodo 6534637, 11.3 GB) and extract
  2. drop events overlapping Kuro Siwo (300 km and 365 days) or Sen1Floods11 (300 km, same or adjacent year)
  3. per image: dB conversion if linear, resample 20 m -> 10 m, cut 512 x 512 chips with >= 50% valid radar
  4. at most 48 chips per event, spread evenly over the event's chips; events with < 8 chips dropped
  5. labels: 1 water, 0 dry, 255 ignore (invalid radar; OSM permanent water that the mask calls dry)
  6. HAND from GLO-30 HAND on each chip grid

Writes data_mm/<event>/<chip>.npz (vv, vh dB float16; dem float32; hand float32; lab uint8),
data_mm/index.csv and data_mm/excluded.csv. Prints MM_PREP_DONE.
"""
import argparse
from contextlib import ExitStack
import csv
import glob
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

os.environ.setdefault("GDAL_CACHEMAX", "256")
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform_bounds

URL = "https://zenodo.org/records/6534637/files/mmflood.zip?download=1"
CHIP, MAX_CHIPS, MIN_CHIPS, MIN_VALID = 512, 48, 8, 0.5
S1F_YEARS = {"Bolivia": 2018, "Ghana": 2018, "India": 2016, "Mekong": 2018, "Nigeria": 2018, "Pakistan": 2017,
             "Paraguay": 2018, "Somalia": 2018, "Spain": 2019, "Sri-Lanka": 2017, "USA": 2019}


def km(lon1, lat1, lon2, lat2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(min(1, a)))


def s1f_centroids(data):
    pts = {}
    for f in glob.glob(os.path.join(data, "S1Hand", "*.tif")):
        e = os.path.basename(f).split("_")[0]
        with rasterio.open(f) as src:
            b = src.bounds if src.crs.is_geographic else transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        pts.setdefault(e, []).append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    return {e: (np.mean([p[0] for p in v]), np.mean([p[1] for p in v])) for e, v in pts.items()}


def download(root):
    z = os.path.join(root, "mmflood.zip")
    if not os.path.exists(os.path.join(root, "extracted.ok")):
        os.makedirs(root, exist_ok=True)
        for t in range(4):
            if subprocess.call(["wget", "-q", "-c", "-O", z, URL]) == 0:
                break
        print(f"downloaded {os.path.getsize(z) / 2**30:.1f} GB", flush=True)
        subprocess.check_call(["unzip", "-q", "-o", z, "-d", root])
        os.remove(z)
        open(os.path.join(root, "extracted.ok"), "w").close()
    for p in sorted(glob.glob(os.path.join(root, "*")) + glob.glob(os.path.join(root, "*", "*")))[:20]:
        print(f"  layout: {p}", flush=True)
    if not glob.glob(os.path.join(root, "**", "s1_raw"), recursive=True):  # archives inside the archive
        for z in glob.glob(os.path.join(root, "**", "*.zip"), recursive=True) + glob.glob(os.path.join(root, "**", "*.tar*"), recursive=True):
            print(f"  extracting inner archive {z}", flush=True)
            if z.endswith(".zip"):
                subprocess.call(["unzip", "-q", "-o", z, "-d", os.path.dirname(z)])
            else:
                subprocess.call(["tar", "-xf", z, "-C", os.path.dirname(z)])
    acts = glob.glob(os.path.join(root, "**", "activations.json"), recursive=True)
    # event folders = parents of every s1_raw folder, wherever the zip put them
    dirs = {os.path.basename(os.path.dirname(p)): os.path.dirname(p)
            for p in glob.glob(os.path.join(root, "**", "s1_raw"), recursive=True) if os.path.isdir(p)}
    print(f"activations.json: {acts[:2]}; {len(dirs)} event folders", flush=True)
    if not dirs:
        raise SystemExit("no s1_raw folders found in the extracted archive")
    return (acts[0] if acts else None), dirs


def to_db(a):
    a = a.astype(np.float32)
    v = a[np.isfinite(a) & (a != 0)]
    if v.size and np.median(v) > 0:  # linear power -> dB
        a = np.where(a > 0, 10 * np.log10(np.maximum(a, 1e-6)), np.nan)
    return a


def resample(src, band, transform, crs, shape, method):
    out = np.full(shape, np.nan, np.float32)
    reproject(src.read(band).astype(np.float32), out, src_transform=src.transform, src_crs=src.crs,
              dst_transform=transform, dst_crs=crs, src_nodata=src.nodata, dst_nodata=np.nan, resampling=method)
    return out


def image_chips(base, name):
    """All valid 512 x 512 chips of one MMFlood image, at 10 m."""
    with rasterio.open(os.path.join(base, "s1_raw", name)) as s1:
        crs, t = s1.crs, s1.transform
        lat = (s1.bounds.top + s1.bounds.bottom) / 2
        res_m = abs(t.a) * (111320 * math.cos(math.radians(lat)) if crs.is_geographic else 1)
        f = max(1.0, res_m / 10.0)
        shape = (int(s1.height * f), int(s1.width * f))
        tt = t * Affine.scale(1 / f)
        vv = to_db(resample(s1, 1, tt, crs, shape, Resampling.bilinear))
        vh = to_db(resample(s1, 2, tt, crs, shape, Resampling.bilinear))
    with rasterio.open(os.path.join(base, "mask", name)) as m:
        mask = resample(m, 1, tt, crs, shape, Resampling.nearest)
    with rasterio.open(os.path.join(base, "DEM", name)) as d:
        dem = resample(d, 1, tt, crs, shape, Resampling.bilinear)
    hp = os.path.join(base, "hydro", name)
    hydro = None
    if os.path.exists(hp):
        with rasterio.open(hp) as h:
            hydro = resample(h, 1, tt, crs, shape, Resampling.nearest)
    valid = np.isfinite(vv) & np.isfinite(vh) & (vv > -49) & (vh > -49)
    vals = set(np.unique(mask[np.isfinite(mask)]).astype(int).tolist())
    water_val = 255 if vals <= {0, 255} and 255 in vals else 1  # masks stored as 0/255 or 0/1(/255 nodata)
    lab = np.full(shape, 255, np.uint8)
    lab[(mask == 0) & valid] = 0
    lab[(mask == water_val) & valid] = 1
    if hydro is not None:
        lab[(hydro > 0) & (lab == 0)] = 255
    chips = []
    for r in range(0, shape[0] - CHIP + 1, CHIP):
        for c in range(0, shape[1] - CHIP + 1, CHIP):
            v = valid[r:r + CHIP, c:c + CHIP]
            if v.mean() < MIN_VALID:
                continue
            d = dem[r:r + CHIP, c:c + CHIP]
            d = np.nan_to_num(d, nan=float(np.nanmedian(d)) if np.isfinite(d).any() else 0.0)
            chips.append({"vv": np.nan_to_num(vv[r:r + CHIP, c:c + CHIP], nan=-50.0), "vh": np.nan_to_num(vh[r:r + CHIP, c:c + CHIP], nan=-50.0),
                          "dem": d, "lab": lab[r:r + CHIP, c:c + CHIP], "transform": tuple(tt * Affine.translation(c, r))[:6], "crs": crs.to_wkt()})
    return chips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="mmflood_raw")
    ap.add_argument("--out", default="data_mm")
    ap.add_argument("--data", default="data")
    ap.add_argument("--ks-events", default="ks_events.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="Fetch only enough eligible regions to prepare one smoke region")
    a = ap.parse_args()
    with ExitStack() as stack:
        prepare(a, stack)


def prepare(a, stack):
    if a.smoke:
        from mm_range_zip import RangeFile, read_member

        source = stack.enter_context(RangeFile(URL))
        archive = stack.enter_context(ZipFile(source))
        members = set(archive.namelist())
        radar = {}
        for name in sorted(members):
            path = PurePosixPath(name)
            if path.suffix == '.tif' and path.parent.name == 's1_raw':
                radar.setdefault(path.parent.parent.name, []).append(path)
        dirs = {code: str(Path(a.raw) / paths[0].parent.parent) for code, paths in radar.items()}
        meta_path = Path(a.raw) / 'activations.json'
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_bytes(read_member(source, archive, 'activations.json'))
        print(f'MMFlood smoke indexed {len(dirs)} regions; downloading eligible regions on demand', flush=True)
    else:
        meta_path, dirs = download(a.raw)
    meta = {}
    if meta_path:
        txt = open(meta_path).read()
        try:
            meta = json.loads(txt)
        except json.JSONDecodeError:  # tolerate trailing commas
            import re
            meta = json.loads(re.sub(r",\s*([}\]])", r"\1", txt))
    if isinstance(meta, list):
        meta = {m.get("code") or m.get("activation") or str(i): m for i, m in enumerate(meta)}
    acts = {}
    for d in sorted(dirs):
        info = meta.get(d) or meta.get(d.split("-")[0]) or {}
        if "lon" not in info or "lat" not in info:  # fall back to the first image's centre
            if a.smoke:
                # Use regions with published coordinates for the smoke diagnostic.
                continue
            tifs = sorted(glob.glob(os.path.join(dirs[d], "s1_raw", "*.tif")))
            if tifs:
                with rasterio.open(tifs[0]) as src:
                    b = src.bounds if src.crs.is_geographic else transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                info = dict(info, lon=(b[0] + b[2]) / 2, lat=(b[1] + b[3]) / 2)
        acts[d] = {"lon": info.get("lon", 0), "lat": info.get("lat", 0), "start": str(info.get("start") or info.get("end") or "2000-01-01"),
                   "country": info.get("country"), "subset": info.get("subset")}
    print(f"metadata example: {next(iter(meta.items()), None)}", flush=True)
    ks = json.load(open(a.ks_events))
    s1f = s1f_centroids(a.data)
    keep, excl = [], []
    for code, info in sorted(acts.items()):
        lon, lat, date = float(info["lon"]), float(info["lat"]), info["start"][:10]
        why = []
        for k, e in ks.items():
            if km(lon, lat, e["lon"], e["lat"]) <= 300:
                dd = abs((np.datetime64(date) - np.datetime64(e["date"])).astype(int))
                if dd <= 365:
                    why.append(f"KuroSiwo {k} ({dd} days)")
        for e, (elon, elat) in s1f.items():
            if km(lon, lat, elon, elat) <= 300 and abs(int(date[:4]) - S1F_YEARS.get(e, 0)) <= 1:
                why.append(f"Sen1Floods11 {e}")
        (excl if why else keep).append((code, info, "; ".join(why)))
    print(f"{len(acts)} activations; {len(excl)} excluded for overlap; {len(keep)} kept", flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "excluded.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "country", "start", "reason"])
        for code, info, why in excl:
            w.writerow([code, info.get("country"), info["start"][:10], why])

    import hand_fetch
    hand_fetch.probe()
    rows = []

    def do_event(item):
        code, info, _ = item
        base = dirs[code]
        if a.smoke:
            print(f'Fetching MMFlood smoke candidate {code}: {len(radar[code])} radar scenes', flush=True)
            for path in radar[code]:
                for modality in ('s1_raw', 'DEM', 'mask', 'hydro'):
                    member = path.parent.parent / modality / path.name
                    if modality == 'hydro' and str(member) not in members:
                        continue
                    destination = Path(base) / modality / path.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(read_member(source, archive, str(member)))
            print(f'MMFlood smoke transferred {source.transferred / 1024**2:.1f} MiB from ZIP', flush=True)
        if not os.path.isdir(base):
            return []
        evdir = os.path.join(a.out, code)
        if os.path.exists(os.path.join(evdir, "done.ok")):
            return [r for r in csv.DictReader(open(os.path.join(evdir, "chips.csv")))]
        chips = []
        for name in sorted(os.listdir(os.path.join(base, "s1_raw"))):
            try:
                chips += [(name, ch) for ch in image_chips(base, name)]
            except Exception as e:
                print(f"  {code}/{name}: {type(e).__name__} {e}", flush=True)
        if len(chips) < MIN_CHIPS:
            return []
        if len(chips) > MAX_CHIPS:
            idx = np.linspace(0, len(chips) - 1, MAX_CHIPS).round().astype(int)
            chips = [chips[i] for i in sorted(set(idx))]
        os.makedirs(evdir, exist_ok=True)
        out = []
        for i, (name, ch) in enumerate(chips):
            tr = Affine(*ch["transform"])
            hand = hand_fetch.hand_on_grid(tr, rasterio.crs.CRS.from_wkt(ch["crs"]), (CHIP, CHIP))
            fn = os.path.join(evdir, f"{i:03d}.npz")
            np.savez_compressed(fn, vv=ch["vv"].astype(np.float16), vh=ch["vh"].astype(np.float16), dem=ch["dem"].astype(np.float32),
                                hand=hand.astype(np.float32), lab=ch["lab"])
            out.append({"event": code, "file": fn, "image": name, "country": info.get("country"), "start": info["start"][:10],
                        "water_px": int((ch["lab"] == 1).sum()), "dry_px": int((ch["lab"] == 0).sum())})
        with open(os.path.join(evdir, "chips.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader(); w.writerows(out)
        open(os.path.join(evdir, "done.ok"), "w").close()
        print(f"  {code} ({info.get('country')}): {len(out)} chips, water px {sum(r['water_px'] for r in out)}, dry px {sum(r['dry_px'] for r in out)}", flush=True)
        return out

    if a.smoke:
        for candidate in keep:
            rows = do_event(candidate)
            if rows:
                break
        if not rows:
            raise ValueError('No eligible MMFlood smoke region produced enough chips')
    else:
        with ThreadPoolExecutor(a.workers) as ex:
            for r in ex.map(do_event, keep):
                rows += r
    with open(os.path.join(a.out, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["event", "file", "image", "country", "start", "water_px", "dry_px"])
        w.writeheader(); w.writerows(rows)
    n_ev = len({r["event"] for r in rows})
    print(f"MMFlood: {n_ev} events, {len(rows)} chips", flush=True)
    print("MM_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
