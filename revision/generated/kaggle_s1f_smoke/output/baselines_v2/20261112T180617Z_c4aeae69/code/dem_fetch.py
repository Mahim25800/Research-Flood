"""Fetch the Copernicus GLO-30 DEM onto every Sen1Floods11 chip grid (data/DEM/<chip>_DEM.tif), as the PC cache did."""
import argparse
import math
import os
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("GDAL_CACHEMAX", "64")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds

DEM_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/"
           "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
           "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")


def fetch(s1_path, dst):
    if os.path.exists(dst):
        return 0
    with rasterio.open(s1_path) as ref:
        prof, b = ref.profile, ref.bounds
        if ref.crs.is_projected:
            b = transform_bounds(ref.crs, "EPSG:4326", *b)
    dem = np.full((prof["height"], prof["width"]), np.nan, np.float32)
    for lat in range(math.floor(b[1]), math.floor(b[3]) + 1):
        for lon in range(math.floor(b[0]), math.floor(b[2]) + 1):
            url = DEM_URL.format(ns="N" if lat >= 0 else "S", lat=abs(lat), ew="E" if lon >= 0 else "W", lon=abs(lon))
            for t in range(3):
                try:
                    with rasterio.open("/vsicurl/" + url) as src:
                        part = np.full_like(dem, np.nan)
                        reproject(rasterio.band(src, 1), part, dst_transform=prof["transform"], dst_crs=prof["crs"],
                                  dst_nodata=np.nan, resampling=Resampling.bilinear)
                        dem = np.where(np.isnan(dem), part, dem)
                    break
                except Exception as e:
                    if "404" in str(e):
                        break  # no tile (sea)
    if np.all(np.isnan(dem)):
        return 0
    dem = np.where(np.isnan(dem), np.nanmin(dem), dem)
    prof.update(count=1, dtype="float32", nodata=None, compress="deflate")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with rasterio.open(dst, "w", **prof) as d:
        d.write(dem, 1)
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    chips = sorted(f for f in os.listdir(os.path.join(a.data, "S1Hand")) if f.endswith(".tif"))
    with ThreadPoolExecutor(a.workers) as ex:
        n = sum(ex.map(lambda s: fetch(os.path.join(a.data, "S1Hand", s), os.path.join(a.data, "DEM", s.replace("S1Hand", "DEM"))), chips))
    print(f"DEM: {n} new chips of {len(chips)}", flush=True)
    print("DEM_DONE", flush=True)
