"""Post-hoc GEOID-Flood model-free controls on the exact archived chip selection."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import time
from zipfile import ZipFile

import numpy as np
import requests

from impactmesh_review_controls import REV, read, sha


ARCHIVED = REV / "generated/geoid_v4_results_20261118/geoid_evaluation_v4"
OUT = REV / "generated/finish_line_b_20261124/geoid_cpu"
RELEASE = "4f1b9c52da789dbad9333e8f62c7b27ff48304bd"
PROTOCOL_HASH = "f9e01319fac638a35a22af8ed3a10517c82233dae9b5c46172f30ca22d503e1a"


def fetch(entry, destination):
    url = f"https://huggingface.co/datasets/links-ads/geoid-flood/resolve/{RELEASE}/{entry['path']}"
    size = entry["size"]
    step = 8 << 20
    if destination.exists() and destination.stat().st_size == size:
        assert sha(destination) == entry["lfs"]["oid"]
        return

    def fetch_range(start):
        end = min(size, start + step) - 1
        for attempt in range(4):
            try:
                params = {"offset": start}
                if attempt:
                    params["attempt"] = attempt
                with requests.get(url, params=params, stream=True,
                                  headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 45)) as response:
                    if response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}" or response.status_code != 206:
                        raise OSError(f"Invalid shard range response: {response.status_code}")
                    block = response.content
                if len(block) != end - start + 1:
                    raise OSError("Incomplete shard range")
                return block
            except (requests.RequestException, OSError):
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))

    with destination.open("a+b") as stream, ThreadPoolExecutor(8) as pool:
        offset = stream.tell() // step * step
        stream.truncate(offset)
        for batch in range(offset, size, step * 8):
            for block in pool.map(fetch_range, range(batch, min(batch + step * 8, size), step)):
                stream.write(block)
            stream.flush()
            if stream.tell() % (256 << 20) == 0:
                print(destination.name, stream.tell() >> 20, "MiB", flush=True)
    assert destination.stat().st_size == size and sha(destination) == entry["lfs"]["oid"]
    print("Verified shard", destination.name, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = REV / "generated/geoid_frozen_execution_20261117/px1-geoid-protocol-20261117-v4.zip"
    assert sha(protocol) == PROTOCOL_HASH
    code = OUT / "protocol"
    with ZipFile(protocol) as archive:
        archive.extractall(code)
    sys.path.insert(0, str(code / "revision"))
    from geoid_evaluate import chips, raster
    from tta_px1 import otsu_dark, valid_adaptive_dark

    inventory = read(ARCHIVED / "inventory.json")
    assert inventory["release"] == RELEASE
    tiles = {row["id"]: row for row in inventory["tiles"]}
    selection = read(ARCHIVED / "chip_selection.json")
    historical = read(ARCHIVED / "results/results.json")["events"]
    hashes = read(ARCHIVED / "raster_sha256.json")
    assert len(selection) == 9 and set(selection) == set(historical)
    assert all(selection[k]["selected"] == historical[k]["selected"] for k in selection)
    wanted = {tiles[tile][layer] for row in selection.values() for tile, _, _ in row["selected"]
              for layer in ("s1_post", "label")}
    by_name = {PurePosixPath(name).name: name for name in wanted}
    assert len(by_name) == len(wanted)
    rasters = OUT / "rasters"

    def extract(entry):
        path = PurePosixPath(entry["path"])
        local = OUT / path.name
        fetch(entry, local)
        with tarfile.open(local, mode="r|") as archive:
            for member in archive:
                name = by_name.get(PurePosixPath(member.name).name)
                if name and member.isfile():
                    destination = rasters / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, destination.open("wb") as sink:
                        shutil.copyfileobj(source, sink)
        local.unlink()  # Remove only the verified temporary shard created in this run.

    if any(not (rasters / n).exists() for n in wanted):
        layers = {PurePosixPath(n).parts[-2] for n in wanted if not (rasters / n).exists()}
        entries = [row for row in read(ARCHIVED / "shard_manifest.json")
                   if PurePosixPath(row["path"]).parts[-2] in layers]
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(extract, entries))
    for name in wanted:
        assert sha(rasters / name) == hashes[name], name

    results = {}
    for event, row in selection.items():
        raw, rv, labels = [], [], []
        for chip, y, x in row["selected"]:
            tile = tiles[chip]
            signal, valid, grid = chips(rasters, tile)
            label, _ = raster(rasters, tile["label"], grid)
            w = np.s_[y:y + 512, x:x + 512]
            raw.append(signal[:, w[0], w[1]].copy())
            rv.append(valid[w].copy())
            labels.append(np.where(valid[w], label[0].filled(255)[w], 255))
        y = np.stack(labels)
        keep = (y == 0) | (y == 2)
        truth = y[keep] == 2
        dry_hist, wet_hist = historical[event]["methods"]["0:none_valid"]["flood"]["histogram"]
        assert truth.sum() == sum(wet_hist) and (~truth).sum() == sum(dry_hist)
        vv, vh = valid_adaptive_dark(raw, rv)
        sample = np.concatenate([a[0].ravel()[::7][m.ravel()[::7]] for a, m in zip(raw, rv)])
        otsu, _ = otsu_dark(sample)
        pred = {"adaptive_vv_vh": np.stack([(a[0] < vv) & (a[1] < vh) & m for a, m in zip(raw, rv)]),
                "otsu_vv": np.stack([(a[0] < otsu) & m for a, m in zip(raw, rv)])}
        metrics = {}
        for arm, mask in pred.items():
            mask = mask[keep]
            tp, fp, fn = (int(m.sum()) for m in (mask & truth, mask & ~truth, ~mask & truth))
            metrics[arm] = {"iou": tp / (tp + fp + fn), "miss": fn / int(truth.sum()),
                            "fpr": fp / int((~truth).sum())}
        results[event] = {**metrics, "thresholds_db": [vv, vh, otsu],
                          "radar_dark_share": int(pred["adaptive_vv_vh"].sum()) / int(np.sum(rv))}
        print(event, "verified and scored", flush=True)
    means = {arm: {key: float(np.mean([row[arm][key] for row in results.values()]))
                   for key in ("iou", "miss", "fpr")} for arm in ("adaptive_vv_vh", "otsu_vv")}
    with (OUT / "model_free_results.json").open("x", encoding="utf-8") as stream:
        json.dump({"release": RELEASE, "selection_sha256": sha(ARCHIVED / "chip_selection.json"),
                   "verified_rasters": len(wanted), "means": means, "events": results}, stream, indent=2)


if __name__ == "__main__":
    main()
