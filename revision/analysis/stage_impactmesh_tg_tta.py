"""Stage the frozen ImpactMesh pairs and verified STANet weights for Kaggle."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import rasterio

from stage_impactmesh_kaggle import HERE, REV, RUN, digest, read, sha


PRIOR = REV / "generated" / "impactmesh_prior_methods"
OUT = PRIOR / "impactmesh_tg_tta"
DATASET = "mostofaseum/px1-impactmesh-tg-tta-paired-inputs"
SOURCE = PRIOR / "stanet_source_gpu_full" / "verified_v1"


def stage():
    base = OUT / "dataset"
    base.mkdir(parents=True)
    archive_path = base / "px1_impactmesh_tg_tta_inputs.zip"
    sys.path.insert(0, str(REV.parent / "src"))
    sys.path.insert(0, str(HERE))
    from verify_inputs import zarr
    import hand_fetch

    def local_hand(lat, lon):
        path = HERE / "hand_sources" / hand_fetch.tile_url(lat, lon, hand_fetch.NAME)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    hand_fetch.download_tile = local_hand
    cohort = read(HERE / "selected.json")
    pins = read(HERE / "input_verification.json")
    temporal = read(PRIOR / "temporal_input_audit.json")["chips"]
    index = {}
    for root in sorted((HERE / "impactmesh_inputs").iterdir()):
        radar = read(root / "S1RTC" / "verified.json")["chips"]
        dem = read(root / "DEM" / "verified.json")["chips"]
        for chip, radar_hash in radar.items():
            if chip in index:
                raise ValueError(f"Duplicate chip {chip}")
            index[chip] = (root, radar_hash, dem[chip])
    chips = [chip for items in cohort.values() for chip in items]
    assert len(cohort) == 13 and len(chips) == 516 and len(set(chips)) == 516
    assert set(chips) == set(pins) == set(index)

    def encode(chip):
        root, radar_hash, dem_hash = index[chip]
        radar_path = root / "S1RTC" / f"{chip}_S1RTC.zarr.zip"
        dem_path = root / "DEM" / f"{chip}_DEM.tif"
        event_path = HERE / "prepared" / f"{chip}.npz"
        lab_path = RUN / "labels" / f"{chip}.npy"
        pin = pins[chip]
        assert sha(radar_path) == radar_hash and sha(dem_path) == dem_hash
        assert sha(event_path) == pin["input_sha256"] and sha(lab_path) == pin["label_sha256"]
        with zarr.ZipStore(str(radar_path), mode="r") as store:
            group = zarr.open_consolidated(store, mode="r")
            flags = group["data_mask"][:]
            times = group["time"][:]
            assert flags[1] == flags[2] == 1 and times[1] < times[2] == pin["timestamp_coordinate"]
            bands = [str(v).lower() for v in group["band"][:]]
            raw = group["bands"][1].astype(np.float32)
            pre_valid = (~group["nan_mask"][1].astype(bool) & np.isfinite(raw).all(0)
                         & (raw > -49.9).all(0))
            pre_vv = raw[bands.index("vv")]
        with np.load(event_path, allow_pickle=False) as existing:
            event_radar, radar_valid, original_terrain_valid = existing["raw"], existing["rv"], existing["hv"]
        assert radar_valid.any() and not (radar_valid & ~pre_valid).any()
        with rasterio.open(dem_path) as ds:
            assert ds.crs.is_projected and ds.crs.linear_units == "metre" and ds.shape == radar_valid.shape
            elev = ds.read(1, masked=True).astype(np.float32).filled(np.nan)
            hand = hand_fetch.hand_on_grid(ds.transform, ds.crs, ds.shape, preserve_nodata=True)
            gy, gx = np.gradient(elev, abs(ds.transform.e), abs(ds.transform.a))
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        hv = np.isfinite(elev) & np.isfinite(slope) & np.isfinite(hand) & (hand >= 0) & (hand < 1e6)
        assert np.array_equal(hv, original_terrain_valid), chip
        assert np.median(pre_vv[radar_valid]) < 0
        stream = BytesIO()
        np.savez_compressed(stream, pre=pre_vv, event=event_radar, valid=radar_valid,
                            terrain_valid=hv, hand=hand, slope=slope)
        return chip, stream.getvalue(), lab_path

    verified = read(SOURCE / "verification.json")
    assert verified["status"] == "passed"
    spec = {"events": cohort, "input": {}, "label": {}, "source": {}, "dataset": DATASET,
            "temporal_audit_sha256": sha(PRIOR / "temporal_input_audit.json"),
            "checkpoint_verification_sha256": sha(SOURCE / "verification.json")}
    for activation, membership in cohort.items():
        for chip in membership:
            assert temporal[activation][chip]["pre_event"]["joint_pixels"] == pins[chip]["radar_valid"]
    with ZipFile(archive_path, "x", compression=ZIP_STORED) as bundle, ThreadPoolExecutor(3) as pool:
        for seed in verified["seeds"]:
            key = str(seed["seed"])
            location = SOURCE / "stanet_source_full" / f"seed{key}" / "model.pt"
            assert sha(location) == seed["checkpoint_sha256"]
            spec["source"][key] = seed
            bundle.write(location, f"checkpoints/seed{key}.pt")
        assert set(spec["source"]) == {"0", "1", "2"}
        for i, (chip, data, lab_path) in enumerate(pool.map(encode, chips), 1):
            spec["input"][chip], spec["label"][chip] = digest(data), pins[chip]["label_sha256"]
            bundle.writestr(f"pairs/{chip}.npz", data)
            bundle.write(lab_path, f"labels/{chip}.npy")
            if i % 50 == 0:
                print(f"Staged {i}/516 paired chips", flush=True)
        bundle.writestr("inputs.json", json.dumps(spec, indent=2))
    (OUT / "inputs.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (OUT / "dataset.sha256").write_text(sha(archive_path) + "\n", encoding="ascii")
    (base / "dataset-metadata.json").write_text(json.dumps({"id": DATASET,
        "title": "PX1 ImpactMesh TG TTA paired inputs", "licenses": [{"name": "other"}]}), encoding="utf-8")
    print(f"Staged {len(chips)} chips, three verified checkpoints, {archive_path.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    stage()
