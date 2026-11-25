"""Kaggle-only ResNet-50 follow-up on the archived GEOID-Flood selection."""

from collections import defaultdict
from types import SimpleNamespace

from sklearn.metrics import average_precision_score, roc_curve


ARMS = {
    "source": "none_valid",
    "adabn": "adabn_valid",
    "post": "adabn_physa_valid_post",
    "ordinary": "adabn_st_valid",
    "darkness": "adabn_physa_valid",
    "joint": "adabn_physa_valid",
}


def materialize_models(mount, code, work, spec):
    from contextlib import ExitStack

    with ExitStack() as stack:
        archives = list(mount.rglob("px1_impactmesh_followup.zip"))
        if archives:
            (path,) = archives
            assert sha(path) == spec["impactmesh_archive_sha256"]
            archive = stack.enter_context(ZipFile(path))
        else:
            (path,) = [p for p in mount.rglob("protocol.json")
                       if sha(p) == spec["impactmesh_protocol_sha256"]]
            root = path.parent
        checkpoints = {}
        for name, expected in spec["impactmesh_files"].items():
            source = archive.open(name) if archives else (root / name).open("rb")
            dest = code / name if name.endswith(".py") else work / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Restore the ImpactMesh adaptation implementation in the temporary
            # copy of the GEOID loader so both ResNet-50 cohorts use the same code.
            with source, dest.open("wb") as writer:
                shutil.copyfileobj(source, writer)
            assert sha(dest) == expected, name
            if name.endswith(".pt"):
                checkpoints[int(dest.stem[-1])] = dest
    assert set(checkpoints) == {0, 1, 2}
    return checkpoints


def prepare_selected(work, root, spec, evaluator, hand_fetch):
    tiles = {name: {"id": name, "activation": row["activation"]}
             for name, row in spec["tiles"].items()}
    for entry in spec["shards"]:
        path = PurePosixPath(entry["path"])
        layer = path.parts[-2]
        assert layer in {"s1grd", "dem", "label"} and path.parts[:3] == ("geoid-flood-heldout", "shards", "test")
        local = work / path.name
        fetch(HF + entry["path"], local, entry["lfs"]["oid"], entry["size"])
        extract(local, layer, root, tiles)
        local.unlink()
    hand_fetch.TILE_DIR = str(work / "hand_source")
    if not hand_fetch.probe():
        raise RuntimeError("Pinned HAND source is unavailable")
    for name, expected in spec["tiles"].items():
        tile = tiles[name]
        for layer in ("s1_post", "dem", "label"):
            assert tile[layer] == expected[layer], (name, layer)
            assert sha(root / tile[layer]) == spec["raster_hashes"][tile[layer]], tile[layer]
        tile["hand"] = expected["hand"]
        if tile["hand"] is None:
            continue
        _, _, (crs, affine, height, width) = evaluator.chips(root, tile)
        hand = hand_fetch.hand_on_grid(affine, crs, (height, width), preserve_nodata=True)
        path = root / tile["hand"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=1,
                           dtype="float32", crs=crs, transform=affine, nodata=np.nan,
                           compress="deflate") as stream:
            stream.write(hand, 1)
        assert sha(path) == spec["raster_hashes"][tile["hand"]], tile["hand"]
    for path in Path(hand_fetch.TILE_DIR).glob("*.tif"):
        assert sha(path) == spec["hand_sources"][path.name], path.name
    print("GEOID_INPUTS_VERIFIED", len(spec["raster_hashes"]), flush=True)
    return tiles


def load_archived_event(evaluator, root, tiles, windows):
    event = defaultdict(list)
    labels = []
    for name, y, x in windows:
        tile = tiles[name]
        raw, valid, grid = evaluator.chips(root, tile)
        dem, _ = evaluator.raster(root, tile["dem"], grid)
        dem = dem[0].astype("float32").filled(np.nan)
        hand = np.full(dem.shape, np.nan, dtype="float32")
        if tile["hand"]:
            values, _ = evaluator.raster(root, tile["hand"], grid)
            hand = values[0].astype("float32").filled(np.nan)
        north, east = np.gradient(dem, 10)
        slope = np.degrees(np.arctan(np.hypot(north, east)))
        terrain_valid = np.isfinite(dem) & np.isfinite(slope) & np.isfinite(hand) & (hand >= 0)
        terrain = np.stack([np.zeros_like(dem), np.nan_to_num(slope, nan=90) / 10,
                            np.where(terrain_valid, np.clip(hand, 0, 100), 100) / 50])
        truth, _ = evaluator.raster(root, tile["label"], grid)
        truth = truth[0].filled(255)
        assert np.isin(truth, [0, 1, 2, 255]).all()
        window = np.s_[y:y + 512, x:x + 512]
        assert valid[window].sum() >= 512 * 512 // 2
        event["raw"].append(raw[:, window[0], window[1]].copy())
        event["radar_valid"].append(valid[window].copy())
        event["terrain_valid"].append(terrain_valid[window].copy())
        event["physics_terrain"].append(terrain[:, window[0], window[1]].astype("float32"))
        labels.append(np.where(valid[window], truth[window], 255).astype("uint8"))
    event["lab"] = [np.full(y.shape, 255, dtype="uint8") for y in labels]
    event["clean"] = [False] * len(labels)
    return event, np.stack(labels)


def flood_metrics(truth, score, threshold):
    pred = score >= threshold
    tp, fp, fn, tn = (int(mask.sum()) for mask in
                      (pred & truth, pred & ~truth, ~pred & truth, ~pred & ~truth))
    fpr, recall, _ = roc_curve(truth, score)
    return dict(ap=float(average_precision_score(truth, score)), iou=tp / (tp + fp + fn),
                miss=fn / (tp + fn), fpr=fp / (fp + tn),
                matched_recall=np.interp([.02, .05], fpr, recall).tolist(),
                tp=tp, fp=fp, fn=fn, tn=tn)


def run_resnet(spec, spec_sha):
    if not torch.cuda.is_available() or not Path("/kaggle/input").is_dir():
        raise RuntimeError("This run requires Kaggle GPU")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    work = Path("/tmp/px1_geoid_resnet")
    out = Path("/kaggle/working/geoid_resnet")
    for path in (work, out):
        path.mkdir()
    assert spec["release"] == RELEASE and spec["geoid_protocol_sha256"] == PROTOCOL_HASH
    assert hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest() == spec_sha
    fetch(PROTOCOL_URL, work / "protocol.zip", PROTOCOL_HASH)
    code = work / "protocol"
    with ZipFile(work / "protocol.zip") as archive:
        archive.extractall(code)
    checkpoints = materialize_models(Path("/kaggle/input"), code, work, spec)
    sys.path[:0] = [str(code / "revision"), str(code / "revision/fresh_sar_test")]
    import geoid_evaluate as evaluator
    import hand_fetch
    import tta_px1
    from experiment import FloodResNet

    assert evaluator.run_one is tta_px1.run_one
    assert (tta_px1.STEPS, tta_px1.ADABN_BATCHES, tta_px1.P_HI, tta_px1.P_LO) == (200, 50, .9, .1)
    assert set(spec["selected"]) == evaluator.ACTIVATIONS
    root = work / "rasters"
    root.mkdir()
    tiles = prepare_selected(work, root, spec, evaluator, hand_fetch)
    ref = spec["calibration"]
    mean, std = (np.asarray(ref[k], dtype="float32") for k in ("mean", "std"))
    run = dict(spec_sha256=spec_sha, gpu=torch.cuda.get_device_name(0),
               torch=torch.__version__, cuda=torch.version.cuda, steps=spec["steps"],
               adaptation_seed=spec["adaptation_seed"], threshold=ref["threshold"])
    save(out / "run.json", run)
    events = {}
    for activation, windows in spec["selected"].items():
        event, labels = load_archived_event(evaluator, root, tiles, windows)
        keep = [(label == 0) | (label == 2) for label in labels]
        truth = np.concatenate([label[mask] == 2 for label, mask in zip(labels, keep)])
        assert [int(truth.sum()), int((~truth).sum())] == spec["support"][activation]
        path = out / f"{activation}_truth.npy"
        np.save(path, truth, allow_pickle=False)
        rows = {}
        for seed, checkpoint in checkpoints.items():
            for arm, method in ARMS.items():
                collected = []

                def observe(index, score):
                    assert index == len(collected) and score.dtype == np.float32
                    collected.append(score[keep[index]])

                args = SimpleNamespace(seed=spec["adaptation_seed"], steps=spec["steps"],
                                       model_factory=FloodResNet, score_observer=observe,
                                       darkness_only=(arm == "darkness"))
                evaluator.run_one(method, "V0", activation, event, mean, std, args,
                                  "cuda", str(checkpoint))
                scores = np.concatenate(collected)
                assert scores.shape == truth.shape and np.isfinite(scores).all()
                assert 0 <= scores.min() <= scores.max() <= 1
                score_file = out / f"{activation}_{seed}_{arm}.npz"
                np.savez_compressed(score_file, scores=scores)
                rows[f"{seed}:{arm}"] = dict(**flood_metrics(truth, scores, ref["threshold"]),
                                             score_sha256=sha(score_file), score_file=score_file.name,
                                             diagnostics=args.adaptation_stats)
                print("GEOID_RESNET_ARM_DONE", activation, seed, arm, flush=True)
        events[activation] = dict(selected=windows, truth_sha256=sha(path), rows=rows)
        save(out / f"{activation}_metrics.json", events[activation])
    save(out / "results.json", dict(run=run, events=events))
    print("GEOID_RESNET_DONE", len(events), flush=True)
