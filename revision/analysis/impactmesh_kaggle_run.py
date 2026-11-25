"""Run the new ImpactMesh comparison on a private Kaggle GPU."""
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from sklearn.metrics import average_precision_score
import torch


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_event(root, chips, hashes):
    event = {key: [] for key in ("raw", "terr", "radar_valid", "terrain_valid")}
    scoring_masks = []
    labels = []
    for chip in chips:
        prepared = root / "prepared" / f"{chip}.npz"
        annotation = root / "labels" / f"{chip}.npy"
        assert sha(prepared) == hashes[chip]["input_sha256"], chip
        assert sha(annotation) == hashes[chip]["label_sha256"], chip
        actual = np.load(annotation, allow_pickle=False)
        assert np.isin(actual, [0, 1, 2, 255]).all(), chip
        keep = (actual == 0) | (actual == 2)
        scoring_masks.append(keep)
        labels.append(actual[keep] == 2)
        with np.load(prepared, allow_pickle=False) as data:
            assert data["raw"].shape == (2, *actual.shape), chip
            assert data["terr"].shape == (3, *actual.shape), chip
            assert data["rv"].shape == data["hv"].shape == actual.shape, chip
            for key, field in zip(event, ("raw", "terr", "rv", "hv")):
                event[key].append(data[field])
    event["physics_terrain"] = event["terr"]
    event["lab"] = [np.full(x.shape[1:], 255, np.uint8) for x in event["raw"]]
    event["clean"] = [False] * len(chips)
    event["hand_ok"] = sum(bool(mask.any()) for mask in event["terrain_valid"])
    return event, scoring_masks, np.concatenate(labels)


def score_arm(run_one, method, mask_mode, model, event, masks, labels, seed, ref, steps, checkpoint, score_path):
    collected = []

    def observer(idx, score):
        assert idx == len(collected) and score.dtype == np.float32
        assert np.isfinite(score).all() and 0 <= score.min() <= score.max() <= 1
        collected.append(score[masks[idx]])

    args = SimpleNamespace(seed=seed, steps=steps, model_factory=model, score_observer=observer,
                           darkness_only=(mask_mode == "darkness_only"))
    start = time.perf_counter()
    run_one(method, "V0", score_path.stem, event, np.asarray(ref["mean"], dtype=np.float32),
            np.asarray(ref["std"], dtype=np.float32), args, "cuda", str(checkpoint))
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    values = np.concatenate(collected)
    assert len(collected) == len(masks) and len(values) == len(labels)
    assert labels.any() and (~labels).any()
    pred = values >= ref["threshold"]
    tp, fp, fn, tn = (int(np.count_nonzero(x)) for x in (
        pred & labels, pred & ~labels, ~pred & labels, ~pred & ~labels))
    np.savez_compressed(score_path, scores=values)
    return dict(ap=float(average_precision_score(labels, values)),
                iou=tp/(tp+fp+fn), miss=fn/(tp+fn), fpr=fp/(fp+tn),
                fdp=fp/(tp+fp) if tp+fp else None, threshold=ref["threshold"],
                tp=tp, fp=fp, fn=fn, tn=tn, seconds=seconds,
                score_file=score_path.name, score_sha256=sha(score_path),
                score_values_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                diagnostics=args.adaptation_stats)


def run(seed, smoke, dataset_sha256, protocol_sha256, input_metadata_sha256, *, relaxed=False):
    if not Path("/kaggle/input").is_dir() or not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a Kaggle GPU.")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(2)
    mount = Path("/kaggle/input")
    candidates = [p for p in mount.rglob("px1_impactmesh_followup.zip") if p.is_file()]
    working = Path("/kaggle/working")
    if candidates:
        assert len(candidates) == 1 and sha(candidates[0]) == dataset_sha256
        root = working / "inputs"
        root.mkdir(exist_ok=False)
        with ZipFile(candidates[0]) as archive:
            archive.extractall(root)
    else:
        # Kaggle also mounts datasets with uploaded ZIPs expanded server-side.
        matches = [p for p in mount.rglob("protocol.json") if sha(p) == protocol_sha256]
        assert len(matches) == 1, "Missing or ambiguous verified input protocol"
        root = matches[0].parent
    assert sha(root / "protocol.json") == protocol_sha256
    for name, expected in input_metadata_sha256.items():
        assert sha(root / name) == expected, name
    spec = read(root / "protocol.json")
    for name, digest in spec["code_sha256"].items():
        assert sha(root / name) == digest, name
    sys.path.insert(0, str(root / "revision/fresh_sar_test"))
    from experiment import FloodResNet, run_one
    from sen1floods11_unet_baseline import UNet
    from tta_px1 import darkness_only_water_mask, adaptive_water_mask

    # Verify that only terrain acceptance changes. Radar-invalid pixels stay excluded.
    raw = torch.full((1, 2, 1, 3), -30.0)
    terrain = torch.zeros((1, 5, 1, 3))
    terrain[:, 3] = torch.tensor([1, 1, 0])
    terrain[:, 4] = torch.tensor([1, 0, 1])
    terrain[:, 2] = torch.tensor([0.0, 0.2, 0.0])
    assert adaptive_water_mask(raw, terrain, -15, -22).flatten().tolist() == [True, False, False]
    assert darkness_only_water_mask(raw, terrain, -15, -22).flatten().tolist() == [True, True, False]
    raw[0, 1, 0, 1] = -12
    assert not darkness_only_water_mask(raw, terrain, -15, -22)[0, 0, 1]

    inputs = read(root / "input_verification.json")
    roster = read(root / "selected.json")
    assert len(roster) == 13 and sum(map(len, roster.values())) == 516
    assert set(inputs) == {c for chips in roster.values() for c in chips}
    steps = 2 if smoke else spec["steps"]
    models = {"resnet50": FloodResNet, "unet": UNet}
    arms = {"ordinary": "adabn_st_valid", "darkness_only": "adabn_physa_valid",
            "joint": "adabn_physa_valid"}
    if relaxed:
        import tta_px1

        assert run_one is tta_px1.run_one and (tta_px1.P_HI, tta_px1.P_LO) == (.9, .1)
        assert spec["steps"] == 200 and spec["models"]["resnet50"]["threshold"] == .430
        # Every arm in this job is plain self-training. No physical rule uses the override.
        tta_px1.P_HI = .7
        models = {"resnet50": FloodResNet}
        arms = {"relaxed": "adabn_st_valid"}
    result = working / ("impactmesh_relaxed" if relaxed else "impactmesh_followup")
    result.mkdir(exist_ok=False)
    metadata = dict(seed=seed, smoke=smoke, steps=steps, archive_sha256=dataset_sha256,
                    protocol_sha256=sha(root / "protocol.json"), python=platform.python_version(),
                    torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
    if relaxed:
        metadata.update(
            control="relaxed", positive_cutoff=tta_px1.P_HI, negative_cutoff=tta_px1.P_LO,
            checkpoint_sha256=spec["models"]["resnet50"]["checkpoints"][str(seed)]["sha256"],
        )
    write(result / "run.json", metadata)
    print("IMPACTMESH_START", json.dumps(metadata), flush=True)
    rows = {}
    run_roster = {"EMSR843": roster["EMSR843"]} if smoke else roster
    for activation, chips in run_roster.items():
        event, masks, labels = load_event(root, chips, inputs)
        metrics = {}
        for backbone, model in models.items():
            ref = spec["models"][backbone]
            checkpoint = root / ref["checkpoints"][str(seed)]["path"]
            assert sha(checkpoint) == ref["checkpoints"][str(seed)]["sha256"]
            metrics[backbone] = {}
            for arm, method in arms.items():
                path = result / f"{activation}_{backbone}_{arm}.npz"
                row = score_arm(run_one, method, arm, model, event, masks, labels, seed, ref,
                                steps, checkpoint, path)
                metrics[backbone][arm] = row
                write(path.with_suffix(".json"), row)
                print("IMPACTMESH_ARM_DONE", activation, backbone, arm,
                      f"seconds={row['seconds']:.1f}", flush=True)
        rows[activation] = {"chips": chips, "metrics": metrics}
    assert all(math.isfinite(arm["ap"]) for row in rows.values()
               for model in row["metrics"].values() for arm in model.values())
    write(result / "results.json", dict(run=metadata, events=rows))
    print("IMPACTMESH_DONE", len(rows), seed, flush=True)
