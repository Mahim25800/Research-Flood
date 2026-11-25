"""Verify completed GEOID-Flood ResNet-50 scores against the frozen v4 labels."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_curve

from run_geoid_resnet import ARMS
from stage_geoid_resnet import FROZEN, ORIGINAL, OUT, REV, digest, read, sha
from verify_impactmesh_kaggle import paired


COMPARISON = ("ordinary", "post", "joint")
METRICS = ("ap", "iou", "miss", "fpr", "recall_02", "recall_05")


def archived_truth(evaluator, root, spec, windows):
    tiles = {}
    observed = []
    for name, y, x in windows:
        tile = spec["tiles"][name]
        if name not in tiles:
            for field in ("s1_post", "label"):
                path = tile[field]
                assert sha(root / path) == spec["raster_hashes"][path], path
            _, valid, grid = evaluator.chips(root, tile)
            labels, _ = evaluator.raster(root, tile["label"], grid)
            label = labels[0].filled(255)
            assert np.isin(label, [0, 1, 2, 255]).all()
            tiles[name] = np.where(valid, label, 255)
        chip = tiles[name][y:y + 512, x:x + 512]
        observed.append(chip[(chip == 0) | (chip == 2)] == 2)
    return np.concatenate(observed)


def score_metrics(path, saved, truth, threshold, wait_for_scores):
    expected_hash = saved["score_sha256"]
    actual_hash = sha(path) if path.is_file() else None
    deadline = time.monotonic() + 600
    while wait_for_scores and actual_hash != expected_hash and time.monotonic() < deadline:
        time.sleep(5)
        actual_hash = sha(path) if path.is_file() else None
    assert actual_hash == expected_hash, path
    with np.load(path, allow_pickle=False) as npz:
        score = npz["scores"]
    assert score.dtype == np.float32 and score.shape == truth.shape
    assert np.isfinite(score).all() and 0 <= score.min() <= score.max() <= 1
    wet = score >= threshold
    tp, fp, fn, tn = (int(mask.sum()) for mask in
                      (wet & truth, wet & ~truth, ~wet & truth, ~wet & ~truth))
    assert [tp, fp, fn, tn] == [saved[k] for k in ("tp", "fp", "fn", "tn")]
    fpr, recall, _ = roc_curve(truth, score)
    actual = [average_precision_score(truth, score), tp / (tp + fp + fn),
              fn / (tp + fn), fp / (fp + tn), *np.interp([.02, .05], fpr, recall)]
    expected = [saved[k] for k in METRICS[:4]] + saved["matched_recall"]
    assert np.allclose(actual, expected, atol=1e-8, rtol=0), path
    return dict(zip(METRICS, map(float, actual)))


def verify(folder, labels_root, wait_for_scores, arms=COMPARISON):
    spec = read(folder / "protocol.json")
    freeze = read(folder / "freeze.json")
    assert digest(json.dumps(spec, sort_keys=True).encode()) == freeze["protocol_sha256"]
    assert sha(folder / "kernel/run.py") == freeze["runner_sha256"]
    assert sha(ORIGINAL / "results/results.json") == spec["geoid_results_sha256"]
    assert sha(FROZEN / "px1-geoid-protocol-20261117-v4.zip") == spec["geoid_protocol_sha256"]
    sys.path.insert(0, str(labels_root.parent / "protocol/revision"))
    import geoid_evaluate as evaluator

    (path,) = (folder / "download_v1").rglob("results.json")
    result = read(path)
    run = result["run"]
    assert run["spec_sha256"] == freeze["protocol_sha256"]
    assert ("T4" in run["gpu"] and run["steps"] == spec["steps"]
            and run["adaptation_seed"] == spec["adaptation_seed"]
            and run["threshold"] == spec["calibration"]["threshold"])
    assert set(result["events"]) == set(spec["selected"]) and len(result["events"]) == 9
    events = {}
    rows_count = checked = 0
    for activation, windows in spec["selected"].items():
        event = result["events"][activation]
        assert event["selected"] == windows
        assert event == read(path.parent / f"{activation}_metrics.json")
        expected_water, expected_dry = spec["support"][activation]
        truth = archived_truth(evaluator, labels_root, spec, windows)
        target = path.parent / f"{activation}_truth.npy"
        assert sha(target) == event["truth_sha256"]
        assert np.array_equal(truth, np.load(target, allow_pickle=False))
        assert truth.sum() == expected_water and (~truth).sum() == expected_dry
        assert set(event["rows"]) == {f"{seed}:{arm}" for seed in (0, 1, 2) for arm in ARMS}
        for row in event["rows"].values():
            assert row["tp"] + row["fn"] == expected_water
            assert row["fp"] + row["tn"] == expected_dry
            assert all(np.isfinite(row[key]) and 0 <= row[key] <= 1 for key in METRICS[:4])
            rows_count += 1
        values = {}
        for arm in arms:
            seeds = []
            for seed in (0, 1, 2):
                row = event["rows"][f"{seed}:{arm}"]
                name = f"{activation}_{seed}_{arm}.npz"
                assert row["score_file"] == name
                seeds.append(score_metrics(path.parent / name, row, truth, run["threshold"], wait_for_scores))
                checked += 1
            values[arm] = {key: float(np.mean([r[key] for r in seeds])) for key in METRICS}
        events[activation] = values
        print(f"Verified {activation}: three checkpoints, {len(arms)} arms", flush=True)
    means = {arm: {key: float(np.mean([r[arm][key] for r in events.values()]))
                   for key in METRICS} for arm in arms}
    contrasts = {f"joint_minus_{arm}": {
        key: paired([r["joint"][key] - r[arm][key] for r in events.values()])
        for key in METRICS} for arm in arms if arm != "joint"}
    return dict(run=run, metadata_rows=rows_count, scores_verified=checked,
                means=means, contrasts=contrasts, events=events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=OUT.with_name(OUT.name + "_candidate"))
    parser.add_argument("--labels-root", type=Path,
                        default=REV / "generated/finish_line_b_20261124/geoid_cpu/rasters")
    parser.add_argument("--wait-for-scores", action="store_true", help="Verify alongside in-progress downloads")
    parser.add_argument("--all-arms", action="store_true", help="Verify all six arms and write a separate report")
    args = parser.parse_args()
    arms = tuple(ARMS) if args.all_arms else COMPARISON
    data = verify(args.folder, args.labels_root, args.wait_for_scores, arms)
    filename = "verified_all_arms.json" if args.all_arms else "verified_comparison.json"
    with (args.folder / filename).open("x", encoding="utf-8") as output:
        json.dump(data, output, indent=2, allow_nan=False)
    print(json.dumps(data["means"], indent=2))
