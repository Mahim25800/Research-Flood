"""Verify saved Kaggle scores against frozen ImpactMesh labels and summarise paired results."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score

from stage_impactmesh_kaggle import HERE, RUN, OUT, read, sha, dump
from stage_impactmesh_kaggle import digest


MODELS = ("resnet50", "unet")
ARMS = ("ordinary", "darkness_only", "joint")
METRICS = ("ap", "iou", "miss", "fpr")


def reference(chips, hashes):
    arrays = []
    for chip in chips:
        path = RUN / "labels" / f"{chip}.npy"
        assert sha(path) == hashes[chip]["label_sha256"]
        label = np.load(path, allow_pickle=False)
        arrays.append(label[(label == 0) | (label == 2)] == 2)
    return np.concatenate(arrays)


def verify(folder, smoke):
    (file,) = folder.rglob("results.json")
    data = read(file)
    run = data["run"]
    original = read(RUN / "results.json")
    spec = read(OUT / "protocol.json")
    hashes = read(HERE / "input_verification.json")
    roster = read(HERE / "selected.json")
    view = {"EMSR843": roster["EMSR843"]} if smoke else roster
    assert run["archive_sha256"] == (OUT / "dataset.sha256").read_text().strip()
    assert run["protocol_sha256"] == digest(json.dumps(spec, indent=2).encode())
    assert run["smoke"] == smoke and run["steps"] == (2 if smoke else 200)
    assert set(data["events"]) == set(view)
    seed = run["seed"]
    for activation, row in data["events"].items():
        assert row["chips"] == view[activation]
        labels = reference(row["chips"], hashes)
        old = original["counts"][activation]["adabn_st_valid"][seed]
        assert int(labels.sum()) == old["tp"] + old["fn"]
        assert int((~labels).sum()) == old["fp"] + old["tn"]
        assert set(row["metrics"]) == set(MODELS)
        for model, arms in row["metrics"].items():
            assert set(arms) == set(ARMS)
            for arm, values in arms.items():
                assert values["threshold"] == spec["models"][model]["threshold"]
                assert values["score_file"] == f"{activation}_{model}_{arm}.npz"
                path = file.parent / values["score_file"]
                assert sha(path) == values["score_sha256"]
                with np.load(path, allow_pickle=False) as result:
                    scores = result["scores"]
                assert scores.dtype == np.float32 and scores.shape == labels.shape
                assert values["score_values_sha256"] == digest(scores.tobytes())
                assert np.isfinite(scores).all() and 0 <= scores.min() <= scores.max() <= 1
                predicted = scores >= values["threshold"]
                tp, fp, fn, tn = (int(np.count_nonzero(x)) for x in (
                    predicted & labels, predicted & ~labels, ~predicted & labels, ~predicted & ~labels))
                assert [values[k] for k in ("tp", "fp", "fn", "tn")] == [tp, fp, fn, tn]
                expected = [average_precision_score(labels, scores), tp/(tp+fp+fn), fn/(tp+fn), fp/(fp+tn)]
                assert np.allclose(expected, [values[k] for k in METRICS], rtol=0, atol=1e-8)
                if tp + fp:
                    assert np.isclose(values["fdp"], fp/(tp+fp))
                else:
                    assert values["fdp"] is None
                assert values["diagnostics"]["physical_additions"] == 0 or arm != "ordinary"
    print(f"Verified seed {seed}: {len(view)} activations, {len(view)*6} score arrays")
    return data


def paired(differences):
    diff = np.asarray(differences, dtype=np.float64)
    wins, losses = int(sum(diff > 0)), int(sum(diff < 0))
    bs = np.random.default_rng(20).choice(diff, size=(10000, len(diff)), replace=True).mean(axis=1)
    return dict(mean=float(diff.mean()), median=float(np.median(diff)), wins=wins, losses=losses,
                ties=len(diff)-wins-losses, ci95=np.quantile(bs, [.025, .975]).tolist(),
                two_sided_sign_p=float(binomtest(wins, wins+losses).pvalue) if wins+losses else None)


def summarise(seeds):
    assert {seed["run"]["seed"] for seed in seeds} == {0, 1, 2}
    assert len({(seed["run"]["gpu"], seed["run"]["torch"], seed["run"]["cuda"]) for seed in seeds}) == 1
    events = {}
    for name in seeds[0]["events"]:
        events[name] = {model: {arm: {metric: float(np.mean([
            d["events"][name]["metrics"][model][arm][metric] for d in seeds
        ])) for metric in METRICS} for arm in ARMS} for model in MODELS}
    means = {model: {arm: {metric: float(np.mean([
        e[model][arm][metric] for e in events.values()
    ])) for metric in METRICS} for arm in ARMS} for model in MODELS}
    contrasts = {}
    for model in MODELS:
        contrasts[model] = {}
        for a, b in (("darkness_only", "ordinary"), ("joint", "darkness_only"), ("joint", "ordinary")):
            contrasts[model][f"{a}_vs_{b}"] = {
                metric: paired([e[model][a][metric]-e[model][b][metric] for e in events.values()])
                for metric in METRICS}
    crossover = paired([(e["unet"]["joint"]["ap"]-e["unet"]["ordinary"]["ap"])
                        -(e["resnet50"]["joint"]["ap"]-e["resnet50"]["ordinary"]["ap"])
                        for e in events.values()])
    return dict(activations=len(events), seed_count=len(seeds), means=means, contrasts=contrasts,
                crossover_joint_minus_ordinary_ap=crossover, events=events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert len(args.directories) == (1 if args.smoke else 3)
    seeds = [verify(path, args.smoke) for path in args.directories]
    if not args.smoke:
        summary = summarise(seeds)
        dump(OUT / "verified_results.json", summary)
        print(json.dumps(summary["means"], indent=2))
