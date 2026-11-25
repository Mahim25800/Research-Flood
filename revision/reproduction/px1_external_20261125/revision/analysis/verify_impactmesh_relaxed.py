"""Verify relaxed-control Kaggle scores and pair them with archived ImpactMesh scores."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_curve

from stage_impactmesh_kaggle import HERE, OUT as PRIOR, REV, RUN, digest, read, sha
from stage_impactmesh_relaxed import FPR, OUT
from verify_impactmesh_kaggle import paired, reference


METRICS = ("ap", "miss", "fpr", "iou", "recall_at_fpr_02", "recall_at_fpr_05")
REFERENCES = {"ordinary": "adabn_st_valid", "physics": "adabn_physa_valid"}


def score_verified(path, saved, labels):
    assert sha(path) == saved["score_sha256"]
    with np.load(path, allow_pickle=False) as archive:
        score = archive["scores"]
    assert score.dtype == np.float32 and score.shape == labels.shape
    assert digest(score.tobytes()) == saved["score_values_sha256"]
    assert np.isfinite(score).all() and 0 <= score.min() <= score.max() <= 1
    wet = score >= saved["threshold"]
    tp, fp, fn, tn = (int(np.count_nonzero(mask)) for mask in
                      (wet & labels, wet & ~labels, ~wet & labels, ~wet & ~labels))
    assert [saved[k] for k in ("tp", "fp", "fn", "tn")] == [tp, fp, fn, tn]
    ap = average_precision_score(labels, score)
    metrics = (ap, fn / (tp + fn), fp / (fp + tn), tp / (tp + fp + fn))
    assert np.allclose(metrics, [saved[k] for k in METRICS[:4]], rtol=0, atol=1e-8)
    fpr, recall, _ = roc_curve(labels, score)
    return {k: float(v) for k, v in zip(METRICS, (*metrics, *np.interp(FPR, fpr, recall)))}


def verify(folder, seed, smoke, protocol, roster, hashes, counts, pins):
    (path,) = folder.rglob("results.json")
    data = read(path)
    run = data["run"]
    assert run["seed"] == seed and run["smoke"] is smoke
    assert run["control"] == "relaxed" and (run["positive_cutoff"], run["negative_cutoff"]) == (.7, .1)
    assert run["steps"] == (2 if smoke else protocol["steps"])
    assert run["archive_sha256"] == protocol["archive_sha256"]
    assert run["protocol_sha256"] == protocol["protocol_sha256"]
    assert run["checkpoint_sha256"] == pins[str(seed)]["sha256"] and "T4" in run["gpu"]
    expected = {"EMSR843": roster["EMSR843"]} if smoke else roster
    assert set(data["events"]) == set(expected)
    rows = {}
    for event, chips in expected.items():
        row = data["events"][event]
        assert row["chips"] == chips and set(row["metrics"]) == {"resnet50"}
        assert set(row["metrics"]["resnet50"]) == {"relaxed"}
        saved = row["metrics"]["resnet50"]["relaxed"]
        assert saved["threshold"] == protocol["source_threshold"]
        labels = reference(chips, hashes)
        old = counts[event]["adabn_st_valid"][seed]
        assert int(labels.sum()) == old["tp"] + old["fn"]
        assert int((~labels).sum()) == old["fp"] + old["tn"]
        name = f"{event}_resnet50_relaxed.npz"
        assert saved["score_file"] == name
        rows[event] = score_verified(path.parent / name, saved, labels)
        stats = saved["diagnostics"]
        assert stats["physical_additions"] == stats["physical_conflicts"] == 0
        assert stats["pseudo_water"] + stats["pseudo_dry"] > 0
    print(f"Verified seed {seed}: {len(rows)} flood score arrays and their metrics")
    return run, rows


def summarise(verified, archived, roster):
    _, seed_rows = zip(*verified)
    events = {}
    for event in roster:
        arms = {"relaxed": {key: float(np.mean([rows[event][key] for rows in seed_rows]))
                            for key in METRICS}}
        arms.update(archived[event])
        events[event] = arms
    means = {arm: {metric: float(np.mean([v[arm][metric] for v in events.values()]))
                   for metric in METRICS} for arm in ("relaxed", *REFERENCES)}
    contrasts = {f"relaxed_minus_{arm}": {
        metric: paired([v["relaxed"][metric] - v[arm][metric] for v in events.values()])
        for metric in METRICS} for arm in REFERENCES}
    return dict(means=means, contrasts=contrasts, events=events)


def kaggle_references(protocol, roster, hashes):
    archived = []
    for seed in protocol["seeds"]:
        (path,) = (PRIOR / "downloads" / f"seed{seed}").rglob("results.json")
        results = read(path)
        run = results["run"]
        assert run["seed"] == seed and not run["smoke"] and run["steps"] == protocol["steps"]
        assert run["archive_sha256"] == protocol["archive_sha256"]
        assert run["protocol_sha256"] == protocol["protocol_sha256"]
        assert set(results["events"]) == set(roster)
        archived.append((path, results))
    output = {}
    for event, chips in roster.items():
        labels = reference(chips, hashes)
        methods = {}
        for name, arm in (("ordinary", "ordinary"), ("physics", "joint")):
            per_seed = []
            for path, data in archived:
                row = data["events"][event]
                assert row["chips"] == chips
                saved = row["metrics"]["resnet50"][arm]
                score_path = path.parent / f"{event}_resnet50_{arm}.npz"
                assert saved["score_file"] == score_path.name and saved["threshold"] == .430
                per_seed.append(score_verified(score_path, saved, labels))
            methods[name] = {key: float(np.mean([values[key] for values in per_seed])) for key in METRICS}
        output[event] = methods
        print(f"Verified archived Kaggle reference scores for {event}", flush=True)
    return dict(run_metadata=[data["run"] for _, data in archived], events=output)


def main(smoke, references_only):
    protocol = read(OUT / "protocol.json")
    roster = read(HERE / "selected.json")
    hashes = read(HERE / "input_verification.json")
    for name, expected in protocol["input_metadata_sha256"].items():
        assert sha(HERE / name) == expected
    counts = read(RUN / "results.json")["counts"]
    pins = read(PRIOR / "protocol.json")["models"]["resnet50"]["checkpoints"]
    path = REV / "analysis/impactmesh_saved_scores.json"
    assert sha(path) == protocol["reference_report_sha256"]
    archived = read(path)
    assert tuple(archived["matched_fpr"]) == FPR and archived["threshold"] == protocol["source_threshold"]
    assert set(roster) == set(archived["events"]) == set(counts)
    downloads = OUT / "downloads"
    kaggle_path = OUT / "verified_kaggle_references.json"
    if references_only:
        result = kaggle_references(protocol, roster, hashes)
        with kaggle_path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        return
    if smoke:
        run, rows = verify(downloads / "smoke", 0, True, protocol, roster, hashes, counts, pins)
        result = dict(run=run, events=rows)
        output = OUT / "verified_smoke.json"
    else:
        verified = [verify(downloads / f"seed{s}", s, False, protocol, roster, hashes, counts, pins)
                    for s in (0, 1, 2)]
        original = {
            event: {name: dict(zip(METRICS, [row[key][metric] for metric in METRICS[:4]] +
                                   row[key]["matched_recall"])) for name, key in REFERENCES.items()}
            for event, row in archived["events"].items()
        }
        kaggle = read(kaggle_path) if kaggle_path.exists() else kaggle_references(protocol, roster, hashes)
        assert set(kaggle["events"]) == set(roster)
        result = dict(activations=len(roster), seed_count=len(verified), threshold=.430, matched_fpr=FPR,
                      run_metadata=[run for run, _ in verified],
                      kaggle_reference_metadata=kaggle["run_metadata"],
                      kaggle=summarise(verified, kaggle["events"], roster),
                      original=summarise(verified, original, roster))
        output = OUT / "verified_results.json"
        print(json.dumps(result["kaggle"]["means"], indent=2))
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--references-only", action="store_true")
    args = parser.parse_args()
    main(args.smoke, args.references_only)
