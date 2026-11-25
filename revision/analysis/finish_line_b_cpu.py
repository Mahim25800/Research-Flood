"""Post-hoc controls on the archived ImpactMesh inputs and Kuro Siwo results."""

from collections import defaultdict
import csv
import json
import sys

import numpy as np
from sklearn.metrics import average_precision_score

from impactmesh_review_controls import DATA, FROZEN, REV, read, sha
from verify_impactmesh_kaggle import paired


OUT = REV / "generated/finish_line_b_20261124"
ARMS = ("adabn_st_valid", "adabn_physa_valid")


def subset(rows):
    delta = [row["delta_ap"] for row in rows.values() if row["delta_ap"] is not None]
    return {"eligible_events": list(rows), "scored_events": len(delta), "paired_ap": paired(delta) if delta else None}


def counted(pred, truth):
    tp, fp, fn = (int(x.sum()) for x in (pred & truth, pred & ~truth, ~pred & truth))
    return {"iou": tp / (tp + fp + fn), "miss": fn / int(truth.sum()), "fpr": fp / int((~truth).sum())}


def impactmesh():
    from zipfile import ZipFile

    snapshot = FROZEN / "protocol_snapshot.zip"
    assert sha(snapshot) == (FROZEN / "snapshot.sha256").read_text().split()[0]
    code = OUT / "impactmesh_frozen_code"
    with ZipFile(snapshot) as archive:
        for name in ("src/tta_px1.py", "src/conformal_px1.py"):
            archive.extract(name, code)
    sys.path.insert(0, str(code / "src"))
    from tta_px1 import otsu_dark, valid_adaptive_dark

    roster = read(DATA / "selected.json")
    pins = read(DATA / "input_verification.json")
    cohort = read(DATA / "cohort.json")
    metrics_by_event, lag_rows = {}, {}
    for event, chips in roster.items():
        raw, valid, truth, keep, lag = [], [], [], [], []
        for chip in chips:
            x, y = DATA / "prepared" / f"{chip}.npz", FROZEN / "labels" / f"{chip}.npy"
            pin = pins[chip]
            assert sha(x) == pin["input_sha256"] and sha(y) == pin["label_sha256"]
            with np.load(x, allow_pickle=False) as arrays:
                raw.append(arrays["raw"])
                valid.append(arrays["rv"])
            ann = np.load(y, allow_pickle=False)
            keep.append((ann == 0) | (ann == 2))
            truth.append(ann[keep[-1]] == 2)
            lag.append(float((np.datetime64(pin["timestamp_coordinate"], "ns") - np.datetime64(cohort[event]["date"])) / np.timedelta64(1, "D")))
        vv, vh = valid_adaptive_dark(raw, valid)
        sampled_vv = np.concatenate([r[0].ravel()[::7][v.ravel()[::7]] for r, v in zip(raw, valid)])
        otsu, _ = otsu_dark(sampled_vv)
        a = np.concatenate([((r[0] < vv) & (r[1] < vh) & v)[m] for r, v, m in zip(raw, valid, keep)])
        b = np.concatenate([((r[0] < otsu) & v)[m] for r, v, m in zip(raw, valid, keep)])
        y = np.concatenate(truth)
        metrics_by_event[event] = {"adaptive_vv_vh": counted(a, y), "otsu_vv": counted(b, y),
                                   "thresholds_db": [vv, vh, otsu]}

        picked = [i for i, days in enumerate(lag) if abs(days) <= 3]
        target = np.concatenate([truth[i] for i in picked]) if picked else np.array([], dtype=bool)
        delta = None
        if target.any() and (~target).any():
            scores = []
            for seed in range(3):
                folder = FROZEN / event / str(seed)
                hashes = {name.replace("\\", "/"): value for name, value in read(folder / "scores.json").items()}
                ap = []
                for arm in ARMS:
                    selected = []
                    for i in picked:
                        path = folder / arm / f"{chips[i]}.npy"
                        assert sha(path) == hashes[path.relative_to(folder).as_posix()]
                        selected.append(np.load(path, allow_pickle=False)[keep[i]])
                    ap.append(average_precision_score(target, np.concatenate(selected)))
                scores.append(ap[1] - ap[0])
            delta = float(np.mean(scores))
        lag_rows[event] = {"median_lag_days": float(np.median(lag)), "selected_chips": len(picked),
                           "total_chips": len(chips), "delta_ap": delta}
        print(event, "radar controls and chip-lag verified", flush=True)

    archived = read(REV / "analysis/impactmesh_saved_scores.json")["events"]
    event_lag = {event: {"delta_ap": archived[event][ARMS[1]]["ap"] - archived[event][ARMS[0]]["ap"]}
                 for event, row in lag_rows.items() if abs(row["median_lag_days"]) <= 3}
    report = {"model_free": {"per_event": metrics_by_event, "means": {
        arm: {key: float(np.mean([row[arm][key] for row in metrics_by_event.values()]))
              for key in ("iou", "miss", "fpr")} for arm in ("adaptive_vv_vh", "otsu_vv")}},
        "lag": {"per_event": lag_rows, "chip_filter": subset({e: r for e, r in lag_rows.items() if r["selected_chips"]}),
                "chip_filter_excluding_EMSR848": subset({e: r for e, r in lag_rows.items() if r["selected_chips"] and e != "EMSR848"}),
                "activation_median_filter": subset(event_lag),
                "activation_median_filter_excluding_EMSR848": subset({e: r for e, r in event_lag.items() if e != "EMSR848"})}}
    return report


def normalization():
    root = REV / "generated/bn_full_verified_20261117"
    csv_path, json_path = root / "per_event_seed.csv", root / "verification_and_comparison.json"
    grouped = defaultdict(dict)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (row["support"], row["weights"], row["normalization"], row["batch"]) == ("all_water", "tent", "source", "1"):
                assert row["seed"] not in grouped[row["event"]]
                grouped[row["event"]][row["seed"]] = float(row["average_precision"])
    assert len(grouped) == 26 and all(set(seeds) == {"0", "1", "2"} for seeds in grouped.values())
    value = float(np.mean([np.mean(list(seeds.values())) for seeds in grouped.values()]))
    reference, = [r for r in read(json_path)["summary"] if (r["support"], r["weights"], r["normalization"], r["batch"]) == ("all_water", "tent", "source", 1)]
    assert abs(value - reference["average_precision"]) < 1e-12
    return {"ap": value, "events": 26, "seeds": [0, 1, 2],
            "csv_sha256": sha(csv_path), "verification_sha256": sha(json_path)}


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    result = {"normalization": normalization(), "impactmesh": impactmesh()}
    with (OUT / "cpu_results.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
