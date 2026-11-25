"""Compare floodwater misses at the TG-TTA false-positive count for each event and seed."""

import argparse
import json
from pathlib import Path

import numpy as np

from verify_impactmesh_tg_tta import labels, paired, read, require, sha


PHYSICS = ("physics_vv", "physics_vv_vh")


def match(scores, truth, budget):
    dry = scores[~truth]
    water = scores[truth]
    require(0 <= budget <= dry.size and water.size > 0, "Invalid false-positive budget")
    if budget == dry.size:
        return {"miss": 0.0, "cutoff": 0.0, "tie_fraction": 1.0, "dry_at_cutoff": int((dry == 0).sum()),
                "water_at_cutoff": int((water == 0).sum()), "conservative_miss": 0.0, "conservative_fpr": 1.0}
    # Use the first dry pixel beyond the budget so all free recall on an ROC plateau is counted.
    k = dry.size - budget - 1
    cutoff = np.partition(dry, k)[k]
    fp_above = int((dry > cutoff).sum())
    fp_tied = int((dry == cutoff).sum())
    fraction = (budget - fp_above) / fp_tied
    require(0 <= fraction <= 1, "Could not match the false-positive budget")
    tp_above = int((water > cutoff).sum())
    tp_tied = int((water == cutoff).sum())
    expected_tp = tp_above + fraction * tp_tied
    conservative_tp = tp_above + (tp_tied if fraction == 1 else 0)
    return {"miss": float(1 - expected_tp / water.size), "cutoff": float(cutoff),
            "tie_fraction": float(fraction), "dry_at_cutoff": fp_tied, "water_at_cutoff": tp_tied,
            "conservative_miss": float(1 - conservative_tp / water.size),
            "conservative_fpr": (budget if fraction == 1 else fp_above) / dry.size}


def scores_at(folder, arm, report, truth):
    path = folder / f"{arm}.npz"
    require(sha(path) == report["score_sha256"], f"Saved score hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        values = bundle["scores"]
    require(values.shape == truth.shape and values.dtype == np.float32
            and np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1,
            f"Invalid saved scores: {path}")
    return values


def analyse(staged, download):
    checked = read(download / "verification.json")
    require(checked["status"] == "passed" and checked["score_arrays"] == 195, "Missing previous score verification")
    spec = read(staged / "inputs.json")
    require(checked["run"]["protocol_sha256"] == sha(staged / "protocol.json"), "Protocol hash mismatch")
    events = {}
    for event, chips in spec["events"].items():
        truth = labels(chips, spec["label"])
        water_pixels, dry_pixels = int(truth.sum()), int((~truth).sum())
        seed_rows = []
        for seed in (0, 1, 2):
            folder = download / "impactmesh_tg_tta" / f"seed{seed}" / event
            report = read(folder / "results.json")
            require(report["seed"] == seed and report["activation"] == event, "Score identity mismatch")
            arms = report["arms"]
            detected = scores_at(folder, "tg_tta", arms["tg_tta"], truth) >= 0.5
            fp = int((detected & ~truth).sum())
            fn = int((~detected & truth).sum())
            require(fp == arms["tg_tta"]["fp"] and fn == arms["tg_tta"]["fn"], "TG-TTA counts changed")
            seed_rows.append({"seed": seed, "fpr": fp / dry_pixels, "tg_tta_miss": fn / water_pixels,
                              "fp_budget": fp, "physics": {arm: match(scores_at(folder, arm, arms[arm], truth), truth, fp)
                                                          for arm in PHYSICS}})
        events[event] = seed_rows
        print(f"Matched {event} across three seeds", flush=True)

    event_means = {event: {"fpr": float(np.mean([row["fpr"] for row in rows])),
                           "tg_tta": float(np.mean([row["tg_tta_miss"] for row in rows])),
                           **{arm: float(np.mean([row["physics"][arm]["miss"] for row in rows])) for arm in PHYSICS},
                           **{arm + "_conservative": float(np.mean([row["physics"][arm]["conservative_miss"]
                                                                   for row in rows])) for arm in PHYSICS}}
                   for event, rows in events.items()}
    overall = {key: float(np.mean([row[key] for row in event_means.values()])) for key in next(iter(event_means.values()))}
    contrasts = {arm: paired([row[arm] - row["tg_tta"] for row in event_means.values()]) for arm in PHYSICS}
    return {"reference_threshold": 0.5, "pair_count": 3 * len(events),
            "method": "Match the TG-TTA false-positive count within seed and event; fractional boundary ties in expectation",
            "exploratory": True, "overall": overall, "contrasts_miss_physics_minus_tg": contrasts,
            "event_means": event_means, "event_seed_results": events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staged", type=Path)
    parser.add_argument("download", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyse(args.staged, args.download)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"overall": report["overall"], "contrasts": report["contrasts_miss_physics_minus_tg"]}, indent=2))


if __name__ == "__main__":
    main()
