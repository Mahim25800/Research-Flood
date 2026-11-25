"""Recompute matched TG-TTA results from Kaggle scores and frozen labels."""

import argparse
import json
from pathlib import Path

import numpy as np

from run_impactmesh_tg_tta import ARMS, metrics, sha
from stage_impactmesh_kaggle import RUN
from verify_impactmesh_kaggle import paired


METRICS = ("ap", "miss", "fpr")
CONTRASTS = (("physics_vv", "tg_tta"), ("physics_vv_vh", "tg_tta"),
             ("tg_tta", "ordinary"), ("physics_vv", "ordinary"), ("physics_vv_vh", "ordinary"))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def labels(chips, hashes):
    vectors = []
    for chip in chips:
        path = RUN / "labels" / f"{chip}.npy"
        require(sha(path) == hashes[chip], f"Frozen label mismatch: {chip}")
        a = np.load(path, allow_pickle=False)
        vectors.append(a[(a == 0) | (a == 2)] == 2)
    return np.concatenate(vectors)


def verify(folder, staged):
    protocol_sha = sha(staged / "protocol.json")
    spec = read(staged / "inputs.json")
    run = read(folder / "run.json")
    require(read(folder / "completed.json") == {"pairs": 39, "protocol_sha256": protocol_sha}
            and run["protocol_sha256"] == protocol_sha and run["seeds"] == [0, 1, 2], "Run seal mismatch")
    events = {}
    for event, chips in spec["events"].items():
        truth = labels(chips, spec["label"])
        rows = []
        for seed in run["seeds"]:
            root = folder / f"seed{seed}" / event
            saved = read(root / "results.json")
            require(saved["seed"] == seed and saved["activation"] == event
                    and set(saved["arms"]) == set(ARMS), f"Event coverage mismatch: {seed}/{event}")
            scores = {}
            for arm in ARMS:
                row = saved["arms"][arm]
                path = root / f"{arm}.npz"
                name = f"{seed}/{event}/{arm}"
                require(sha(path) == row["score_sha256"], f"Score hash mismatch: {name}")
                with np.load(path, allow_pickle=False) as data:
                    values = data["scores"]
                require(values.shape == truth.shape and values.dtype == np.float32
                        and np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1,
                        f"Invalid score array: {name}")
                score = metrics(values, truth)
                for key, value in score.items():
                    require(abs(value - row[key]) < 1e-8, f"Metric mismatch: {name}/{key}")
                scores[arm] = score
            details = saved["diagnostics"]["arms"]
            require(details["tg_tta"]["updates"] == 10 and details["ordinary"]["added_wet"] == 0,
                    f"Failed matched adaptation diagnostics: {seed}/{event}")
            rows.append(scores)
        events[event] = {arm: {key: float(np.mean([row[arm][key] for row in rows])) for key in METRICS}
                         for arm in ARMS}
        print(f"Verified {event} (15 score arrays)", flush=True)
    overall = {arm: {key: float(np.mean([row[arm][key] for row in events.values()])) for key in METRICS}
               for arm in ARMS}
    contrasts = {f"{a} - {b}": {key: paired([row[a][key] - row[b][key] for row in events.values()])
                                for key in METRICS} for a, b in CONTRASTS}
    return {"status": "passed", "score_arrays": len(events) * len(ARMS) * len(run["seeds"]),
            "run": run, "overall": overall, "contrasts": contrasts, "events": events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staged", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = verify(args.output / "impactmesh_tg_tta", args.staged)
    with (args.output / "verification.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
