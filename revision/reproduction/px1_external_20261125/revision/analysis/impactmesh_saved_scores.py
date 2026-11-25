"""Reproduce the manuscript's ImpactMesh tables and figure from frozen CPU scores."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_curve

REVISION = Path(__file__).resolve().parents[1]
ROOT = REVISION / "fresh_sar_test"
RUN = ROOT / "frozen_comparison_v2"
ARMS = ("adabn_st_valid", "adabn_physa_valid")
FPR = np.array([0.02, 0.05])


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyse():
    roster = read_json(ROOT / "selected.json")
    cohort = read_json(ROOT / "cohort.json")
    inputs = read_json(ROOT / "input_verification.json")
    frozen = read_json(RUN / "results.json")
    threshold = frozen["threshold"]
    result = {}
    for event, chips in roster.items():
        labels = []
        for chip in chips:
            path = RUN / "labels" / (chip + ".npy")
            assert digest(path) == inputs[chip]["label_sha256"]
            labels.append(np.load(path))
        keep = [(a == 0) | (a == 2) for a in labels]
        y = np.concatenate([a[m] == 2 for a, m in zip(labels, keep)])
        lags = [(np.datetime64(inputs[c]["timestamp_coordinate"], "ns") -
                 np.datetime64(cohort[event]["date"])) / np.timedelta64(1, "D") for c in chips]
        event_result = {"chips": len(chips), "flood_fraction": float(y.mean()),
                        "lag_days_min_median_max": [float(f(lags)) for f in (min, np.median, max)]}
        for arm in ARMS:
            values = []
            for seed in range(3):
                folder = RUN / event / str(seed)
                hashes = {key.replace("\\", "/"): digest for key, digest in read_json(folder / "scores.json").items()}
                scores = []
                for chip, mask in zip(chips, keep):
                    name = Path(arm) / (chip + ".npy")
                    assert digest(folder / name) == hashes[name.as_posix()]
                    scores.append(np.load(folder / name)[mask])
                score = np.concatenate(scores)
                ap = average_precision_score(y, score)
                assert np.isclose(ap, frozen["primary"]["events"][event]["seeds"][seed][arm], atol=1e-12, rtol=0)
                pred = score >= threshold
                tp, fp, fn, tn = map(int, (np.count_nonzero(pred & y), np.count_nonzero(pred & ~y), np.count_nonzero(~pred & y), np.count_nonzero(~pred & ~y)))
                assert frozen["counts"][event][arm][seed] == dict(tp=tp, fp=fp, fn=fn, tn=tn)
                fpr, recall, _ = roc_curve(y, score)
                values.append(dict(ap=ap, iou=tp/(tp+fp+fn), miss=fn/(tp+fn), fpr=fp/(fp+tn),
                                   fdp=fp/(tp+fp) if tp+fp else None,
                                   matched_recall=np.interp(FPR, fpr, recall)))
            event_result[arm] = {
                key: np.mean([v[key] for v in values], axis=0).tolist()
                if all(v[key] is not None for v in values) else None for key in values[0]
            }
        result[event] = event_result
        print(event, "AP and counts verified", flush=True)
    ap_delta = np.array([row[ARMS[1]]["ap"]-row[ARMS[0]]["ap"] for row in result.values()])
    bootstrap = np.random.default_rng(20261120).choice(ap_delta, size=(10000, len(ap_delta)), replace=True)
    recall = np.array([[row[a]["matched_recall"] for a in ARMS] for row in result.values()])
    common_fdp = [row for row in result.values() if all(row[a]["fdp"] is not None for a in ARMS)]
    return dict(threshold=threshold, matched_fpr=FPR.tolist(), events=result,
                ap_bootstrap_ci=np.quantile(bootstrap.mean(axis=1), [.025, .975]).tolist(),
                matched_recall_mean=recall.mean(axis=0).tolist(),
                matched_recall_wins=(recall[:, 1] > recall[:, 0]).sum(axis=0).tolist(),
                common_fdp_n=len(common_fdp),
                common_fdp_mean=[float(np.mean([row[a]["fdp"] for row in common_fdp])) for a in ARMS])


def plot(events, path):
    order = sorted(events, key=lambda e: events[e][ARMS[1]]["ap"]-events[e][ARMS[0]]["ap"], reverse=True)
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(7.2, 4.15),
                             gridspec_kw={"width_ratios": [1, 1, .82]})
    fig.subplots_adjust(left=.123, right=.985, top=.82, bottom=.15, wspace=.25)
    for ax, metric, title, ticks in zip(axes, ("ap", "miss", "fpr"),
                                        ("(a) Floodwater AP", "(b) Floodwater missed", "(c) Dry-pixel FPR"),
                                        ([0,25,50,75,100], [0,25,50,75,100], [0,10,20,30])):
        a, b = ([100*events[e][arm][metric] for e in order] for arm in ARMS)
        y = np.arange(len(order))
        ax.hlines(y, a, b, color="#aeb5bc", lw=1.1)
        ax.scatter(a, y, s=23, facecolors="white", edgecolors="#394e62", label="Ordinary self-training", zorder=3)
        ax.scatter(b, y, s=23, color="#bf4b25", marker="D", label="Physics-guided", zorder=4)
        ax.set(title=title, xlabel="Percent", xticks=ticks, xlim=(-2,ticks[-1]+3))
        ax.set_yticks(y, order)
        ax.grid(axis="x", color="#e3e7ed", lw=.6)
        ax.tick_params(length=0, labelsize=9)
        ax.spines[["top","right","left"]].set_visible(False)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .97),
               ncol=2, frameon=False, fontsize=9)
    fig.savefig(path, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


if __name__ == "__main__":
    result = analyse()
    (REVISION / "analysis/impactmesh_saved_scores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot(result["events"], REVISION / "paper_wip/figures/fig_impactmesh_paired.pdf")
