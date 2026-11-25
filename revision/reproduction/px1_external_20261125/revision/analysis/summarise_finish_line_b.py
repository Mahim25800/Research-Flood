"""Summarize regenerated ImpactMesh controls and exploratory label-free measures."""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from finish_line_b_cpu import ARMS, OUT, DATA, REV, read, sha


CONTROLS = ("none_valid", "adabn_valid", "adabn_physa_valid_post")


def main(impactmesh_only=False):
    roster = read(DATA / "selected.json")
    earlier = read(REV / "analysis/impactmesh_saved_scores.json")["events"]
    pins = read(DATA / "input_verification.json")
    model_free = read(OUT / "cpu_results.json")["impactmesh"]["model_free"]["per_event"]
    matched, checks, points = {}, [], []
    for event, chips in roster.items():
        rows = read(OUT / "inference_controls" / f"{event}.json")["seeds"]
        assert len(rows) == 3
        matched[event] = {arm: np.mean([row[arm]["matched_recall"] for row in rows], axis=0).tolist()
                          for arm in CONTROLS}
        matched[event].update({arm: earlier[event][arm]["matched_recall"] for arm in ARMS})
        checks.extend(row[arm]["archive_score_hash_match"] for row in rows for arm in CONTROLS)
        teacher = [row["initial_teacher"] for row in rows]
        teacher_water = float(np.mean([x["teacher_water"] / x["radar"] for x in teacher]))
        shares = [x["teacher_dry_on_eligible"] / x["eligible"] for x in teacher if x["eligible"]]
        dark = valid = 0
        vv, vh, _ = model_free[event]["thresholds_db"]
        for chip in chips:
            path = DATA / "prepared" / f"{chip}.npz"
            assert sha(path) == pins[chip]["input_sha256"]
            with np.load(path, allow_pickle=False) as a:
                radar, mask = a["raw"], a["rv"]
                dark += int((mask & (radar[0] < vv) & (radar[1] < vh)).sum())
                valid += int(mask.sum())
        points.append({"event": event, "cohort": "ImpactMesh", "radar_dark_share": dark / valid,
                       "teacher_water_share": teacher_water,
                       "teacher_dry_on_eligible_share": float(np.mean(shares)) if shares else None,
                       "delta_ap": earlier[event][ARMS[1]]["ap"] - earlier[event][ARMS[0]]["ap"]})

    geo = OUT / "geoid_cpu/model_free_results.json"
    if not impactmesh_only:
        baseline = read(REV / "generated/geoid_v4_results_20261118/geoid_evaluation_v4/results/results.json")["events"]
        for event, row in read(geo)["events"].items():
            m = baseline[event]["methods"]
            delta = [m[f"{seed}:adabn_physa_valid"]["flood"]["ap"] - m[f"{seed}:adabn_st_valid"]["flood"]["ap"]
                     for seed in (0, 1, 2)]
            points.append({"event": event, "cohort": "GEOID-Flood", "radar_dark_share": row["radar_dark_share"],
                           "delta_ap": float(np.mean(delta))})

    arms = (*CONTROLS, *ARMS)
    values = {arm: np.asarray([row[arm] for row in matched.values()]) for arm in arms}
    target = values[ARMS[1]]
    means = {arm: values[arm].mean(axis=0).tolist() for arm in arms}
    contrasts = {arm: {"mean_delta_recall": (target - values[arm]).mean(axis=0).tolist(),
                       "physics_wins": (target > values[arm]).sum(axis=0).tolist()}
                 for arm in arms[:-1]}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), layout="constrained")
    stats = {}
    colors = {"ImpactMesh": "#a3482c", "GEOID-Flood": "#305e87"}
    for ax, key, label in zip(axes, ("radar_dark_share", "teacher_dry_on_eligible_share"),
                              ("Adaptive dark radar (%)", "Teacher-dry among rule candidates (%)")):
        stats[key] = {}
        for cohort, color in colors.items():
            group = [r for r in points if r["cohort"] == cohort and r.get(key) is not None]
            if not group:
                continue
            x, y = [100 * r[key] for r in group], [100 * r["delta_ap"] for r in group]
            ax.scatter(x, y, color=color, label=cohort)
            stats[key][cohort] = {"n": len(group), "spearman_rho": float(spearmanr(x, y).statistic)}
        ax.set_xlabel(label)
        ax.axhline(0, color="#aaaaaa", lw=.7)
    axes[0].set_ylabel("Physics minus ordinary AP (points)")
    axes[0].legend(frameon=False, fontsize=8)
    suffix = "_impactmesh" if impactmesh_only else ""
    fig.savefig(OUT / f"exploratory_headroom{suffix}.pdf")
    plt.close(fig)
    result = {"fpr": [.02, .05], "mean_recall": means, "physics_contrasts": contrasts,
              "control_hash_matches": [int(sum(checks)), len(checks)], "event_recall": matched,
              "exploratory_diagnostic": {"points": points, "within_cohort": stats}}
    with (OUT / f"saved_score_followups{suffix}.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impactmesh-only", action="store_true")
    main(parser.parse_args().impactmesh_only)
