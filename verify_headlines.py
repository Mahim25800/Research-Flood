"""Recover headline estimates from the bundled activation-level records."""

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent / "revision/generated"


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def report(label, differences):
    wins = sum(value > 0 for value in differences)
    print(f"{label}: {100 * mean(differences):+.2f} points, {wins}/{len(differences)} increases")


impact = read("impactmesh_kaggle_followup/verified_results.json")["events"]
for model in ("unet", "resnet50"):
    report(f"ImpactMesh {model}, joint vs ordinary",
           [row[model]["joint"]["ap"] - row[model]["ordinary"]["ap"] for row in impact.values()])
geo = read("geoid_v4_results_20261118/geoid_evaluation_v4/verification.json")
for field in ("primary_vs_same_rule_postprocessing", "secondary_vs_ordinary_self_training"):
    report(f"GEOID-Flood U-Net {field}", list(geo[field]["paired_ap"].values()))
geo_path = "geoid_resnet50_followup_20261125_candidate/"
geo_full = ROOT / (geo_path + "verified_all_arms.json")
result = read(geo_path + ("verified_all_arms.json" if geo_full.exists() else "verified_comparison.json"))
for arm in ("ordinary", "post", "darkness"):
    if arm in result["means"]:
        report(f"GEOID-Flood ResNet-50 joint vs {arm}",
               [row["joint"]["ap"] - row[arm]["ap"] for row in result["events"].values()])
relaxed = read("impactmesh_relaxed_control_20261124/verified_results.json")["kaggle"]["events"]
report("ImpactMesh ResNet-50 joint vs relaxed ordinary",
       [row["physics"]["ap"] - row["relaxed"]["ap"] for row in relaxed.values()])
matched = read("impactmesh_prior_methods/impactmesh_tg_tta/matched_false_alarms_v1.json")["event_means"]
for arm in ("tg_tta", "physics_vv", "physics_vv_vh"):
    print(f"Matched STANet {arm} mean miss rate: {100 * mean(row[arm] for row in matched.values()):.2f}%")
