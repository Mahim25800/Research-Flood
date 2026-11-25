# PX1 results

The [current manuscript](../paper/main.pdf) uses activation-level means. The analysis first averages the three source checkpoints within an activation, then weights activations equally. External metrics score floodwater against dry pixels and exclude permanent water and nodata. AP uses unrounded scores.

## Verified external results

| Comparison | Key result | Archive |
| --- | --- | --- |
| ImpactMesh two-backbone crossover | Joint physics gained 12.17 AP points on U-Net and 14.15 on ResNet-50 over ordinary self-training. Adding terrain to darkness changed AP by −0.12 and +0.22 points, respectively. | [Activation-level results](../revision/reproduction/px1_external_20261125/revision/generated/impactmesh_kaggle_followup/verified_results.json) |
| ImpactMesh relaxed teacher | Lowering the positive cutoff to 0.70 reduced misses by 17.98 points and increased FPR by 3.40 points. Joint physics exceeded the relaxed control by 11.47 AP points, with wins on all 13 activations. | [Matched T4 analysis](../revision/reproduction/px1_external_20261125/revision/generated/impactmesh_relaxed_control_20261124/verified_results.json) |
| ImpactMesh paired-input STANet | At the TG-TTA port’s matched mean FPR of 1.65%, mean misses were 66.68% for TG-TTA, 68.28% for VV physics and 67.31% for VV+VH physics. | [Matched false alarms](../revision/reproduction/px1_external_20261125/revision/generated/impactmesh_prior_methods/impactmesh_tg_tta/matched_false_alarms_v1.json) |
| ImpactMesh acquisition timing | Filtering chips to ±3 days of activation and excluding EMSR848 gave +13.68 mean AP points, +7.63 median and 10 of 11 improvements. | [Timing sensitivity](../revision/reproduction/px1_external_20261125/revision/generated/finish_line_b_20261124/cpu_results.json) |
| GEOID-Flood preregistered U-Net | The primary gain over same-rule post-processing was +3.50 AP points (*p* = 0.0195). The secondary gain over ordinary self-training was +0.17 points. | [Original verification](../revision/reproduction/px1_external_20261125/revision/generated/geoid_v4_results_20261118/geoid_evaluation_v4/verification.json) |
| GEOID-Flood post-hoc ResNet-50 | Joint physics gained +1.19 AP points over ordinary and +1.63 over post-processing. Darkness alone gained +0.99 over ordinary. Terrain added +0.20 over darkness (95% interval −0.16 to +0.72). All 162 saved score arrays passed CPU verification. | [Six-arm verification](../revision/reproduction/px1_external_20261125/revision/generated/geoid_resnet50_followup_20261125_candidate/verified_all_arms.json) |

The archival reports include comparisons at fixed source thresholds and matched false alarms. Retrospective matching interpolates per-activation, per-checkpoint ROC curves using target labels. Its purpose is to compare methods at similar false-alarm rates.

## Historical benchmarks

The older [Sen1Floods11](sen1floods11/), [Kuro Siwo](kuro_siwo/) and [Kaggle analysis](final/final_analysis_kaggle.md) records support the historical experiments in the supplement. The [baseline fidelity audit](../revision/BASELINE_FIDELITY_AUDIT.md) describes the earlier implementations. The [supplement](../paper/supplement.pdf) explains the assumptions and scope of the flood-level calibration results.
