# PX1 external evaluation reproduction materials

This release contains nine source checkpoints, the frozen evaluation metadata, prediction manifests, verified summaries and analysis scripts for the ImpactMesh and GEOID-Flood follow-ups. The original ImpactMesh metadata archive is preserved in `archives/`. Check every published file against `SHA256SUMS`.

## Reproduce the headline tables

Run `python verify_headlines.py` from this directory. It recomputes the four-cell AP gains, the GEOID-Flood primary contrast and the relaxed teacher contrast from the activation records. Exact per-pixel recomputation additionally requires the archived data and scores. The published archives contain their file manifests and checksums. The raw prediction arrays and prepared raster copies are outside this release.

For score-level checks, use the run layouts in the archived protocols. Put the original prepared inputs and predictions at their documented paths in the project checkout and overlay the `revision/` files in this release. The scripts `revision/fresh_sar_test/verify_comparison.py`, `revision/analysis/verify_impactmesh_kaggle.py`, `revision/analysis/verify_impactmesh_relaxed.py`, `revision/analysis/verify_impactmesh_tg_tta.py` and `revision/analysis/verify_geoid_resnet.py --all-arms` check the original score arrays against their labels. The last script expects the selected GEOID-Flood radar and label rasters and the frozen v4 evaluator. The frozen protocol and public shard references identify the inputs. Access to archived private Kaggle outputs is arranged through the corresponding author.

## Checkpoint provenance

- `checkpoints/resnet50/` holds the three source segmentation checkpoints fine-tuned on Sen1Floods11. The encoder was initialized from the Sentinel-1 SSL4EO-S12 ResNet-50 weights by Wang et al., arXiv:2211.07044 (CC BY 4.0). The released checkpoints contain the fine-tuned encoder and the trained segmentation head. [Upstream weights and attribution](https://github.com/zhu-xlab/SSL4EO-S12)
- `checkpoints/unet/` holds the three U-Net source checkpoints trained on Sen1Floods11. [Source dataset](https://github.com/cloudtostreet/Sen1Floods11)
- `checkpoints/stanet/` holds the three paired-input checkpoints trained on the modified Sen1Floods11 source. Source calibration details are in `revision/generated/impactmesh_prior_methods/stanet_source_gpu_full/verified_v1/verification.json`.

The original GEOID-Flood U-Net study is documented in the public preregistration. The GEOID-Flood ResNet-50 and relaxed ImpactMesh runs are post-hoc follow-ups. [Preregistration](https://osf.io/wv4se/)
