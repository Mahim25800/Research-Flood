# ImpactMesh darkness-only ablation and U-Net crossover

This is a new, post-hoc follow-up on the original 13 ImpactMesh activations and 516 chips. Every full Kaggle job evaluates one seed and both backbones. Each backbone compares ordinary self-training, the radar-darkness rule with terrain gates removed, and the original joint radar-and-terrain rule. The shared schedule uses 50 AdaBN batches and 200 adaptation steps. The darkness-only arm retains both radar darkness thresholds, radar validity and the original teacher confidence and conflict rules.

The primary metric is floodwater AP on the original scored support. Map metrics use each backbone's frozen Sen1Floods11 source threshold, 0.430 for ResNet-50 and 0.208 for U-Net. Checkpoint training recipes differ across backbones. Crossover comparisons therefore measure portability across those trained models on the same imagery.

The private payload includes only hash-verified ImpactMesh prepared inputs, existing source weights, metadata and allowlisted frozen code. The archive SHA-256 is f42dea09a71f77d40ef108ca4d4dd960f43871c6222a49e969a390c3a294209c. The frozen original results are read-only. The experiment uses no Kuro Siwo or GEOID-Flood inputs.

Each completed seed job stores unrounded scores, metrics, contingency counts, per-arm adaptation diagnostics and timings. The verifier checks all arrays against the frozen labels and recomputes AP, IoU, misses and false alarms. It averages seeds within activations before computing paired differences.

- Input dataset: https://www.kaggle.com/datasets/mostofaseum/px1-impactmesh-ablation-crossover-inputs
- Smoke job: https://www.kaggle.com/code/mostofaseum/px1-impactmesh-ablation-crossover-smoke
- Full jobs: https://www.kaggle.com/code/mostofaseum/px1-impactmesh-ablation-crossover-seed0, https://www.kaggle.com/code/mostofaseum/px1-impactmesh-ablation-crossover-seed1, https://www.kaggle.com/code/mostofaseum/px1-impactmesh-ablation-crossover-seed2
