# When do dark-water pseudo-labels help test-time adaptation in SAR flood mapping?

PX1 studies whether a model trained on Sentinel-1 radar can improve its floodwater maps when it adapts to a new event without target labels. We compare ordinary teacher–student self-training with an adaptive VV/VH darkness rule, with and without terrain gates based on height above nearest drainage (HAND) and slope.

[Current manuscript](paper/main.pdf) · [Supplement](paper/supplement.pdf) · [Verified results](results/README.md) · [Reproduction bundle](revision/reproduction/px1_external_20261125/) · [GEOID-Flood preregistration](https://osf.io/wv4se/)

## Main findings

The table compares joint radar-and-terrain self-training with standard ordinary self-training. Floodwater average precision (AP) excludes permanent water and ignored pixels. Gains are in percentage points, calculated before rounding.

| Cohort | Backbone | Ordinary AP | Joint AP | Gain | Activations improved |
| --- | --- | ---: | ---: | ---: | ---: |
| ImpactMesh-Flood | U-Net | 47.91% | 60.09% | +12.17 | 12/13 |
| ImpactMesh-Flood | SSL4EO-S12 ResNet-50 | 45.81% | 59.95% | +14.15 | 12/13 |
| GEOID-Flood | U-Net | 67.75% | 67.91% | +0.17 | 8/9 |
| GEOID-Flood | SSL4EO-S12 ResNet-50 | 66.39% | 67.58% | +1.19 | 8/9 |

- **Preregistered GEOID-Flood result.** The primary U-Net contrast was against same-rule post-processing. Joint self-training gained **3.50 AP points** (8/9 activations, one-sided exact sign test *p* = 0.0195). The ResNet-50 GEOID-Flood follow-up was post hoc.
- **What drove the gain.** Radar darkness accounted for most of the observed AP gain on ImpactMesh and in the GEOID-Flood ResNet-50 follow-up. On GEOID-Flood, darkness alone added **0.99 AP points** over ordinary self-training. HAND and slope added **0.20 points** beyond darkness alone (95% paired interval −0.16 to +0.72).
- **Stronger ordinary control.** On ImpactMesh, lowering the teacher’s positive cutoff from 0.90 to 0.70 cut misses and raised false alarms. Joint physics still gained **11.47 AP points** and improved recall by **11.91** and **13.33 points** at matched 2% and 5% false-positive rates on the same Kaggle T4 stack.
- **Paired-input comparison.** On STANet, the physics variants reduced misses and increased false alarms at the fixed threshold. Mean miss rates were similar to the independent TG-TTA port at matched false alarms. [Detailed results](results/README.md)

The larger gain on ImpactMesh recurred across both backbones. The comparison leaves open which differences in observations, labels, adaptation protocols or source training explain the pattern. Matched false-alarm rates use reference labels retrospectively. Historical conformal calibration experiments and their assumptions are documented in the supplement.

## Reproduce the headline numbers

From the repository root, run:

```bash
python revision/reproduction/px1_external_20261125/verify_headlines.py
```

The bundle includes nine source checkpoints, frozen protocols, prediction manifests, verified summaries and scripts. The headline script aggregates the archived activation-level records using only Python’s standard library. Exact per-pixel recomputation requires the referenced score arrays and prepared labels, which are available through the corresponding author. See the [reproduction guide](revision/reproduction/px1_external_20261125/README.md) and [pinned release](https://github.com/Mahim25800/Research-Flood/tree/e233cddf7ba7a9518bf1d5ecc7d410124f93f311).

## Repository guide

| Path | Contents |
| --- | --- |
| [`paper/`](paper/README.md) | Current manuscript and supplement, LaTeX source and highlights |
| [`results/`](results/README.md) | Verified external results index and historical experiments |
| [`revision/`](revision/README.md) | Project status, protocol history and reproduction materials |
| [`src/`](src/) | Data preparation, source training, adaptation and calibration |
| [`analysis/`](analysis/) | Analysis for the historical benchmarks |
| [`mmflood/`](mmflood/) | Historical MMFlood evaluation |
| [`docs/`](docs/) | Literature and background documentation |

The manuscript is a working draft. See [CITATION.cff](CITATION.cff) for repository citation metadata.
