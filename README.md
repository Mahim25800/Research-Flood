# Safe test-time adaptation for SAR flood mapping

Code, stored results and manuscript for **"Safe test-time adaptation for SAR flood mapping: physics-checked self-training with a flood-level miss-rate guarantee"**.

A flood-mapping network trained on some floods (Sentinel-1 radar) is adapted to a new flood without any labels:

1. **AdaBN**: batch-normalisation statistics are re-estimated on the new flood.
2. **Physics-checked self-training**: a mean teacher produces confident pseudo-labels, which are corrected with a per-flood radar darkness test (VV and VH, Otsu-based, capped) and terrain limits (HAND < 5 m, slope < 3°) before each student update.
3. **Flood-level miss-rate guarantee**: the decision threshold is set by conformal risk control on other flood events, so the expected share of missed flood water per flood stays below a chosen target α.

## Main results

| | Sen1Floods11 (11 floods, leave one out) | Kuro Siwo (26 events) | MMFlood (30 activations, pre-registered) |
| --- | --- | --- | --- |
| No adaptation, IoU at 0.5 / 20% guarantee | 0.551 / 0.479 | 0.186 / 0.565 | 0.193 / 0.110 |
| TENT | 0.504 / 0.442 | 0.576 / 0.467 | 0.192 / 0.096 |
| AdaBN | 0.573 / 0.479 | 0.570 / 0.562 | 0.194 / 0.098 |
| **Proposed** (AdaBN + adaptive physics check) | 0.561 / 0.494 | **0.673 / 0.677** | 0.203 / **0.112** |

* The mean miss rate stayed at or below the target on all three datasets.
* On MMFlood (fixed before testing) the proposed method beat TENT on 28/30 and AdaBN on 28/30 activations under the 20% guarantee (Holm-corrected p < 1e-7).
* Results held with an ImageNet-pretrained ResNet-34 backbone and under one-at-a-time changes of the method's settings.

## Repository layout

| Folder | Contents |
| --- | --- |
| `src/` | Data download, U-Net baseline, training, test-time adaptation, conformal calibration, Kuro Siwo evaluation, HAND/DEM fetching |
| `mmflood/` | MMFlood preparation (overlap exclusion, chipping), evaluation, pre-registered analysis, per-activation tables, raw-data check |
| `extensions/` | ResNet-34 backbone patch, mixed Sen1Floods11 + Kuro Siwo training, second-backbone analysis, sensitivity analysis, example maps |
| `kaggle/` | Kaggle runner and the exact notebook cells used (scripts embedded with MD5 checks) |
| `scripts/windows/` | Batch files used for the local (Windows, GPU) runs |
| `analysis/` | Scripts that make every paper figure from stored results |
| `data/` | Small metadata files (Kuro Siwo event locations, MMFlood activations used, per-activation results) |
| `results/` | Stored probability histograms and summaries behind every table and figure |
| `paper/` | LaTeX manuscript (Elsevier CAS template), figures, references, highlights, graphical abstract |
| `docs/` | Annotated literature list (91 references) |

## Reproducing

```bash
pip install -r requirements.txt
python src/download_sen1floods11.py --out data
python src/train_px1.py --data data --out runs              # U-Net, three seeds
python src/conformal_px1.py ...                             # leave-one-flood-out calibration
python src/ks_download.py --out data_ks                     # Kuro Siwo test events
python src/ks_eval.py --data data --ks data_ks --runs runs --out runs_ks --methods none,tent,adabn,physa,adabn_physa
python mmflood/mm_prep.py ...; python mmflood/mm_eval.py ...; python mmflood/mm_analyse_act.py ...
```

The figures can be rebuilt from `results/` alone, e.g. `python analysis/paper_figs.py results`.
GPU runs were done on a local RTX GPU and on Kaggle (T4 x2); `kaggle/` contains the exact cells.

## Data

* Sen1Floods11 (Bonafilia et al., 2020), Kuro Siwo (Bountos et al., 2023), MMFlood (Montello et al., 2022), GLO-30 HAND.
* Datasets and model weights are not included; the download scripts fetch the public data.

## Citation

If you use this code, please cite the paper (citation will be added after publication).
