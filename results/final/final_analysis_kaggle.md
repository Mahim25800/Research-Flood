Final Kaggle run (notebook notebookd6a6e506fc, 10 Nov 2026), U-Net, Kuro Siwo, radar only (V0).

| Method | 0.5 | 10% guar. | 20% guar. | 30% guar. | 20%: mean miss rate | 20%: events meeting |
| --- | --- | --- | --- | --- | --- | --- |
| none (model 0) | 0.258 | 0.490 | 0.586 | 0.561 | 0.170 | 16.0/26 |
| tent (model 0) | 0.586 | 0.358 | 0.530 | 0.579 | 0.177 | 19.0/26 |
| adabn (model 0) | 0.550 | 0.397 | 0.560 | 0.578 | 0.170 | 16.0/26 |
| physa (3 models) | 0.627 | 0.500 | 0.666 | 0.655 | 0.171 | 19.0/26 |
| adabn_physa (3 models) | 0.673 | 0.565 | 0.677 | 0.644 | 0.170 | 17.0/26 |

Sensitivity, AdaBN + adaptive physics, model seed 0:
| Setting | 0.5 | 10% | 20% | 30% | miss rate 20% | meeting |
| --- | --- | --- | --- | --- | --- | --- |
| default | 0.654 | 0.561 | 0.665 | 0.625 | 0.170 | 16/26 |
| cap_m2 (-17/-24) | 0.655 | 0.543 | 0.659 | 0.625 | 0.170 | 17/26 |
| cap_p2 (-13/-20) | 0.640 | 0.576 | 0.655 | 0.616 | 0.170 | 16/26 |
| hand3 | 0.655 | 0.561 | 0.665 | 0.626 | 0.170 | 16/26 |
| hand10 | 0.654 | 0.560 | 0.664 | 0.625 | 0.170 | 16/26 |
| steps100 | 0.628 | 0.541 | 0.638 | 0.622 | 0.172 | 15/26 |
| steps400 | 0.671 | 0.571 | 0.671 | 0.628 | 0.170 | 15/26 |
| conf80 | 0.651 | 0.567 | 0.655 | 0.619 | 0.170 | 17/26 |
| conf95 | 0.637 | 0.553 | 0.654 | 0.619 | 0.170 | 15/26 |

MMFlood Otsu (no model), 30 activations: mean IoU 0.090, median 0.063, pixel-pooled 0.087.
MMFlood raw-data check: mm_raw_checks.json.
