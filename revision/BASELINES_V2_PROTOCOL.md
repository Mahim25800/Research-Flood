# Corrected baseline protocol — 12 November 2026

Status: implemented and CPU tested; not yet run on Kaggle GPU. These are explicit
event-wise binary-segmentation ports, not exact reproductions of the original
classification experiments. Historical results and method keys are retained.
This document is a proposed computational protocol, not a filed preregistration.
Amendment 1 has NOT been filed, as confirmed by the author. No GEOID-Flood or
FloodPlanet data may be accessed under this work package.

## Scope and schedule

Existing UNet V0 checkpoints only. KS: 26 events, source seeds 0/1/2. MMFlood:
69 regions from 30 activations, source seeds 0/1/2. Sen1Floods11: 11 held-out
events, existing LOEO seed-0 checkpoints. Regions from one MMFlood activation
must remain grouped in subsequent statistical analysis; source seeds are not
independent events. ResNet baselines and component ablations are outside this run.

Every method starts from the same event-specific source checkpoint. Each event
receives 200 adaptation steps with batches of eight random 256-pixel square
crops (smaller only if images require it), rotations and horizontal flips.
The crop RNG is fixed at 0 across methods; checkpoint seed varies independently.
Normalisation uses source-training images, excluding the held-out S1F event.
Labels never enter the four new adapters. Final prediction performs no optimizer
update and uses per-image batch statistics. This is transductive event adaptation,
not the original continual online evaluation protocol.

Smoke mode uses one event, two chips, two adaptation steps and two Fisher batches;
it retains the full 32-view CoTTA prediction rule. Smoke results are not paper evidence.

## Methods and declared choices

| Output key | Mechanisms and settings |
|---|---|
| `tent_v2` | Entropy minimisation; BN affine parameters only; running statistics disabled during adaptation AND prediction; Adam, lr 0.001. |
| `eata_v2` | Pixel reliability entropy < 0.4 ln(2); cosine redundancy filter against moving mean probabilities (decay 0.9), margin 0.05; detached entropy weighting; source Fisher anchoring weight 2000; Adam, lr 0.001. Empty selected sets skip the update. |
| `sar_v2` | Reliability filtering before and after SAM perturbation (radius 0.05); BN affine only; Adam, lr 0.001; entropy EMA decay 0.9 and recovery threshold 0.2. Recovery restores source parameters, optimizer and clears EMA. Perturbations are restored on empty selections and exceptions. |
| `cotta_v2` | All-parameter Adam, lr 0.0001; fixed source anchor; EMA teacher decay 0.99; anchor confidence threshold 0.9; 32 inverse-aligned augmented logit views below threshold; stochastic source restoration probability 0.01; final prediction uses the gated teacher. |

EATA Fisher is estimated once per checkpoint from 20 source-only pseudo-labelled
batches with seed 1729, averaging squared batch-mean cross-entropy gradients.
This is a batch-gradient approximation, not a per-pixel empirical Fisher estimate.
Source labels are not loaded for this computation. LOEO excludes the held-out
event from Fisher and normalisation. Missing/incomplete Fisher tensors fail closed.

CoTTA uses invertible right-angle rotations/reflections and a shared VV/VH
radiometric offset uniformly in [-1.5, 1.5] dB, scaled by source channel standard
deviations. Anchor confidence is averaged over pixels and batch. These are declared
SAR segmentation choices replacing RGB transforms; their suitability requires
empirical evaluation. Teacher logits, not probabilities, are averaged before
softmax. The classification optimizers, schedules and all hyperparameters are
not reproduced exactly: the shared PX1 schedule and Adam are explicit adaptations.

### Prespecified exploratory binary sensitivities

- `eata_v2_d09`: identical EATA port except cosine margin 0.9. In two classes,
  margin 0.05 can reject almost all later pixels, especially near a balanced mean.
- `sar_v2_scaled`: identical SAR port except recovery threshold
  0.2 ln(2)/ln(1000), to expose dependence on entropy scale.

Report BOTH settings, update/skip counts and recovery counts; do not choose the
best setting using evaluation labels or suppress a setting because it changes
the proposed method's ranking. These choices precede the new runs, but are not
OSF-registered. They do not establish globally optimal baseline tuning.

The comparison controls `none`, `adabn`, `adabn_physa` use the existing implementation
and checkpoints. They are not newly validated implementations of the proposed
physics method. The package only adds alternate KS label scoring to staged legacy
code. KS retains both all-water and permanent-water-ignored histograms.

## Provenance and acceptance

Each output has a new immutable directory, code/runtime metadata, checkpoint and
input fingerprints, method settings, event coverage and histogram checksums.
The verifier rejects incomplete, duplicate, extra or checksum-mismatched artifacts
and rejects smoke outputs unless explicitly allowed. Historical aggregators must
not be pointed at the new files without a deliberate schema-aware analysis update.
Threshold calibration, statistical comparisons and revised manuscript numbers
remain subsequent work; none are inferred from synthetic testing.

CPU evidence: 15 mechanism tests; an actual UNet worker using synthetic raster
inputs passes all nine settings, both label-count checks and overwrite refusal.
This does not test network downloads, GPU memory, real-data convergence or runtime.

## Primary implementation references

- TENT: https://github.com/DequanWang/tent/blob/master/tent.py
- EATA: https://github.com/mr-eggplant/EATA/blob/main/eata.py
- SAR: https://github.com/mr-eggplant/SAR/blob/main/sar.py
- CoTTA: https://github.com/qinenergy/CoTTA/blob/main/cifar/cotta.py

Reference mechanisms were inspected during this repair. Upstream links are mutable;
our exact evaluated code is embedded and fingerprinted in each notebook and output.
