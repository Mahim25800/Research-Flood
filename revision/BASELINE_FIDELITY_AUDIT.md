# Baseline fidelity audit — 12 November 2026

Inspected base revision: a3ef825. Historical arrays and registered documents remain unchanged.
This audit supersedes the handoff's recommendation to launch GPU A/B unchanged.

| Stored method | Correct interpretation | Missing or different behaviour | Action |
|---|---|---|---|
| `tent` | TENT-R, event-wise entropy adaptation with running-statistics inference | Reference TENT disables BN running statistics | Added separate `tent_batch` / TENT-B diagnostic; rerun all existing datasets |
| `eata` | EATA-inspired | No redundancy filter; Fisher estimated on target batches; running-statistics inference | Do not claim reproduction; implement and document segmentation/source-Fisher protocol before rerun |
| `sar` | SAR-inspired | No collapse recovery; running-statistics inference; architecture/parameter selection differs | Restore and test recovery and specify binary-class scaling before rerun |
| `cotta` | CoTTA-inspired | No source-confidence gate; four SAR augmentations; probability averaging and running-statistics inference | Specify SAR-compatible reference protocol and test before rerun |

Primary implementation sources inspected 12 Nov:
- https://github.com/DequanWang/tent/blob/master/tent.py
- https://github.com/mr-eggplant/EATA/blob/main/eata.py
- https://github.com/mr-eggplant/SAR/blob/main/sar.py
- https://github.com/qinenergy/cotta/blob/main/cifar/cotta.py

TENT-B fixes only BN semantics, retaining the event-wise 200-step schedule, pixel entropy,
Adam 1e-3 and per-chip inference. It is not claimed to reproduce every original online protocol.
Fresh loading per event resets the model. The historical `tent` key remains unchanged.
CPU regression tests check BN inference semantics, frozen non-BN weights, label independence,
and patch compatibility. They do not establish empirical accuracy or algorithm superiority.

## Next run

`python revision/build_tent_audit.py` builds a self-contained Kaggle notebook using the existing
bundle, embedding current code and SHA256 fingerprints. It defaults to Kuro Siwo; choose `s1f`
or `mm` with `PX1_AUDIT_DATASET` for separate runs. Do not run it on this PC.
It preserves diagnostics in a separate directory, records job failure and does not print a
success marker after failed jobs. Validate bundle compatibility with a small Kaggle smoke run
before a full run. No GPU job was launched during this audit.

The old GPU A/B scripts still reproduce custom variants. Do not launch them as reference-method
comparisons until EATA/SAR/CoTTA fidelity is resolved. The new notebook does not use GEOID-Flood.

## Other issues requiring follow-through

1. The Otsu dark-share statistic in `rev_analyse.load_otsu` uses label-valid support and a labelled
   histogram denominator. It is retrospective, not a fully label-free deployment indicator.
   A new indicator must use a predeclared radar-valid mask independent of ground truth.
2. Source-only curves use leave-one-event-out source models, unlike external-event models.
   Treat transfer as empirical until an appropriate exchangeability proof or split protocol exists.
3. Overlapping LOEO calibration sets couple event scores. Historical Wilcoxon p-values alone
   do not establish independent-event inference for the full fitted calibration procedure.
4. The supervised reference's nonsignificant difference is not equivalence/non-inferiority.
5. `rev_analyse.load` merges artifacts by path/key, sometimes retaining one run's all-water slot
   and another run's flood-only slot. A provenance-aware manifest and strict duplicate checks
   are required before treating reruns as a new final result set.
6. Missing HAND falls back to relative height; quantify missingness and predeclare handling.
7. Complete no-HAND/no-slope/darkness-only and post-processing ablations; freeze before external test.
8. Song and Lee (2026), DOI 10.1109/LGRS.2026.3718817, is close related work. Institutional abstract
   verified; full-text/implementation comparison remains outstanding.
# Implementation update — 12 November 2026

The historical findings below motivated a separate corrected implementation in
`baselines_v2.py`. See `BASELINES_V2_PROTOCOL.md` for mechanisms, settings and
declared segmentation deviations, and `BASELINES_V2_RUNBOOK.md` for six Kaggle
notebooks. CPU mechanism and synthetic pipeline tests passed; real-data GPU runs
remain outstanding. Old result keys and measurements have not been replaced.

