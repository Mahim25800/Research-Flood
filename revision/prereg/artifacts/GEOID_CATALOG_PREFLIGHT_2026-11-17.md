# GEOID-Flood v1.1 preflight, 17 November 2026

The user authorized the frozen external evaluation. Accessed the public tree
metadata and the pinned release's `geoid-flood-heldout/tile_catalog.parquet`.
No radar, DEM, HAND or target label rasters were downloaded. No model ran.

- Release commit: `4f1b9c52da789dbad9333e8f62c7b27ff48304bd`.
- Catalog SHA-256: `9b531b90ca534d30ed07d5d6c0d1b85133e7f8228c7c67c87c0635a955bca165`.
- Local catalog: `revision/generated/geoid_frozen_execution_20261117/tile_catalog.parquet`.
- The catalog has all 1,301 unique event/AOI/tile keys and all rows are in test.
- Activation and tile counts: EMSR857 (208), EMSR858 (10), EMSR861 (580),
  EMSR864 (157), EMSR865 (51), EMSR866 (191), EMSR868 (31), EMSR870 (31),
  EMSR871 (42). Six other activation numbers in the published range do not
  appear in this catalog. This is distinct from later chip eligibility checks.

The published protocol ZIP's evaluator requires that the actual activation set
equal every consecutive number from EMSR857 to EMSR871. This would reject the
official inventory before inference. The attached patch changes only that
validation constant to the nine IDs in the pinned catalog. It leaves the
1,301-tile check, data selection, model, checkpoints, calibration, outcomes,
and statistics unchanged. It has not been applied to the published archive.

Overlap screening so far: Kuro Siwo's local inventory dates end in 2022 and
Sen1Floods11 acquisition dates end in 2019. No overlap in EMS activation IDs
with the MMFlood development manifest. The GEOID-Flood activations have event
dates from January to March 2026. Retain the source records for the overlap
audit. Count eligible activations only after applying the registered rules.

Proposed protocol note:

> A metadata-only check of the pinned GEOID-Flood v1.1 held-out tile catalog
> confirmed 1,301 tiles across nine activations within the EMSR857–EMSR871
> range. The evaluator's coverage check previously required all fifteen
> consecutive activation numbers. We correct the check to require the nine
> catalog activations listed above. The dataset revision, tile inventory,
> eligibility rules, methods, checkpoints, outcomes, and statistical tests
> remain as previously specified. The discrepancy was discovered before
> downloading radar or label rasters or running inference.
