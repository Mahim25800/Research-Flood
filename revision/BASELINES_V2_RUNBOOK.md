# Kaggle baseline runs

Prepared 12 November 2026. See `HANDOFF_2026-11-12_VALIDITY.md` for Kaggle smoke results.
Read `BASELINES_V2_PROTOCOL.md` before interpreting these runs.

## Launch sequence

1. Import `generated/baselines_v2/ks_smoke.ipynb` into a NEW Kaggle notebook/session.
2. Select GPU (T4 x2 if available), enable Internet, and attach the existing PX1
   checkpoint bundle (dataset version source ID 21857658, as in the prior notebooks).
   Verify the input attachment in Kaggle; imported metadata may not attach it automatically.
3. Run all. Require `BASELINES_V2_NOTEBOOK_DONE`, a completed `run_manifest.json`,
   and successful output verification. Inspect logs for finite outputs and method
   update/skip/recovery counts. A synthetic test pass does not replace this smoke run.
4. In separate fresh sessions, use `s1f_smoke.ipynb` and `mm_smoke.ipynb` before
   committing compute to those datasets.
5. Once each smoke run passes, launch its corresponding `ks_full.ipynb`,
   `s1f_full.ipynb` or `mm_full.ipynb`. Every full notebook repeats CPU mechanism
   tests and a real-data smoke run before full evaluation.

The standalone MMFlood smoke reads the official ZIP index and selected raster
members via HTTP byte ranges. It tries sorted regions with published coordinates
that pass the existing overlap filter until one meets the eight-chip minimum.
Its archive transfer is capped at 512 MiB, and a server that ignores byte ranges
causes an error. The full MMFlood notebook still prepares the entire collection.

All six notebooks are under `revision/generated/baselines_v2/`. Do not launch
the historical A/B scripts for the corrected comparisons. This package supersedes
the earlier single `tent_batch_audit.ipynb` for the baseline repair campaign.

KS/MM full runs cover source seeds 0, 1, 2; S1F uses existing seed-0 LOEO models.
At most one worker runs per GPU. CoTTA's 32-view teacher is expensive: actual
wall time and memory remain unmeasured. The worker monitor has a 10.5-hour budget
including elapsed setup time, but setup itself is not forcibly time-limited.
If a notebook exceeds budget, retain its partial outputs and logs. Do not silently
reduce steps/views or reuse its output directory. Report the failure and revise
the launch partitioning while retaining identical scientific settings.

## Outputs and verification

Save the entire `/kaggle/working/baselines_v2/<timestamp_uuid>/` output, including
code, environment.txt, setup logs, worker logs and manifests. Do not copy it over
historical results. Every new session gets a new directory; pre-existing `/tmp/px1`
or `/tmp/px1_override` causes a refusal rather than deletion.

Validate a downloaded full worker directory with:

```powershell
python revision/verify_baseline_outputs.py PATH_TO_FULL_SEED_DIRECTORY
```

Use `--allow-smoke` only when checking a smoke directory. Successful workers have
`BASELINES_V2_DONE`; successful notebooks have `BASELINES_V2_NOTEBOOK_DONE`.
Full workers require KS 26 events, S1F 11 events or MM 69 regions/30 activations.
Missing checkpoints, unexpected coverage and incomplete artifacts are errors.

The notebook uses existing project data-preparation scripts for S1F, Kuro Siwo,
MMFlood and terrain. It contains no GEOID-Flood/FloodPlanet acquisition step.
Amendment 1 is not filed; that external-data work remains on hold.

## Local reproducibility

```powershell
python revision/test_baselines_v2.py
python revision/test_baseline_pipeline.py
python revision/build_baseline_runs.py
```

The pipeline test uses synthetic rasters and CPU only; fixtures/logs are retained
under ignored `revision/generated/test_artifacts/`. The builder compiles embedded
Python and notebook cells and records payload hashes in `sources.json`. No Kaggle
submission or GPU execution occurs during these local commands. A successful
baseline campaign still needs calibration, event-level statistical analysis and
manuscript reconciliation before publication claims can be reassessed.
