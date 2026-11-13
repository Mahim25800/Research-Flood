# PX1 handoff — 12 November 2026 validity repair

Latest prior handoff: 10 Nov 20:20. Repository fetched to
`F:\Coding\Research\Research-Flood`, branch `codex/publication-validity`, base `a3ef825`.
Changes are local and uncommitted. No remote publication, GPU run or new evaluation-data download.

## Latest update: baseline repair and Kaggle preparation

### Kaggle smoke results, 13 November

- KS smoke `mahim2580/px1-baselines-v2-ks-smoke`, run
  `20261112T141344Z_18bb5d28`: completed, local verifier passed nine outputs,
  one event/two chips. Elapsed 566.5 seconds including setup.
- S1F smoke `mahim2580/px1-baselines-v2-s1f-smoke`, run
  `20261112T180617Z_c4aeae69`: completed, local verifier passed nine outputs,
  Bolivia/two chips. Elapsed 413.9 seconds including setup.
- Both are smoke tests, not full experiments or paper performance evidence. In both
  runs default SAR recovered on both steps; scaled sensitivity did not. EATA default
  skipped one of two updates. Report sensitivities and do not interpret these tiny
  samples as a ranking. CoTTA used the plain teacher branch on both smoke examples;
  the augmented branch remains tested on CPU only.
- MM smoke `mahim2580/px1-baselines-v2-mm-smoke` hit its 35-minute cell timeout
  before evaluation. Saved setup log shows S1F download completed; no MM preparation
  completion message or evaluation outputs were saved. Full MM prep downloads an
  11.3 GB archive and prepares all regions even in smoke mode. The exact point
  inside this silent stage is not established; investigate before re-running.
  Partial artifacts: `generated/kaggle_mm_smoke/output/`.
- MM smoke retry `mahim2580/px1-baselines-v2-mm-smoke-range`, run
  `20261112T185715Z_383fce57`: passed in 353.5 seconds including setup. The new
  standalone smoke mode fetched 136.9 MiB from the ZIP, prepared 44 chips from
  `EMSR107-5` after two earlier candidates missed the eight-chip minimum, and
  evaluated two chips. All nine outputs passed local verification. CoTTA again
  used the plain teacher. Full MMFlood preparation still uses the full archive.
  Last quota: 1.02 hours, refreshing 14 Nov 06:00 Dhaka.
- OSF Amendment 1 remains unfiled; no GEOID-Flood/FloodPlanet access.

User confirmed Amendment 1 is NOT filed and authorized baseline repairs only;
GEOID-Flood/FloodPlanet remain untouched. Historical results are retained.

- Added `baselines_v2.py`: corrected TENT BN semantics, EATA redundancy and source
  Fisher anchoring, SAR two-stage reliability/SAM/recovery, and CoTTA gated teacher
  ensemble/restoration. These are documented segmentation ports, not exact original
  classification reproductions. See `BASELINES_V2_PROTOCOL.md` for all deviations.
- Added separate EATA binary-diversity and SAR entropy-scale sensitivities; both
  must be reported. No target-label hyperparameter selection is authorized.
- Fifteen CPU mechanism tests passed. Synthetic actual-UNet pipeline passed all
  nine settings, both KS label-count definitions and output overwrite refusal.
  Evidence: `generated/test_artifacts/059175fbfff84090951f73ae82b7a38e/test_report.json`.
- Prepared six self-contained smoke/full notebooks for existing KS/S1F/MM data,
  with runtime tests, fresh directories, input/code/output hashes and verification.
  See `BASELINES_V2_RUNBOOK.md`. These supersede the single TENT audit launcher.
- No Kaggle GPU execution has occurred. Real-data smoke runs, runtime/memory checks,
  complete baseline evaluation and new-result analysis remain outstanding.

The historical next-priority items 1–2 below are superseded by this package;
the next action is its real-data Kaggle smoke sequence. Other priorities remain.

## Completed in the preceding validity pass

- Audited TENT/EATA/SAR/CoTTA implementations against primary repositories. See
  `BASELINE_FIDELITY_AUDIT.md`. Legacy results are retained and visibly labelled as variants.
- Added `tent_batch` (TENT-B) to `src/tta_px1.py`: official BN batch-statistics semantics with
  the existing event-wise schedule. `tent` retains historical behaviour and result keys.
- Added a checked bundle patch and an offline-packaged Kaggle notebook builder.
  Notebook: `generated/tent_batch_audit.ipynb`; source fingerprints alongside it.
  It defaults to existing Kuro Siwo data. It is not yet empirically validated on Kaggle.
- Five CPU tests passed, checking inference BN semantics, non-BN parameter freezing,
  independence from evaluation labels, legacy behaviour and Kaggle patch-chain compatibility.
- Repaired table/figure output paths; regenerated seven tables and three figure pairs
  from the unchanged `rev_analysis_cpu3.json`.
- Corrected manuscript baseline labels, abstract/conclusion, equivalence claims, calibration
  qualifications, interpretation of H4, and unsupported causal explanations. Added Song and Lee
  (2026) to a separate `audit_refs.bib` so regeneration of the main bibliography cannot erase it.
- Identified that the historical Otsu dark-share statistic uses label-valid support. Its existing
  results are retrospective; they cannot support a label-free operational rule.
- Added `prereg/AMENDMENT_2_DRAFT_VALIDITY.md` as an unfiled discussion draft. Registered text
  and Amendment 1 are unchanged. User confirmation of Amendment 1 filing remains pending.

## Validation and limits

Commands from repo root:
```
python revision/test_validity.py
python revision/check_paper_source.py
python revision/build_tent_audit.py
python analysis/make_tables.py revision/results/rev_analysis_cpu3.json
python analysis/paper_figs_rev.py revision/results/rev_analysis_cpu3.json
git diff --check
```
`pdflatex` was blocked by lean-ctx's command allowlist. The tracked `main_wip.pdf` remains
the older draft; do not claim the current source has compiled or been visually verified.
The full 91-reference content audit is unfinished. No publication readiness is claimed.

## Next priorities

1. Smoke-test TENT-B on Kaggle, then evaluate existing datasets/seeds; do not mix it with old `tent`.
2. Finish reference-faithful EATA/SAR/CoTTA protocols and tests before launching old GPU A/B plans.
3. Add complete component/post-processing ablations and quantify terrain missingness.
4. Fix artifact provenance/duplicate merging before final analysis; audit calibration model identity
   and statistical dependence. A shared calibration target is not exact matching of achieved risk.
5. Finalise preregistration decisions and obtain filing confirmation before GEOID-Flood/FloodPlanet.
6. Complete nearest-paper full-text comparison; verify SAR-compatible foundation model inputs
   (standard Prithvi-EO-2.0 is six-band HLS optical, not a drop-in Sentinel-1 fallback).
7. Freeze, rerun, regenerate, reconcile all prose numbers, compile and visually inspect the paper.

GPU quota reset reported by prior handoff: Sat 14 Nov 06:00 Dhaka; not checked live this session.
Existing project rules remain: no PC GPU work, no password/API-key handling, no deletion of
existing data, no account creation, and all project files under F:\Coding\Research.
