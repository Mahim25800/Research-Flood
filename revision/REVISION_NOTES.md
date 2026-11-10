# PX1 revision (started 10 Nov 2026, after the Q1-readiness review)

Paper: paper_latex/main.tex (backup of the pre-revision version: main_v1_backup.tex). Tables are generated:
`python make_tables.py results_rev/<json>` -> paper_latex/tables/*.tex; figures: `python paper_figs_rev.py results_rev/<json>`.
Analysis JSON comes from rev_analyse.py on Kaggle (printed as REV_JSON lines; pulled with the browser as gzip+base64
"REVB64:...:END" and stored by `python rev/pull_json.py <name>`).

Kaggle (account mahim2580), all notebooks import a tiny cell that downloads revision/*.py from
github.com/Mahim25800/Research-Flood at a pinned commit and exec()s a runner. Import trick: File > Import Notebook,
inject a File into input[type=file]; notebook metadata.kaggle.dataSources attaches the inputs
(bundle dataset version 21857658; kernel versions px1-mmflood-final 367062493, px1-adaptive-physics 366960729,
extras notebookd6a6e506fc 367828676).
- notebookf299e49801: rev_cpu.py (analysis only). v2 = results_rev/rev_analysis_cpu2.json (all stored methods, precision).
- notebookda878fed05: rev_cpu2.py (change detection on Kuro Siwo) DONE -> results_rev/ks_change_summary.json.
- notebook42da559468: rev_gpu.py v1 started 10 Nov ~14:25 BDT (GPU T4 x2, quota 6.2 h left this week, resets Sat 14 Nov 06:00 BDT).
  Runs EATA, SAR, CoTTA + none/tent/adabn/adabn_physa again on Kuro Siwo (3 models, slot 1 = flood-water-only scores),
  then ks_finetune (5-fold supervised reference), then rev_analyse. Ends with REV_GPU_DONE. Its own analysis uses the OLD
  rev_analyse (no precision, fewer methods): after it finishes, re-run notebookf299e49801 with its kernel version id
  added to dataSources and the newest commit, then pull the JSON, regenerate tables/figures, fill the paper.

Still to write into the paper after the GPU run: EATA/SAR/CoTTA rows (tables come automatically), section on label
convention (KSf), pre-flood change detection (numbers above) and the fine-tuned reference; abstract <= 250 words;
update "six adaptation baselines" claims; Table tab:prec; check every number in the prose against the tables.

After the quota reset (Sat 14 Nov): MMFlood with EATA/SAR/CoTTA (3 models), Sen1Floods11 LOEO with EATA/SAR/CoTTA,
and a SAR-pretrained backbone (SSL4EO-S12 ResNet-50 MoCo, Sentinel-1) on Kuro Siwo + MMFlood.
Round JSON numbers to 5 decimals in rev_analyse next time (0.5505 printed as 0.550, paper used 0.551).

## Top-journal route (user agreed 10 Nov 14:36 BDT)
1. Foundation-model backbone: TerraMind (takes Sentinel-1; terratorch has a Sen1Floods11 config, github.com/IBM/terramind) or
   Prithvi 2.0 (benchmarked on S1 in arXiv 2511.01990: 0.57 mIoU on FloodPlanet). Replace AdaBN by layer-norm adaptation;
   physics-checked self-training unchanged. Needs GPU quota after Sat 14 Nov. NOT started.
2. New publicly pre-registered test: draft in preregistration_2_DRAFT.md (sent to user). Primary data = GEOID-Flood held-out
   2026 events (arXiv 2608.02315, github.com/links-ads/geoid-flood; release status/units NOT verified, URL could not be
   fetched); fallback FloodPlanet (dc.cyverse.org/dataset/floodplanet). DO NOT download either before the user deposits on OSF.
3. Stratified MMFlood analysis by a label-free criterion (post hoc), no GPU.
GEOID-Flood verified from its repo README (cloned, docs only): data on HF links-ads/geoid-flood, held-out tree
geoid-flood-heldout = EMSR857-871, 1,301 tiles, get with scripts/get_data.py --tree geoid-flood-heldout --layer s1grd label.
Repo has terratorch configs incl. terramind_base. Pre-registration draft updated (H1-H6, safer set). Open: H6 "open water
visible" rule (derive on MMFlood/Kuro Siwo from stored Otsu counts or raw backscatter, label-free), TerraMind check,
commit hash; then the USER deposits on OSF; only then download.

## OSF (10 Nov 2026 ~14:55 BDT)
User registered preregistration_2_DRAFT.md AS IT WAS on OSF: https://osf.io/c38jx (public). Frozen copy:
preregistration_2_REGISTERED_osf_c38jx.md. It still contains [CHECK] items (commit hash, H6 rule, TerraMind).
Plan: finalise H6 rule (from stored MMFlood/Kuro Siwo results, label-free) + commit hash + TerraMind detail, send the
user a final document to file as an UPDATE on OSF; no GEOID-Flood/FloodPlanet download before the user confirms the update.
Whether the uploaded file was the first (H1-H5 old) or second draft (H1-H6) was not confirmed; user said "as it is"
(the last file sent was the H1-H6 version).
