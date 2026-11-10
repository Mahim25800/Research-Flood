# PX1 handoff — state as of Tue 10 Nov 2026, 20:20 (Asia/Dhaka)

Paste this file into a new session to continue the project. Everything needed is in the GitHub repo and on the PC; the cloud workspace of the old session is not shared.

## 1. The project in one paragraph
A paper on Sentinel-1 (SAR) flood mapping. Models trained on Sen1Floods11 are adapted to each new flood without labels (AdaBN, then self-training on pseudo-labels checked against radar darkness, HAND and slope), and the map threshold is calibrated across flood events so the miss rate stays at a chosen target. Evaluated on Sen1Floods11 (11 floods, leave one out), Kuro Siwo (26 events) and MMFlood (30 activations, pre-registered). Goal: a top Q1 remote sensing journal (ISPRS JPRS / TGRS / RSE). Template: Elsevier CAS single column.

- **Current title:** Physics-checked test-time adaptation for SAR flood mapping with flood-level miss-rate calibration
- **Authors:** MD Sakib Hasan, Mostofa Seum, Meer Mursalin, Tanvir Ahmed (corresponding), Rifath Mahmud, Syeda Anika Tasnim — Dept. of Computer Science, AIUB.

## 2. Where everything is
| What | Where |
| --- | --- |
| Code, results, paper (public) | https://github.com/Mahim25800/Research-Flood (branch `main`, last commit `ff6f7a8`) |
| Revision work in progress | repo folder `revision/` (see its `REVISION_NOTES.md`) |
| Revised paper draft (mid-revision) | `revision/paper_wip/main.tex`, `main_wip.pdf`; the submitted-style earlier version is `paper/` and `revision/paper_wip/main_v1_backup.tex` |
| Analysis results of the revision | `revision/results/*.json` (`rev_analysis_cpu3.json` is the current one) |
| Table and figure generators | `analysis/make_tables.py`, `analysis/paper_figs_rev.py` (read the analysis JSON) |
| Files on the PC | `F:\Coding\Research` (data, model weights, logs, `paper_writing\latex`). Rule: store nothing outside this folder; C: is full; no GPU work on the PC |
| Kaggle | account `mahim2580`, GPU T4 x2, 30 h/week, resets Saturdays 06:00 Dhaka. Dataset `mahim2580/px1-flood-bundle` holds code + model weights |
| Pre-registration 2 | OSF https://osf.io/c38jx (public, registered 10 Nov 2026). Exact text: `revision/prereg/preregistration_2_REGISTERED_osf_c38jx.md`. Amendment 1 (to be filed by the user as an update): `revision/prereg/preregistration_2_AMENDMENT_1.md` |
| Running notes document | Claude Doc `b6ab55dd-ae64-4119-83db-4743b1a218c9` (PX1 tab) — older results write-ups |

## 3. Results so far (all in the repo)
**Original experiments (complete):**
- Kuro Siwo, same-dataset calibration, 20% miss-rate target: proposed 0.677 IoU, no adaptation 0.565, AdaBN 0.562, TENT 0.468. At threshold 0.5: proposed 0.673, TENT 0.576, AdaBN 0.570, none 0.186.
- MMFlood pre-registered test: H1 (beats TENT) and H2 (beats AdaBN) supported, 28/30 activations each; H3 (beats no adaptation at 0.5) not supported (11/30); H4 (mean miss rate ≤ 0.20) supported (0.177). Absolute IoU about 0.2 at 0.5 and 0.11 at the 20% target.
- ResNet-34 backbone and sensitivity analysis done. On MMFlood at 0.5 the method is *below* no adaptation for ResNet-34 (0.188 vs 0.192; 0.182 vs 0.207 with extra training floods).

**Revision analyses done on 10 Nov (from stored histograms, no GPU):**
- **Source-only calibration** (threshold set on Sen1Floods11 only, no target labels, 20% target). Kuro Siwo: no adaptation IoU 0.366 / miss 0.583; TENT 0.346 / 0.073; AdaBN 0.425 / 0.072; proposed 0.557 / 0.062 with 25 of 26 events under the target. MMFlood: fails for none (0.28), AdaBN (0.24), proposed (0.26); only TENT under (0.19).
- **Per-flood rule** (conformal quantile over events, α = δ = 0.2): 80–83% of floods meet the target on all three datasets. Kuro Siwo IoU under it: proposed 0.652, none 0.529, AdaBN 0.535, TENT 0.413.
- **Tests of "TTA harms" on MMFlood** (matched miss rate): none vs AdaBN +0.012, Holm p = 0.024; none vs TENT +0.014, p = 0.11; proposed vs none +0.002, p = 0.76.
- **Change detection with Kuro Siwo pre-flood images** (model-free): flood-water IoU 0.527; plain per-flood dark threshold 0.532 (all water: 0.560).

**GPU run of 10 Nov, collected the same evening (Kuro Siwo, 26 events, 3 models, 20% miss-rate target, same-dataset calibration):**
- Proposed 0.677 IoU; no adaptation 0.565; CoTTA 0.574, SAR 0.564, AdaBN 0.562, EATA 0.558, TENT 0.467, plain self-training 0.457. The proposed method beats each baseline on 21-25 of 26 events (Holm p <= 5e-4). None of the six baselines beats the unadapted model at a matched miss rate, although all look strong at threshold 0.5 (0.56-0.60 against 0.186).
- Supervised reference (5-fold fine-tuning on labelled Kuro Siwo floods): 0.657. Level with the proposed method (p = 0.71); do not claim the method beats it.
- Flood water only (permanent water ignored): proposed 0.565, fine-tuned 0.581, no adaptation 0.460, baselines 0.350-0.471. Order unchanged, but the worst event of the proposed method misses 75% of its flood water (38% with all water).
- Source-only calibration: physics check WITHOUT AdaBN has a higher IoU than the full method at the 20% target (0.662 against 0.557) but is unreliable (one event misses 78%; mean miss rate above target at 30%). An earlier over-claim about this was removed from the paper.
- Model-free: per-event dark threshold 0.532 IoU on flood water at a 34% miss rate, within 0.04 of the method at threshold 0.5 (0.569). Stated as a limitation.

## 4. What the review said and how the paper is changing
A critical review (10 Nov) judged the first draft not ready: title/abstract over-claimed ("safe", "guarantee"), main result not supported on MMFlood, calibration uses target-dataset labels, weak base models, missing baselines (EATA, SAR, CoTTA), no foundation model, inconsistencies, pre-registration not public.

New framing (already written into `revision/paper_wip/main.tex`):
1. Rankings change at a matched miss rate (TENT strong at 0.5, weakest when calibrated).
2. On Kuro Siwo, adaptation is what makes a source-calibrated threshold valid (58% missed → 6%).
3. Honest negative part: on MMFlood no gain over no adaptation, source-only calibration fails.

Done in the draft (23 pages, compiles, abstract about 248 words): new title, abstract, introduction, related work, calibration section, datasets, compared methods, Kuro Siwo results with all six baselines, supervised reference, flood-water-only table (`tab_ksf`), model-free and change-detection paragraph, "Calibration without target labels", "Per-flood rule", MMFlood section with qualifications, discussion, limitations (i)-(xii), conclusion, highlights, `references.bib` regenerated (DOI instead of aggregator links). Tables are generated files in `tables/`. Backup before the last edit: `main_before_gpu_results.tex` (cloud workspace only).

**Not yet done in the draft:** EATA/SAR/CoTTA on Sen1Floods11 and MMFlood (the paper says they were run on Kuro Siwo only); Sentinel-1-pretrained backbone; foundation model; confirmatory GEOID-Flood test; graphical abstract still shows the old framing; hand check of the 91 references. Do not submit yet.

## 5. Running and scheduled right now
- Nothing is running. Kaggle GPU quota: 28.0 of 30 h used this week; resets Sat 14 Nov 06:00 Dhaka.
- Finished Kaggle versions: GPU run `notebook42da559468` = kernel version `367979892`; analysis `notebookf299e49801` = kernel version `368069757` (code at commit `543e14f`, inputs: bundle `21857658`, kernels `367062493`, `366960729`, `367828676`, `367979892`).
- The old session has a self-reminder for Sat 14 Nov 06:15 Dhaka to launch the two second-round GPU runs. A new session must do this by hand (Section 6, step 2). The Claude desktop app must be open on the PC for the browser work.

## 6. To do, in order
1. **Pre-registration amendment (waiting for the user).** `preregistration_2_AMENDMENT_1.md` settles the open items: code pinned to commit `836d225`; H6 withdrawn as confirmatory (the label-free Otsu dark share did not separate floods: Kuro Siwo rho = 0.28, p = 0.16; MMFlood rho = -0.39, p = 0.03, no clean cut-off), so the confirmatory set is H1-H5 with Holm over five; foundation model = TerraMind, fallback Prithvi-EO-2.0, exploratory (its "Sentinel-1 GRD input" was written from memory, not yet checked against the model card). The user files it on https://osf.io/c38jx as an update. **No GEOID-Flood or FloodPlanet data may be downloaded before the user confirms it is filed.**
2. **After the quota reset (Sat 14 Nov 06:00 Dhaka):** launch two GPU notebooks in parallel: `rev_gpuA.py` (EATA/SAR/CoTTA on MMFlood, 3 models, ~10 h) and `rev_gpuB.py` (the same baselines on Sen1Floods11, plus a ResNet-50 U-Net with Sentinel-1 SSL4EO-S12 MoCo weights on Kuro Siwo and MMFlood, ~10 h). Then re-run the analysis and update the paper.
3. **Confirmatory test on GEOID-Flood** (after step 1 is confirmed): held-out activations EMSR857–871 (1,301 tiles; Hugging Face `links-ads/geoid-flood`, `scripts/get_data.py --tree geoid-flood-heldout --layer s1grd label` from github.com/links-ads/geoid-flood). Chips 512×512, ≤ 48 per event, unit = activation, three U-Nets, hypotheses H1–H5 as registered and amended. Needs a new prep script (like `mmflood/mm_prep.py`) and an eval script (like `mmflood/mm_eval.py`).
4. **Foundation-model backbone:** TerraMind (github.com/IBM/terramind; has a Sen1Floods11 terratorch config; confirm its Sentinel-1 input format from the model card) or Prithvi 2.0. Fine-tune on Sen1Floods11; replace AdaBN by adapting layer-norm parameters; physics-checked self-training unchanged; evaluate on Kuro Siwo, MMFlood, GEOID-Flood including source-only calibration. Not started. Budget one to two weekly quotas.
5. **Gain against dark share** (secondary analysis, no GPU): report the relation between the method's gain and the Otsu dark share (`otsu` block of the analysis JSON) on all datasets.
6. **Final paper pass:** all tables/figures regenerated (`python analysis/make_tables.py <json>`, `python analysis/paper_figs_rev.py <json>`), every number in the prose checked against the tables, graphical abstract redone in the new framing, references checked by hand (91 entries; AI-assisted search is declared), archive the repo on Zenodo for a DOI (needs the user's account), then submit.

## 7. How the Kaggle runs are operated (built-in browser of the Claude desktop app)
- Every notebook is one small cell that downloads `revision/*.py` from the GitHub repo at a pinned commit and `exec`s a runner (see `revision/README.md`).
- To load a cell: open a notebook editor, File > Import Notebook, and inject an `.ipynb` file object into `input[type=file]` with JavaScript; set `window.confirm = () => true` first. The notebook's `metadata.kaggle` block sets accelerator (`nvidiaTeslaT4` = T4 x2), internet and `dataSources`, so inputs attach without clicking. Ids in use: bundle dataset version `21857658`; kernel versions `367062493` (px1-mmflood-final), `366960729` (px1-adaptive-physics), `367828676` (notebookd6a6e506fc).
- Save: click `button[aria-label="Save Version"]`, then the dialog's "Save".
- Outputs cannot be downloaded from the browser. Results are printed to the log (`REV_JSON` lines), read from the log page, gzip+base64-encoded in the page and returned as text.
- Quota: `POST /api/i/kernels.KernelsService/GetAcceleratorQuotaStatistics` from the page.
- About 28 empty draft notebooks exist from an earlier Kaggle outage; they can be reused.

## 8. Rules the user has set
- Never handle the Kaggle password or API keys; never create accounts for the user (OSF, Zenodo are done by the user).
- Files on the PC only under `F:\Coding\Research`; delete nothing without permission; no GPU work on the PC.
- Git commits use the author `Mahim25800 <Mahim25800@users.noreply.github.com>`; the user's email is private.
- Sabbir Hossain is not an author. CRediT roles and the AI-use statement follow the group's earlier (JVCI) paper and are already in the manuscript.
- Report results as they are, including negative ones; hypotheses are fixed before data are seen.

## 9. Known weak points (be upfront about them)
- MMFlood accuracy stays near 0.2 IoU; much of its labelled water is not dark in SAR.
- Source-only calibration held on Kuro Siwo and failed on MMFlood; no label-free way to tell the cases apart yet.
- Base models are small; no foundation model yet.
- The first pre-registration (MMFlood) was never publicly timestamped; this cannot be repaired and the paper says so.
- Kuro Siwo is a sample (33 of 43 events) and served as the development set.
- On Kuro Siwo the method is level with, not better than, supervised fine-tuning, and a model-free dark threshold is close on flood water.
- EATA, SAR and CoTTA have so far been run on Kuro Siwo only.
- GEOID-Flood's held-out set has only 15 activations, so moderate effects may not reach significance.
