# Pre-registration 2 (DRAFT, not yet deposited)

**Project:** Physics-checked test-time adaptation for SAR flood mapping with flood-level miss-rate calibration
**Status:** draft of 10 Nov 2026. To be deposited on OSF (osf.io/prereg) by the authors **before any file of the test dataset is downloaded**. Items marked [CHECK] must be settled before depositing.

## 1. Why a second pre-registration
The first confirmatory test (MMFlood, 30 activations) was written down before the data were downloaded but was not deposited in a public registry. Its result was mixed. This second test is deposited publicly and targets the claims the revised paper rests on.

## 2. What is frozen
- **Code:** github.com/Mahim25800/Research-Flood at the commit recorded on the day of deposit [CHECK: paste commit hash].
- **Models:** the three U-Nets trained on the Sen1Floods11 training split (seeds 0, 1, 2), unchanged.
- **Proposed method:** AdaBN followed by physics-checked self-training with the adaptive rule (`adabn_physa`), all settings as in the paper (caps -15/-22 dB, HAND < 5 m, slope < 3 deg, 200 steps, confidence 0.9/0.1).
- **Compared methods:** no adaptation, TENT, AdaBN (and EATA, SAR, CoTTA as secondary).
- **Calibration:** mean rule (conformal risk control) at a 20% miss-rate target, (a) on the 11 Sen1Floods11 floods only ("source-only"), (b) leave one event out within the test dataset ("same-dataset"). Per-flood rule at alpha = 0.2, delta = 0.2 as secondary.

## 3. Test data
- **Primary:** the held-out tree of GEOID-Flood v1.1 (Hugging Face `links-ads/geoid-flood`, folder `geoid-flood-heldout`, CC BY 4.0): Copernicus EMS activations EMSR857 to EMSR871, 1,301 tiles of 1024 x 1024 pixels, Sentinel-1 GRD VV and VH at 10 m, three-class labels (background, permanent water, flooded water). These activations are from 2026 and post-date every dataset used so far (Sen1Floods11, Kuro Siwo, MMFlood), so no event can overlap. Only the dataset's documentation has been read; no data file has been downloaded.
- **Fallback if the primary cannot be obtained:** FloodPlanet (19 events, Sentinel-1 with labels drawn on PlanetScope images), after excluding events within 300 km and one year of any Sen1Floods11, Kuro Siwo or MMFlood event.
- **Preprocessing (fixed):** backscatter in dB (converted if the dataset card states linear power); tiles cut into 512 x 512 chips with at least 50% valid radar, at most 48 chips per event chosen at evenly spaced positions; HAND from the GLO-30 HAND tiles. Permanent water is scored as the dataset labels it; a second score ignores permanent water.
- **Unit:** the EMS activation (all its areas of interest merged). Activations with fewer than 8 chips are dropped. Values are averaged over the three models per event.

## 4. Hypotheses (confirmatory)
Tests are over events. Unless stated otherwise: one-sided Wilcoxon signed-rank, Holm-corrected across H1-H6, alpha = 0.05.
- **H1, H2 (better than other adaptation methods at a matched miss rate).** With same-dataset calibration at the 20% target, the proposed method's IoU is higher than TENT's (H1) and AdaBN's (H2). *Held on Kuro Siwo and on MMFlood.*
- **H3 (does no harm).** With same-dataset calibration at the 20% target, the proposed method's IoU is not lower than the unadapted model's by more than 0.01 (non-inferiority: one-sided Wilcoxon on the per-event difference plus 0.01). *Held on Kuro Siwo and on MMFlood.*
- **H4 (adaptation moves a source-calibrated threshold towards validity).** With calibration on Sen1Floods11 only at the 20% target, the proposed method's per-event miss rate is lower than the unadapted model's. *Held clearly on Kuro Siwo; on MMFlood the mean was lower (0.26 against 0.28) but this was not tested.*
- **H5 (per-flood rule).** With same-dataset calibration, alpha = 0.2 and delta = 0.2, the share of events that miss at most 20% of their water is not significantly below 0.80 (exact one-sided binomial test). *Held on all three datasets.*
- **H6 (where the method helps).** Among events in which open water is visible to the radar, the proposed method's IoU at a matched miss rate (same-dataset calibration, 20% target) is higher than the unadapted model's. "Visible" is decided without labels, from the event's own backscatter, by a rule that will be fixed on MMFlood and written here before deposit. [CHECK: rule and threshold]

## 5. Secondary, reported whatever the outcome
- Whether the proposed method's mean miss rate with source-only calibration is at or below 0.20 (it was on Kuro Siwo and was not on MMFlood).
- IoU of the proposed method against no adaptation over all events at a matched miss rate (superiority).
- IoU at threshold 0.5; precision and false-alarm rate; worst per-event miss rate.
- EATA, SAR and CoTTA under the same rules.
- The same analysis with a Sentinel-1-capable foundation model fine-tuned on Sen1Floods11 (planned: TerraMind [CHECK]), where AdaBN is replaced by adapting the layer-norm parameters. Pre-specified but exploratory, because the adaptation recipe for that backbone is new.

## 6. What counts as support
Each hypothesis is reported as supported or not supported. No hypothesis is added, dropped or reworded after the data are seen, and any deviation from this text is listed in the paper. H1-H3 and H5 test whether earlier results repeat on new floods; H4 and H6 test the two new claims.

## 7. Stopping and exclusions
No interim looks. A run that crashes is repeated with the same seed. Events are excluded only by the rules in Section 3.
