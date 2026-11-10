# Pre-registration 2, Amendment 1

**Registration:** https://osf.io/c38jx (registered 10 Nov 2026)
**Date of this amendment:** 10 Nov 2026
**Status of the test data when this amendment was written:** no file of GEOID-Flood or FloodPlanet has been downloaded or opened. Only the dataset documentation has been read. The amendment is filed before any download.

The registered text contained four items marked [CHECK]. This amendment settles them. Nothing else in the registered text changes.

## 1. Code version (Section 2, first item)
The frozen code is github.com/Mahim25800/Research-Flood at commit `836d225c1d3de814144fd556129dead2ec542d3e`. Scripts written later only to read and cut the new dataset into chips (preparation and evaluation wrappers) will be added in later commits and listed in the paper. The models, the adaptation methods, their settings and the analysis script (`revision/rev_analyse.py`) are those of this commit.

## 2. Hypothesis H6 is withdrawn as a confirmatory hypothesis (Section 4)
The registered text said that the rule deciding in which events "open water is visible to the radar" would be fixed on MMFlood and written down before deposit. We tried to do so, using data already in hand (Kuro Siwo and MMFlood) and a label-free quantity: the share of an event's pixels on the dark side of its own Otsu split of VV backscatter.

This quantity did not separate events in which the proposed method helps from events in which it does not. On Kuro Siwo the gain over the unadapted model was positive at every dark share (Spearman correlation 0.28, p = 0.16, 26 events). On MMFlood the correlation was weak and negative (-0.39, p = 0.03, 30 activations) with no clear cut-off. We therefore have no rule that we could justify in advance.

H6 is withdrawn. The confirmatory hypotheses are **H1 to H5** as registered, and the Holm correction is applied across these five. The relation between the gain of the proposed method and the dark share is moved to Section 5 (secondary, reported whatever the outcome).

## 3. Foundation-model analysis (Section 5, last item)
The planned model is TerraMind (IBM/ESA, base size, run with TerraTorch) with its Sentinel-1 GRD input, fine-tuned on the Sen1Floods11 training split. If it cannot be run within the available compute, Prithvi-EO-2.0 is used instead; if neither can be run, this is stated in the paper. This analysis stays exploratory, as registered, and no confirmatory hypothesis depends on it.

## 4. Consequence for Section 6
The sentence "H1-H3 and H5 test whether earlier results repeat on new floods; H4 and H6 test the two new claims" now reads: "H1-H3 and H5 test whether earlier results repeat on new floods; H4 tests the new claim."
