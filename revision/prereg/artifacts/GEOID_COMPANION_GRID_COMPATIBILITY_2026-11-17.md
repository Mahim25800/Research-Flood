# GEOID-Flood companion raster grid correction

Kaggle notebook version 2 verified the pinned inputs, passed radar preparation,
and selected 424 chips across all nine catalog activations. It stopped before
scoring on exact radar/DEM transform equality for EMSR857-1-1. Its failure logs,
grid audit and chip selection are retained.

The exact radar and DEM are both 1024 × 1024, EPSG:32736. In Rasterio affine order:

- Radar: `[10.000000000000284, 0, 481279.9999999997, 0, -10.00000000000091, 7301120.000000001]`
- DEM: `[10, 0, 481280, 0, -10, 7301120]`

The maximum absolute coordinate difference over their four raster corners is
9.313225746154785e−10 m. The companion label matched the radar transform exactly.
Across the 1,301 tiles, 1,193 DEM transforms differed under exact comparison.
That count alone does not establish the size of the other differences.

Protocol ZIP v4 keeps exact CRS and dimension matching and uses a 1e−6 m
absolute tolerance per corner coordinate, with zero relative tolerance.
The exact rejected pair passes, while different shapes, CRSs, a 0.1 m offset,
and a small scale discrepancy that accumulates past the corner tolerance fail.
Actual misalignment still stops preparation. No pixels are moved or resampled.
The only changed archive member is the evaluator. The release, eligibility,
selection, checkpoint, adaptation and scoring definitions are unchanged.
No held-out model scores were available when this correction was made.
