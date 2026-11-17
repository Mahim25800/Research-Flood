# Raster grid compatibility correction

Kaggle run version 1 verified the published v2 protocol, the three checkpoints,
the catalog and all seven shards. It stopped before scoring any activation:
the evaluator compared its affine pixel scales to exactly +10 and −10 metres.

A bounded radar shard probe inspected ten pinned, post-event radar GeoTIFFs.
Seven failed that exact check despite having 1024 × 1024 pixels, a projected
metric CRS and zero rotation. A concrete example is
`EMSR857-1-10_s1grd_post_20260123T031726.tif`, EPSG:32736, with transform
`[10.000000000000512, 0.0, 501759.9999999997, 0.0, -10.0, 7301120.0]`
in Rasterio affine order. That example is a rejected sample, not necessarily
the first rejected tile in the evaluator's sorted order. The rerun saves a
grid audit that identifies the first failure of the original check.

Protocol ZIP v3 allows a 0.000001 m absolute tolerance on the two nominal
pixel-scale coefficients, with zero relative tolerance. The shape, projected
CRS, zero-rotation and companion-raster alignment checks are unchanged. No
rasters are resampled or rounded. The catalog, activation eligibility, chip
selection, methods, checkpoints, and analysis stay identical to v2.

The revised check passed the ten inspected headers and rejected simulated
out-of-tolerance scale, rotation, shape and geographic CRS cases. All archive
members except the grid check in the evaluator are byte-identical to v2.
This compatibility correction was made before examining held-out outcomes.
