# Prospective SAR backbone replication

Prepared before access to any ImpactMesh raster, mask or prediction. This is a local protocol draft. No registration has been submitted.

## Materials and sampling

The SSL4EO-S12 Sentinel-1 MoCo ResNet-50 encoder uses TorchGeo revision 78659dc90b7b27c97bd381380f8dcbb834c64359. The encoder SHA-256 is 906e4356da0ff06ce0388cba50dd4f310ba6abbee706c5303c129d8aead98082. A 64-channel feature pyramid and binary segmentation head are initialized on the source task. Only VV and VH enter the network.

ImpactMesh-Flood revision 4432601ff08e6fb256f4ac3a59d9d30c1f069774 supplies the fresh target. The three published split lists are an inventory. No ImpactMesh split trains the source model or selects its checkpoint. The screening window was EMSR780–856 and activation year 2025, with official Copernicus flood category. Require eight unique listed chips, sort chip IDs and take up to 48 evenly spaced indices per activation. selected.json gives the exact 516 IDs across 13 activations. This is a project-specific external cohort. The publisher's official event-holdout subset is much smaller. All event dates postdate the prior Kuro Siwo and Sen1Floods11 events and precede the prior 2026 confirmatory events. No selected ID occurs in the prior MMFlood activation list. Recurring geography is allowed. Inspect shared weather episodes using metadata before registration and report any dependence.

## Model and comparison

From this directory run python train_source.py --seed N separately for N=0,1,2. Train on Sen1Floods11 only. Use source-training channel normalization, source-validation selection, AdamW, learning rate 0.0001, batch four and 60 epochs. Hash the three selected segmentation checkpoints and record preprocessing, environment and source normalization before target execution. Start each arm and activation from its same-seed source checkpoint. Use the same AdaBN, crops and augmentations, 200 updates, batch eight, Adam at 0.0001 and EMA 0.99. Only the original validity-corrected physical pseudo-label rule changes. Use experiment.paired_predictions after constructing the event inputs. No target labels enter adaptation.

Use the event-time VV/VH RTC observation. Verify channel order, units, fill values and acquisition time from publisher metadata. Apply the dB conversion only to linear power. Align DEM and independent GLO-30 HAND to the SAR grid with recorded transforms and support masks. Compute physical slope in degrees with pixel spacing in metres. Supply valid radar, HAND and slope masks. Unknown terrain disables that cue. Verify the publisher's label codes and nodata with its loader before evaluation. Use no-water as negative and floodwater as positive, ignoring permanent water and nodata. Store per-chip probabilities losslessly in float32. Predeclare and document any input incompatibility before scoring. Report exclusions and preserve their IDs.

## Endpoint and interpretation

Primary endpoint is exact floodwater-only AP. Pool pixels within each activation and seed. Average the three paired differences within activation and weight activations equally. Report all differences, wins, losses, ties, undefined events and a one-sided exact sign test at 0.05. With all 13 events defined and no ties, 10 wins are needed. The sign test concerns direction. Quantify magnitude with mean and median paired differences. For secondary map outcomes fix one common threshold using source-validation data only before target access and report the actual IoU, miss rate, FPR and false discovery proportion for each arm. Record cue coverage and conflict counts. Any demonstrated effect applies jointly to the backbone and RTC domain used here. An isolated backbone effect requires a scratch-backbone comparison on the same events.

## Status and commands

python experiment.py runs a CPU smoke test using the downloaded, digest-verified encoder, a decoder backward pass and both adaptation arms on synthetic data. python paired_test.py --out NEW.json regenerates the chip roster. python paired_test.py --predictions RUN --out NEW.json computes the primary paired analysis. RUN/labels contains per-chip label arrays. RUN/EVENT/SEED/ARM contains per-chip score arrays with the same filenames. The real-data ingestion and alignment adapter, source-trained checkpoints, and threshold are pending. Treat the smoke test as implementation evidence only.
