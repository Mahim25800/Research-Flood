# PX1 manuscript and supplementary material

**Working draft, 25 November 2026.** *When do dark-water pseudo-labels help test-time adaptation in SAR flood mapping?*

- [Main PDF](main.pdf) and [LaTeX source](main.tex)
- [Supplement PDF](supplement.pdf) and [LaTeX source](supplement.tex)
- [Highlights](highlights.txt)
- [Results index](../results/README.md) and [reproduction guide](../revision/reproduction/px1_external_20261125/README.md)

The current paper leads with the two-backbone, two-cohort AP comparison. It retains the preregistered GEOID-Flood U-Net contrast against same-rule post-processing and labels the ResNet-50 crossover and relaxed teacher as post hoc. It reports the ImpactMesh matched-FPR STANet comparison and acquisition-timing sensitivity. Supplement Section S3 contains the complete GEOID-Flood ResNet-50 readback across six arms.

## Build the PDFs

Run from this directory with a LaTeX distribution that provides `latexmk` and the packages used by the Elsevier CAS template.

```bash
latexmk -pdf main.tex supplement.tex
```

The current sources use the included bibliography, figures, tables and CAS template files. Older figure assets remain in this directory for historical source revisions. The included PDFs are local builds of the working draft.
