# PX1 manuscript (LaTeX, Elsevier CAS single-column template)

Same template as the group's earlier Elsevier submission (cas-sc), same author order and affiliation.

- `main.tex`: complete manuscript (18 pages: 6 figures, 8 tables). No open items remain.
- `main.pdf`: compiled manuscript.
- `references.bib`: the 90 checked references (same keys as literature_list.md).
- `highlights.txt`: highlights for the separate upload (each under 85 characters).
- `figures/`:
  - `fig_method.pdf` (Fig. 1, method overview; source `fig_method.tex`, TikZ)
  - `fig_sites.pdf` (Fig. 2, study sites; script `figure_scripts/paper_map.py`)
  - `fig_guarantee.pdf` (Fig. 3, IoU and realised miss rate vs target; `figure_scripts/paper_figs.py`)
  - `fig_ks_events.pdf` (Fig. 4, Kuro Siwo per event; `paper_figs.py`)
  - `fig_maps.pdf` (Fig. 5, example maps on Kuro Siwo; `figure_scripts/paper_fig_maps.py` from `maps_rows_compact.json`, produced by maps.py on Kaggle)
  - `fig_mmflood_water.pdf` (Fig. 6, water share and MMFlood per-activation IoU; `paper_fig_mm.py`)
  - `graphical_abstract.pdf/.png` (separate upload; source `graphical_abstract.tex`, bars `ga_bars.pdf`)
  - `paper_numbers.json`, `guarantee_curves.json`, `water_fraction.json`: every number the figures and Table 4 use.
- `cas-sc.cls`, `cas-common.sty`, `cas-model2-names.bst`, `thumbnails/`: Elsevier CAS class files, so it compiles anywhere.

All figures are computed from stored results only (no model runs): the probability histograms in
runs_ks/ and runs_loeo/, and the MMFlood per-activation table and pixel counts read from the Kaggle logs
(px1-mmflood-final, px1-mmflood-activation), saved in figure_scripts/.

Compile: `latexmk -pdf main.tex` (or upload the folder to Overleaf; Overleaf also has the CAS class built in).
