# Revision experiments

Added after the first full draft, to test the claims more strictly.

| File | Purpose |
| --- | --- |
| `rev_patch.py` | Adds EATA, SAR and CoTTA as test-time adaptation baselines, and a flood-water-only score for Kuro Siwo (permanent water ignored) |
| `ks_finetune.py` | Supervised reference: fine-tunes on the labelled Kuro Siwo floods that the guarantee calibration uses (5 folds by event) |
| `rev_analyse.py` | Analysis from stored histograms: calibration on Sen1Floods11 only (no target labels), per-flood guarantee, paired tests |
| `rev_gpu.py`, `rev_cpu.py` | Kaggle runners |

Kaggle notebook cell (inputs: the PX1 bundle dataset and the outputs of the earlier notebooks):

```python
import os, urllib.request
SHA, ST = "<commit>", "/tmp/px1_stage"
os.makedirs(ST, exist_ok=True)
for f in ["rev_analyse.py", "rev_patch.py", "r34_patch.py", "ks_finetune.py", "rev_gpu.py", "rev_cpu.py"]:
    urllib.request.urlretrieve(f"https://raw.githubusercontent.com/Mahim25800/Research-Flood/{SHA}/revision/{f}", f"{ST}/{f}")
exec(open(f"{ST}/rev_gpu.py").read())   # or rev_cpu.py
```
