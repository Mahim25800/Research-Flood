# Revision experiments

**12 November validity audit:** read `BASELINE_FIDELITY_AUDIT.md` before any new
GPU run. Historical EATA/SAR/CoTTA results are custom variants. TENT-B is a new,
separately keyed diagnostic; build its notebook with `python revision/build_tent_audit.py`
from the repository root. CPU checks: `python revision/test_validity.py`.
The revised manuscript source has changed; the tracked `main_wip.pdf` is the older draft.
Do not interpret it as a rendered version of the current source until rebuilt.

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
