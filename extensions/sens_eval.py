"""
Sensitivity of the proposed method (AdaBN + adaptive physics-checked self-training) to its settings, on Kuro Siwo.

One setting is changed at a time from the defaults fixed before the MMFlood test:
  cap   : dark-threshold caps VV/VH (default -15/-22 dB)          -> -13/-20 and -17/-24
  hand  : HAND limit for the add rule (default 5 m)               -> 3 and 10
  steps : self-training steps (default 200)                       -> 100 and 400 (passed as --steps)
  conf  : teacher confidence thresholds P_HI/P_LO (0.9/0.1)       -> 0.8/0.2 and 0.95/0.05

The module constants of tta_px1 are overridden before ks_eval runs, so the code path is exactly the one used for
the reported results. Usage: python sens_eval.py --config hand3 --model-seed 0 --out runs_sens/hand3 [ks_eval args]
"""
import sys

import tta_px1

CONFIGS = {
    "cap_m2": {"VV_CAP": -17.0, "VH_CAP": -24.0},
    "cap_p2": {"VV_CAP": -13.0, "VH_CAP": -20.0},
    "hand3": {"HAND_LOW": 3.0},
    "hand10": {"HAND_LOW": 10.0},
    "steps100": {},
    "steps400": {},
    "conf80": {"P_HI": 0.8, "P_LO": 0.2},
    "conf95": {"P_HI": 0.95, "P_LO": 0.05},
}

if __name__ == "__main__":
    i = sys.argv.index("--config")
    cfg = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    for k, v in CONFIGS[cfg].items():
        assert hasattr(tta_px1, k), k
        setattr(tta_px1, k, v)
    print(f"sensitivity config {cfg}: {CONFIGS[cfg]} (HAND_LOW {tta_px1.HAND_LOW}, caps {tta_px1.VV_CAP}/{tta_px1.VH_CAP}, "
          f"P_HI/P_LO {tta_px1.P_HI}/{tta_px1.P_LO})", flush=True)
    import runpy
    sys.argv[0] = "ks_eval.py"
    runpy.run_path("ks_eval.py", run_name="__main__")
