"""Verify all returned score hashes, rank statistics, and paired support."""
import json
import math
from pathlib import Path
import numpy as np
from frozen_run import DEST, immutable_files
from paired_test import ARMS, roster
from verify_inputs import sha

def verify():
    saved=json.loads((DEST/"results.json").read_text())
    assert immutable_files()==json.loads((DEST/"freeze.json").read_text())["files"]
    diffs=[]
    for event,chips in roster().items():
        expected={str(Path(a)/(c+".npy")) for a in ARMS for c in chips}
        for seed in range(3):
            work=DEST/event/str(seed)
            manifest=json.loads((work/"scores.json").read_text())
            assert set(manifest)==expected and all(sha(work/n)==h for n,h in manifest.items())
            counts=[saved["counts"][event][a][seed] for a in ARMS]
            assert len({(v["tp"]+v["fn"],v["fp"]+v["tn"]) for v in counts})==1
        rows=saved["primary"]["events"][event]["seeds"]
        if rows:
            delta=sum(r[ARMS[1]]-r[ARMS[0]] for r in rows)/len(rows)
            assert len(rows)==3 and abs(delta-saved["primary"]["events"][event]["delta"])<1e-12
            diffs.append(delta)
    wins=sum(x>0 for x in diffs)
    n=sum(x!=0 for x in diffs)
    p=sum(math.comb(n,k) for k in range(wins,n+1))/2**n if n else None
    assert p==saved["primary"]["sign_p"] and np.isclose(np.mean(diffs),saved["primary"]["mean_delta"])
    return dict(score_hashes=sum(len(chips)*6 for chips in roster().values()),defined_events=len(diffs),
                wins=wins,median_delta=float(np.median(diffs)),exact_p=p)
if __name__=="__main__":
    result=verify()
    with (DEST/"verification.json").open("x") as f:
        json.dump(result,f,indent=2)
    print(result)
