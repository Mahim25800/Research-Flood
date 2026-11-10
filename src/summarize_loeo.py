"""Leave-one-event-out summary: radar only (V0) vs radar + elevation (V1), paired by held-out flood.

Writes runs_loeo/summary.json and runs_loeo/summary.md.
Paired test: sign test and Wilcoxon signed-rank (if scipy is available) on per-event IoU differences.
"""
import glob
import json
import math
import os

runs = {}
for p in glob.glob(os.path.join("runs_loeo", "*", "results.json")):
    r = json.load(open(p))
    runs[(r["variant"], r["holdout"])] = r

events = sorted({h for _, h in runs})
rows, diffs, diffs_clean = [], [], []
pool = {v: {"all": [0, 0], "clean": [0, 0]} for v in ["V0", "V1"]}
for e in events:
    a, b = runs.get(("V0", e)), runs.get(("V1", e))
    if not a or not b:
        rows.append({"event": e, "status": "incomplete"})
        continue
    row = {"event": e, "chips": a["holdout_chips"],
           "v0_all": a["holdout_iou"]["all"], "v1_all": b["holdout_iou"]["all"],
           "v0_clean": a["holdout_iou"].get("clean"), "v1_clean": b["holdout_iou"].get("clean")}
    row["diff_all"] = round(row["v1_all"] - row["v0_all"], 4)
    diffs.append(row["diff_all"])
    if row["v0_clean"] is not None and row["v1_clean"] is not None:
        row["diff_clean"] = round(row["v1_clean"] - row["v0_clean"], 4)
        diffs_clean.append(row["diff_clean"])
    rows.append(row)


def sign_test(d):
    pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
    n = pos + neg
    if n == 0:
        return {"wins": pos, "losses": neg, "p_two_sided": None}
    k = min(pos, neg)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"wins": pos, "losses": neg, "p_two_sided": round(p, 4)}


def wilcoxon(d):
    try:
        from scipy.stats import wilcoxon as w
        return round(float(w(d).pvalue), 4) if len(d) >= 5 else None
    except Exception:
        return None


summary = {
    "events": rows,
    "mean_diff_all": round(sum(diffs) / len(diffs), 4) if diffs else None,
    "median_diff_all": round(sorted(diffs)[len(diffs) // 2], 4) if diffs else None,
    "sign_test_all": sign_test(diffs), "wilcoxon_p_all": wilcoxon(diffs),
    "mean_diff_clean": round(sum(diffs_clean) / len(diffs_clean), 4) if diffs_clean else None,
    "sign_test_clean": sign_test(diffs_clean), "wilcoxon_p_clean": wilcoxon(diffs_clean),
    "n_events_complete": len(diffs),
}
os.makedirs("runs_loeo", exist_ok=True)
with open(os.path.join("runs_loeo", "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

f3 = lambda x: "-" if x is None else f"{x:.3f}"
lines = ["| Held-out flood | Chips | Radar only | + elevation | Change | Change, clean |", "| --- | --- | --- | --- | --- | --- |"]
for r in sorted([r for r in rows if "diff_all" in r], key=lambda r: -r["diff_all"]):
    lines.append(f"| {r['event']} | {r['chips']} | {f3(r['v0_all'])} | {f3(r['v1_all'])} | {r['diff_all']:+.3f} | "
                 f"{'-' if r.get('diff_clean') is None else format(r['diff_clean'], '+.3f')} |")
lines.append("")
lines.append(f"Mean change {summary['mean_diff_all']}, wins/losses {summary['sign_test_all']['wins']}/{summary['sign_test_all']['losses']}, "
             f"sign-test p {summary['sign_test_all']['p_two_sided']}, Wilcoxon p {summary['wilcoxon_p_all']}")
with open(os.path.join("runs_loeo", "summary.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print("LOEO_SUMMARY_DONE")
