"""
LaTeX tables of the revised paper, written from the analysis JSON of rev_analyse.py into paper_latex/tables/*.tex.

Usage: python make_tables.py results_rev/rev_analysis.json
Summary entries: [mean IoU, mean miss rate, worst miss rate, floods meeting, floods, mean threshold, precision, false alarms].
"""
import json
import os
import sys

R = json.load(open(sys.argv[1]))
S = R["summary"]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "revision", "paper_wip", "tables")
os.makedirs(OUT, exist_ok=True)
NAME = {"none": "No adaptation", "st": r"Self-training \citep{chen2026cimttda}", "tent": "TENT-R (running statistics)", "tent_batch": "TENT-B (batch statistics)", "eata": "EATA-inspired",
        "sar": "SAR-inspired", "cotta": "CoTTA-inspired", "adabn": "AdaBN", "phys": "Physics-checked self-training, fixed rule",
        "physa": "Physics-checked self-training, adaptive rule", "adabn_phys": "AdaBN + physics check, fixed rule",
        "adabn_physa": "AdaBN + physics check, adaptive rule (proposed)", "ft": "Fine-tuned on labelled target floods (supervised)"}
SHORT = dict(NAME, adabn_physa="Proposed", st="Self-training", ft="Fine-tuned (supervised)", physa="Physics check only")


def f3(x):
    return f"{x:.3f}"


def best(vals, i, fmt=f3, lo=False):
    v = [x for x in vals if x is not None]
    b = (min(v) if lo else max(v)) if v else None
    return "--" if vals[i] is None else (r"\textbf{" + fmt(vals[i]) + "}" if abs(vals[i] - b) < 5e-4 else fmt(vals[i]))


def get(D, m, rule, j=0):
    e = S.get(D, {}).get(m, {}).get(rule)
    return None if e is None or len(e) <= j else e[j]


def write(name, lines):
    open(os.path.join(OUT, name + ".tex"), "w").write("\n".join(lines) + "\n")
    print(name, len(lines))


def iou_table(D, methods, name, extra_rows=()):
    methods = [m for m in methods if m in S[D]]
    cols = ["t0.5", "loeo_0.1", "loeo_0.2", "loeo_0.3", "q_0.2_0.2"]
    L = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{Mean rule, target $\alpha$} & \multicolumn{2}{c}{Per-flood rule, $\alpha=\delta=0.2$} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
         r"Method & 0.5 & 10\% & 20\% & 30\% & IoU & Floods meeting \\", r"\midrule"]
    sup = [m for m in methods if m == "ft"]
    main = [m for m in methods if m != "ft"]
    for m in main:
        cells = [best([get(D, x, c) for x in main], main.index(m)) for c in cols]
        meet = get(D, m, "q_0.2_0.2", 3); n = get(D, m, "q_0.2_0.2", 4)
        L.append(f"{NAME[m]} & " + " & ".join(cells) + f" & {meet:.1f}/{n} \\\\")
    for m in sup:
        L.append(r"\midrule")
        meet = get(D, m, "q_0.2_0.2", 3); n = get(D, m, "q_0.2_0.2", 4)
        L.append(f"{NAME[m]} & " + " & ".join(f3(get(D, m, c)) for c in cols) + f" & {meet:.1f}/{n} \\\\")
    for r in extra_rows:
        L.append(r)
    L += [r"\bottomrule", r"\end{tabular}"]
    write(name, L)


ALLM = ["none", "st", "tent", "tent_batch", "eata", "sar", "cotta", "adabn", "physa", "adabn_phys", "adabn_physa", "ft"]
iou_table("S1F", ["none", "st", "tent", "eata", "sar", "cotta", "adabn", "phys", "physa", "adabn_phys", "adabn_physa"], "tab_s1f")
iou_table("KS", ALLM, "tab_ks", [r"\midrule", r"Otsu threshold on VV (no model) & 0.456 & -- & -- & -- & -- & -- \\"])
iou_table("MM", ALLM, "tab_mm", [r"\midrule", r"Otsu threshold on VV (no model) & 0.090 & -- & -- & -- & -- & -- \\"])
if "KSf" in S:
    iou_table("KSf", ALLM, "tab_ksf")

# calibration source: same dataset vs Sen1Floods11 only
L = [r"\begin{tabular}{llcccccccc}", r"\toprule",
     r" & & \multicolumn{4}{c}{Same-dataset calibration} & \multicolumn{4}{c}{Source-only calibration (Sen1Floods11)} \\",
     r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
     r"Dataset & Method & IoU & Miss & Worst & Meeting & IoU & Miss & Worst & Meeting \\", r"\midrule"]
for D, label in [("KS", "Kuro Siwo"), ("MM", "MMFlood")]:
    ms = [m for m in ["none", "tent", "eata", "sar", "cotta", "adabn", "physa", "adabn_physa"] if m in S[D] and "x_S1F_0.2" in S[D][m]]
    for i, m in enumerate(ms):
        c = []
        for rule in ["loeo_0.2", "x_S1F_0.2"]:
            e = S[D][m][rule]
            miss = f3(e[1]) if e[1] <= 0.2 else r"\textit{" + f3(e[1]) + "}"
            c += [f3(e[0]), miss, f"{e[2]:.2f}", f"{e[3]:.1f}/{e[4]}"]
        L.append((label if i == 0 else "") + f" & {SHORT[m]} & " + " & ".join(c) + r" \\")
    if D == "KS":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_transfer", L)

# per-flood rule on the three datasets
L = [r"\begin{tabular}{llcccccc}", r"\toprule",
     r" & & \multicolumn{3}{c}{$\delta=0.2$ (at least 80\% of floods)} & \multicolumn{3}{c}{$\delta=0.1$ (at least 90\% of floods)} \\",
     r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
     r"Dataset & Method & IoU & Meeting & Worst & IoU & Meeting & Worst \\", r"\midrule"]
for D, label in [("S1F", "Sen1Floods11"), ("KS", "Kuro Siwo"), ("MM", "MMFlood")]:
    ms = [m for m in ["none", "tent", "adabn", "adabn_physa"] if m in S[D]]
    for i, m in enumerate(ms):
        c = []
        for rule in ["q_0.2_0.2", "q_0.2_0.1"]:
            e = S[D][m][rule]
            c += [best([S[D][x][rule][0] for x in ms], i), f"{e[3]:.1f}/{e[4]} ({100 * e[3] / e[4]:.0f}\\%)", f"{e[2]:.2f}"]
        L.append((label if i == 0 else "") + f" & {SHORT[m]} & " + " & ".join(c) + r" \\")
    if D != "MM":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
write("tab_perflood", L)

# precision and false alarms on Kuro Siwo (adaptive rule)
if len(S["KS"]["none"]["t0.5"]) >= 8:
    ms = [m for m in ["none", "st", "tent", "eata", "sar", "cotta", "adabn", "physa", "adabn_physa"] if m in S["KS"]]
    L = [r"\begin{tabular}{lcccccc}", r"\toprule", r" & \multicolumn{3}{c}{Threshold 0.5} & \multicolumn{3}{c}{Mean rule, 20\% target} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}", r"Method & Miss rate & Precision & False alarms & Miss rate & Precision & False alarms \\", r"\midrule"]
    for m in ms:
        c = []
        for rule in ["t0.5", "loeo_0.2"]:
            e = S["KS"][m][rule]
            c += [f3(e[1]), f3(e[6]), f"{100 * e[7]:.1f}\\%"]
        L.append(f"{SHORT[m]} & " + " & ".join(c) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_prec", L)

# paired tests used in the text
T = R["tests"]
for D in T:
    for rule, rows in T[D].items():
        for a, b, d, w, n, p, ph in rows:
            print(f"TEST {D} {rule}: {a} vs {b}: {d:+.3f}, {w}/{n}, p={p:.2g}, holm={ph:.2g}")
