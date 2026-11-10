import csv
import re

exec(open("refs.py").read())

ORDER = sorted(CATS)


def plain(s):
    """BibTeX-escaped text -> readable text for Markdown/CSV."""
    rep = {"$^2$": "²", "{\\\"a}": "ä", "{\\'e}": "é", "{\\'o}": "ó", "{\\~a}": "ã", "{\\`e}": "è", "{\\v{s}}": "š", "\\&": "&", "---": "—", "--": "–"}
    for k, v in rep.items():
        s = s.replace(k, v)
    return s.replace("{", "").replace("}", "")


def short_authors(a):
    names = [n.strip() for n in plain(a).split(" and ")]
    last = [n.split(",")[0] for n in names]
    if "others" in last:
        last = [x for x in last if x != "others"]
        return f"{last[0]} et al."
    if len(last) == 1:
        return last[0]
    if len(last) == 2:
        return f"{last[0]} & {last[1]}"
    return f"{last[0]} et al."


def link(r):
    return f"https://doi.org/{r['doi']}" if r["doi"] else r["url"]


# ---------------- BibTeX
FIELD = {"article": "journal", "inproceedings": "booktitle", "misc": "howpublished", "book": "publisher", "manual": "organization"}
out = []
for r in R:
    f = [f"  author = {{{r['authors']}}}", f"  title = {{{{{r['title']}}}}}", f"  year = {{{r['year']}}}",
         f"  {FIELD[r['type']]} = {{{r['venue']}}}"]
    d = r["details"]
    m = re.search(r"vol\. ([\w]+)", d)
    if m:
        f.append(f"  volume = {{{m.group(1)}}}")
    m = re.search(r"no\. ([\w–\-]+)", d)
    if m:
        f.append(f"  number = {{{m.group(1)}}}")
    m = re.search(r"pp\. ([\d\-]+)", d)
    if m:
        f.append(f"  pages = {{{m.group(1)}}}")
    m = re.search(r"art\. ([\w]+)", d)
    if m:
        f.append(f"  note = {{Article {m.group(1)}}}")
    if r["type"] == "book" and d:
        f.append(f"  edition = {{{d.split()[0]}}}")
    if r["doi"]:
        f.append(f"  doi = {{{r['doi']}}}")
    f.append(f"  url = {{{r['url']}}}")
    out.append(f"@{r['type']}{{{r['key']},\n" + ",\n".join(f) + "\n}")
open("references.bib", "w").write("% PX1 paper references (" + str(len(R)) + " entries), checked online 9-10 Nov 2026\n\n" + "\n\n".join(out) + "\n")

# ---------------- CSV
with open("references.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["#", "Category", "Cite key", "Authors", "Year", "Title", "Venue", "Details", "DOI", "Link", "What it is", "How we use it"])
    i = 0
    for c in ORDER:
        for r in [r for r in R if r["cat"] == c]:
            i += 1
            w.writerow([i, CATS[c], r["key"], plain(r["authors"]), r["year"], plain(r["title"]), plain(r["venue"]), plain(r["details"]),
                        r["doi"], link(r), r["summary"], r["use"]])

# ---------------- Markdown
L = ["# PX1 paper: literature list", "",
     f"{len(R)} sources, grouped by topic. Every entry was checked against a live source (publisher, proceedings, arXiv or an institutional repository) on 9 Nov 2026. "
     "Each entry says what the work is and where it fits in our paper. BibTeX keys match `references.bib`; the same table is in `references.csv`.", "",
     "## Where our paper sits (what the search found)", "",
     "- **Conformal prediction for flood maps** is rare. Konidakis et al. (IGARSS 2025) build pixel-level prediction sets for optical Sentinel-2 flood maps without domain shift; Li, Matgen and Chini (TGRS 2026) use conformal risk control on SAR flood maps for out-of-distribution abstention and set-valued predictions. Neither bounds the miss rate across flood events or combines the guarantee with test-time adaptation. "
     "General conformal segmentation (Mossina et al. 2024) treats images as units. We calibrate a miss-rate (FNR) guarantee with *flood events* as the exchangeable unit (cf. Dunn et al. 2023) across three SAR datasets.",
     "- **Test-time adaptation for SAR flood mapping already exists**: Chen et al. (2026, JGR: Machine Learning and Computation) adapt SAR inundation models to each new coastal flood with pseudo-labels and EMA updates. Our self-training baseline follows that recipe. What we found no precedent for is checking the pseudo-labels against radar and terrain physics, combining this with AdaBN, and attaching a flood-level miss-rate guarantee. TTA also exists in general remote sensing (Wang et al. 2024; Liang et al. 2024), and events-as-domains generalisation for floods (Turkmenli et al. 2025).",
     "- **HAND and terrain** are used as post-hoc masks or exclusion maps (Twele et al. 2016; Zhao et al. 2021; Misra et al. 2025), as Bayesian priors (Tupas et al. 2023) or as model inputs (Tian et al. 2026). "
     "We use radar physics plus HAND/slope to correct pseudo-labels *during* label-free adaptation.",
     "- **TTA can fail under real shift** (Niu et al. 2023). Our MMFlood result (TENT and AdaBN hurt under the guarantee; the physics check prevents it) adds evidence from Earth observation.",
     "- These are search findings, not proof of absence. Re-check the newest 2025–2026 papers just before submission.", "",
     "## Contents", ""]
for c in ORDER:
    n = sum(r["cat"] == c for r in R)
    L.append(f"- {c}. {CATS[c]} ({n})")
L.append("")
i = 0
for c in ORDER:
    L += [f"## {c}. {CATS[c]}", ""]
    for r in [r for r in R if r["cat"] == c]:
        i += 1
        det = f", {plain(r['details'])}" if r["details"] else ""
        L.append(f"**{i}. {short_authors(r['authors'])} ({r['year']}).** {plain(r['title'])}. *{plain(r['venue'])}*{det}. [{('doi:' + r['doi']) if r['doi'] else 'link'}]({link(r)})  ")
        L.append(f"What it is: {r['summary']}  ")
        L.append(f"How we use it: {r['use']}  ")
        L.append(f"Key: `{r['key']}`")
        L.append("")
open("literature_list.md", "w", encoding="utf-8").write("\n".join(L))
print(len(R), "entries written")
