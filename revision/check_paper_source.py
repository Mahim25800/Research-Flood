"""Check manuscript inputs/references and abstract length without claiming a TeX build."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / 'paper_wip'


if __name__ == '__main__':
    text = (PAPER / 'main.tex').read_text(encoding='utf-8')
    abstract = text.split(r'\begin{abstract}', 1)[1].split(r'\end{abstract}', 1)[0]
    words = len(abstract.split())
    missing = []
    for path in re.findall(r'\\input\{([^}]+)\}', text):
        file = PAPER / path
        if not file.with_suffix('.tex').exists():
            missing.append(str(file))
    for path in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text):
        if not (PAPER / path).exists():
            missing.append(str(PAPER / path))
    bib = '\n'.join((PAPER / (name.strip() + '.bib')).read_text(encoding='utf-8')
                    for name in re.search(r'\\bibliography\{([^}]+)\}', text)[1].split(','))
    keys = set(re.findall(r'@\w+\s*\{\s*([^,]+),', bib))
    all_text = text + '\n' + '\n'.join(p.read_text(encoding='utf-8') for p in (PAPER / 'tables').glob('*.tex'))
    used = {key.strip() for entry in re.findall(r'\\cite\w*\{([^}]+)\}', all_text) for key in entry.split(',')}
    report = dict(abstract_words_whitespace=words, missing_inputs=missing,
                  missing_citations=sorted(used - keys), references_cited=len(used),
                  note='Static source checks only; not a TeX compilation or reference-content audit.')
    print(json.dumps(report, indent=2))
    if missing or used - keys or words > 250:
        raise SystemExit(1)
