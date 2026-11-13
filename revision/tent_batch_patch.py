"""Add a separately keyed TENT batch-statistics diagnostic to a staged PX1 bundle.

Apply after r34_patch.py and rev_patch.py. Historical `tent` behaviour is retained.
This fixes BN semantics only; the event-wise schedule is still a PX1 adaptation.
"""
import argparse
from pathlib import Path

MARKER = 'ALL_METHODS += ["tent_batch"]'
REPLACEMENTS = [
    ('    if method in ("adabn", "tent"):\n',
     '    if method in ("adabn", "tent", "tent_batch"):\n'),
    ('    if pre == "tent":\n        model.train()\n',
     '''    if pre in ("tent", "tent_batch"):
        model.train()
        if pre == "tent_batch":
            for mod in model.modules():
                if isinstance(mod, torch.nn.BatchNorm2d):
                    mod.track_running_stats = False
                    mod.running_mean = None
                    mod.running_var = None
'''),
]


def patch(source):
    if MARKER in source:
        if 'if pre in ("tent", "tent_batch")' not in source:
            raise ValueError("Partial TENT-B patch detected")
        return source
    for old, new in REPLACEMENTS:
        if source.count(old) != 1:
            raise ValueError(f"TENT-B patch mismatch: {old!r}")
        source = source.replace(old, new)
    lines = source.splitlines(keepends=True)
    positions = [i for i, line in enumerate(lines) if line.startswith('ALL_METHODS = ')]
    if len(positions) != 1:
        raise ValueError("Expected one ALL_METHODS declaration")
    lines.insert(positions[0] + 1, MARKER + '\n')
    source = ''.join(lines)
    compile(source, 'tta_px1.py', 'exec')
    return source


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('code_dir', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    path = args.code_dir / 'tta_px1.py'
    result = patch(path.read_text(encoding='utf-8'))
    if not args.check:
        path.write_text(result, encoding='utf-8')
    print('TENT_BATCH_PATCH_OK')
