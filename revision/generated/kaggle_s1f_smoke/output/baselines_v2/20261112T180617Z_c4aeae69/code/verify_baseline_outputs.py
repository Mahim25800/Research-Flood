"""Validate v2 manifests/histograms before downstream statistical analysis."""
import argparse
import json
from pathlib import Path
import numpy as np
from eval_baselines_v2 import digest, BINS


def verify(directory, allow_smoke=False):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema') != 'px1-baselines-v2' or manifest.get('status') != 'completed':
        raise ValueError('Only completed v2 workers can be analysed')
    if manifest['mode'] != 'full' and not allow_smoke:
        raise ValueError('Smoke results must not enter final analysis')
    events = manifest['events']
    records = manifest['records']
    if len(events) != len(set(events)) or sorted(r['event'] for r in records) != sorted(events):
        raise ValueError('Incomplete or duplicate event records')
    outputs = set()
    for record in records:
        if set(record['outputs']) != set(manifest['methods']):
            raise ValueError('Incomplete method coverage')
        for item in record['outputs'].values():
            name = item['file']
            if Path(name).name != name or name in outputs:
                raise ValueError('Invalid or duplicate output filename')
            outputs.add(name)
            file = directory / name
            if digest(file) != item['sha256']:
                raise ValueError(f'Changed result file: {name}')
            with np.load(file, allow_pickle=False) as data:
                hist = data['histogram']
                if hist.shape != (2, 2, BINS) or not np.issubdtype(hist.dtype, np.integer) or (hist < 0).any():
                    raise ValueError(f'Invalid histogram: {name}')
    if outputs != {p.name for p in directory.glob('*.npz')}:
        raise ValueError('Unrecorded histogram files present')
    return dict(dataset=manifest['dataset'], seed=manifest['seed'], mode=manifest['mode'],
                events=len(events), methods=manifest['methods'], histogram_files=len(outputs), status='verified')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('--allow-smoke', action='store_true')
    args = p.parse_args()
    print(json.dumps(verify(args.directory, args.allow_smoke), indent=2))
