"""Compare verified full runs at threshold 0.5 and a 20% calibration target."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from eval_baselines_v2 import ALL_METHODS, BINS, digest
from rev_analyse import curves, k_crc
from verify_baseline_outputs import verify


SEEDS = {'s1f': {0}, 'ks': {0, 1, 2}, 'mm': {0, 1, 2}}
COUNTS = {'s1f': (11, 11), 'ks': (26, 26), 'mm': (69, 30)}
METRICS = ('iou', 'miss_rate', 'false_alarm_rate')


def read_workers(paths):
    workers = {}
    for path in paths:
        verify(path)
        manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
        dataset, seed = manifest['dataset'], manifest['seed']
        key = dataset, seed
        if dataset not in SEEDS or seed not in SEEDS[dataset] or key in workers:
            raise ValueError(f'Unexpected or duplicate worker: {key}')
        records = manifest['records']
        coverage = {record['event']: record['activation'] for record in records}
        if (len(coverage), len(set(coverage.values()))) != COUNTS[dataset]:
            raise ValueError(f'Unexpected coverage for {key}')
        if set(manifest['methods']) != set(ALL_METHODS):
            raise ValueError(f'Missing comparison methods for {key}')
        histograms = {}
        for record in records:
            unit = record['activation']
            reference = None
            for method in ALL_METHODS:
                with np.load(path / record['outputs'][method]['file'], allow_pickle=False) as file:
                    hist = file['histogram']
                support = hist.sum(axis=-1)
                if reference is None:
                    reference = support
                elif not np.array_equal(support, reference):
                    raise ValueError(f'Label support mismatch: {key}, {record["event"]}, {method}')
                entry = histograms.setdefault((unit, method), np.zeros_like(hist, dtype=np.int64))
                entry += hist
        workers[key] = (coverage, histograms, digest(path / 'manifest.json'))
    return workers


def analyse(workers):
    rows = []
    for dataset in sorted({dataset for dataset, _ in workers}):
        seeds = {seed for name, seed in workers if name == dataset}
        if seeds != SEEDS[dataset]:
            raise ValueError(f'{dataset} requires seeds {sorted(SEEDS[dataset])}; found {sorted(seeds)}')
        coverage, first, _ = workers[dataset, min(seeds)]
        for seed in sorted(seeds):
            current_coverage, current, _ = workers[dataset, seed]
            if current_coverage != coverage or any(
                not np.array_equal(current[key].sum(axis=-1), hist.sum(axis=-1)) for key, hist in first.items()
            ):
                raise ValueError(f'Event or label coverage differs across {dataset} seeds')
        for seed in sorted(seeds):
            _, histograms, _ = workers[dataset, seed]
            for subset, label in enumerate(('all_water', 'flood_water_only') if dataset == 'ks' else ('all_water',)):
                for method in ALL_METHODS:
                    units = {unit: histograms[unit, method][subset] for unit in sorted(set(coverage.values()))}
                    stats = {unit: curves(hist) for unit, hist in units.items()}
                    for unit, hist in units.items():
                        fnr, iou, _, far = stats[unit]
                        cal = [stats[other][0] for other in units if other != unit and units[other][1].sum() > 0]
                        if not cal:
                            raise ValueError(f'Empty calibration pool for {dataset}/{unit}')
                        for rule, k in (('fixed_0.5', BINS // 2), ('loeo_target_0.2', k_crc(cal, 0.2))):
                            rows.append(dict(dataset=dataset, label=label, seed=seed, unit=unit, method=method,
                                             rule=rule, threshold=k / BINS, iou=float(iou[k]),
                                             miss_rate=float(fnr[k]) if hist[1].sum() else None,
                                             false_alarm_rate=float(far[k]) if hist[0].sum() else None))
    return rows


def mean(values):
    observed = [value for value in values if value is not None]
    return float(np.mean(observed)) if observed else None


def summarise(rows):
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in ('dataset', 'label', 'rule', 'method'))
        groups.setdefault(key, {}).setdefault(row['unit'], []).append(row)
    output = []
    for (dataset, label, rule, method), units in sorted(groups.items()):
        entry = dict(dataset=dataset, label=label, rule=rule, method=method, units=len(units))
        for metric in METRICS:
            values = [mean([row[metric] for row in seeds]) for seeds in units.values()]
            entry[metric] = mean(values)
            entry[f'{metric}_units'] = sum(value is not None for value in values)
        output.append(entry)
    return output


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workers', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    workers = read_workers(args.workers)
    rows = analyse(workers)
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out / 'per_unit_seed.csv', rows)
    write_csv(args.out / 'comparison.csv', summarise(rows))
    provenance = {f'{dataset}/{seed}': worker[2] for (dataset, seed), worker in workers.items()}
    (args.out / 'inputs.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
