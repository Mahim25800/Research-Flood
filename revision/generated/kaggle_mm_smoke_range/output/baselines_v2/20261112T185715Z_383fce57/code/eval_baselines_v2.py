"""Isolated event-wise evaluation on existing S1F/KS/MM data only.

Outputs never match legacy histogram globs. No data downloads are performed here.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import re
from pathlib import Path
import sys
import time

import numpy as np
import torch
from baselines_v2 import Adapter, Config, METHODS, estimate_source_fisher, sar_augment

HERE = Path(__file__).resolve().parent
if (HERE.parent / 'src').exists():
    sys.path.insert(0, str(HERE.parent / 'src'))
BINS = 1000
SENSITIVITIES = {'eata_v2_d09': 'eata_v2', 'sar_v2_scaled': 'sar_v2'}
ALL_METHODS = ('none', 'adabn', 'adabn_physa') + METHODS + tuple(SENSITIVITIES)


def method_config(method, config):
    if method == 'eata_v2_d09':
        return replace(config, diversity_margin=0.9)
    if method == 'sar_v2_scaled':
        return replace(config, recovery_entropy=0.2 * np.log(2) / np.log(1000))
    return config


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def source_pairs(pairs, held_out=None):
    return [p for p in pairs if held_out is None or p[0].split('_')[0] != held_out]


class Crops:
    def __init__(self, raw, mean, std, device, seed=0, batch=8, size=256):
        if not raw or any(r.shape[0] != 2 or not np.isfinite(r).all() for r in raw):
            raise ValueError('Empty/invalid VV/VH inputs')
        self.raw, self.device = raw, device
        self.mean = np.asarray(mean, dtype='float32')[:, None, None]
        self.std = np.asarray(std, dtype='float32')[:, None, None]
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError('Invalid source normalisation')
        self.rng, self.batch = np.random.default_rng(seed), batch
        self.size = min(size, *(min(r.shape[-2:]) for r in raw))

    def __call__(self):
        values = []
        for index in self.rng.integers(0, len(self.raw), self.batch):
            r = self.raw[index]
            h, w = r.shape[-2:]
            y, x = self.rng.integers(0, h - self.size + 1), self.rng.integers(0, w - self.size + 1)
            value = (r[:, y:y+self.size, x:x+self.size] - self.mean) / self.std
            value = np.rot90(value, int(self.rng.integers(4)), axes=(1, 2))
            if self.rng.random() < 0.5:
                value = value[:, :, ::-1]
            values.append(value.copy())
        return torch.from_numpy(np.stack(values)).to(self.device)

    def image(self, raw):
        return torch.from_numpy(((raw - self.mean) / self.std).copy())[None].to(self.device)


def score_arrays(model, method, ev, mean, std, config, fisher, steps, device, seed=0):
    # Only raw inputs reach Adapter. Labels are consulted after adaptation completes.
    torch.manual_seed(seed)
    crops = Crops(ev['raw'], mean, std, device, seed)
    adapter = Adapter(model, method, config, fisher)
    augment = sar_augment(torch.as_tensor(std))
    for _ in range(steps):
        adapter.step(crops(), augment)
    hist = np.zeros((2, 2, BINS), dtype=np.int64)
    with torch.no_grad():
        for i, raw in enumerate(ev['raw']):
            p = adapter.predict(crops.image(raw), augment).softmax(1)[0, 1].cpu().numpy()
            bins = np.clip((p * BINS).astype('int64'), 0, BINS-1)
            for subset, labels in [(0, ev['lab'][i])] + ([(1, ev['lab_alt'][i])] if 'lab_alt' in ev else []):
                if labels.shape != p.shape:
                    raise ValueError('Prediction/label shape mismatch')
                for label in (0, 1):
                    hist[subset, label] += np.bincount(bins[labels == label], minlength=BINS)
    return hist, dict(adapter.stats)


def dataset_events(name, work):
    from sen1floods11_unet_baseline import read_split, event_of
    if name == 'ks':
        from ks_eval import events_of
        return events_of(str(work / 'data_ks'))
    if name == 'mm':
        from mm_eval import events_of
        return events_of(str(work / 'data_mm'))
    all_pairs = sum([read_split(str(work / 'data'), s) for s in ('train', 'valid', 'test', 'bolivia')], [])
    return {event: [p for p in all_pairs if event_of(p[0]) == event] for event in sorted({event_of(p[0]) for p in all_pairs})}


def load_event(name, event, files, work):
    if name == 'ks':
        from ks_eval import load_ks_event
        ev = load_ks_event(files)
        ev['lab_alt'] = []
        for file, label in zip(files, ev['lab']):
            with np.load(file) as data:
                alt = label.copy()
                alt[data['mask'] == 1] = 255
                ev['lab_alt'].append(alt)
        return ev
    if name == 'mm':
        from mm_eval import load_mm_event
        return load_mm_event(files)
    from tta_px1 import load_event as load_s1f
    args = argparse.Namespace(data=str(work / 'data'))
    return load_s1f(args, event, files, set())


def validate_histogram(hist, ev):
    if hist.shape != (2, 2, BINS) or (hist < 0).any():
        raise ValueError('Invalid histogram')
    for subset, key in [(0, 'lab'), (1, 'lab_alt')]:
        expected = sum(int(((y == 0) | (y == 1)).sum()) for y in ev.get(key, []))
        if int(hist[subset].sum()) != expected:
            raise ValueError(f'Histogram pixel count mismatch for {key}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['ks', 's1f', 'mm'])
    p.add_argument('--work', type=Path, default=Path('/tmp/px1'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--methods', default=','.join(ALL_METHODS))
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = p.parse_args()
    if args.seed not in (0, 1, 2) or (args.dataset == 's1f' and args.seed != 0):
        p.error('Existing checkpoints: external seeds 0/1/2; S1F LOEO seed 0')
    methods = args.methods.split(',')
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(ALL_METHODS):
        p.error('Unknown or duplicate method')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this run; CPU unit tests are separate')
    work = args.work.resolve()
    events = dataset_events(args.dataset, work)
    if not events:
        raise ValueError('No eligible events')
    if not args.smoke:
        expected = {'ks': 26, 's1f': 11, 'mm': 69}[args.dataset]
        if len(events) != expected:
            raise ValueError(f'Unexpected event/region count {len(events)} != {expected}; inspect dataset before running')
        if args.dataset == 'mm' and len({e.split('-')[0] for e in events}) != 30:
            raise ValueError('Expected 30 MMFlood activations')
    if args.smoke:
        event = sorted(events)[0]
        events = {event: events[event][:2]}
    # Refuse reuse: completed and partial runs both remain immutable.
    args.out.mkdir(parents=True, exist_ok=False)
    config = Config()  # smoke uses the same 32-view CoTTA predictor as the full run
    steps, fisher_batches = (2, 2) if args.smoke else (200, 20)
    manifest = dict(schema='px1-baselines-v2', dataset=args.dataset, seed=args.seed,
        mode='smoke' if args.smoke else 'full', config=asdict(config), steps=steps,
        fisher_batches=fisher_batches, methods=methods,
        method_configs={m: asdict(method_config(m, config)) for m in methods if m in METHODS or m in SENSITIVITIES},
        events=sorted(events), status='started',
        software=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda),
        code={path.name: digest(path) for path in HERE.glob('*.py')}, records=[])
    write_json(args.out / 'manifest.json', manifest)
    try:
        from sen1floods11_unet_baseline import UNet, read_split, load_s1
        from tta_px1 import run_one
        training = read_split(str(work / 'data'), 'train')
        source_raw = {s: load_s1(str(work / 'data/S1Hand' / s)) for s, _ in training}
        source_hash = {s: digest(work / 'data/S1Hand' / s) for s in source_raw}
        cache = {}
        for event, files in sorted(events.items()):
            selected = source_pairs(training, event if args.dataset == 's1f' else None)
            if not selected:
                raise ValueError('Empty source training subset')
            checkpoint = work / ('runs_loeo' if args.dataset == 's1f' else 'runs') / (
                f'V0_hold{event}_seed0' if args.dataset == 's1f' else f'V0_seed{args.seed}') / 'model.pt'
            key = str(checkpoint)
            if key not in cache:
                values = [source_raw[s] for s, _ in selected]
                joined = np.concatenate([r.reshape(2, -1) for r in values], axis=1)
                mean, std = joined.mean(1).astype('float32'), joined.std(1).astype('float32') + 1e-6
                model = UNet(cin=2).to(args.device)
                model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
                fisher = None
                if any(m.startswith('eata_v2') for m in methods):
                    torch.manual_seed(1729)
                    source_crops = Crops(values, mean, std, args.device, seed=1729)
                    fisher = estimate_source_fisher(model, (source_crops() for _ in range(fisher_batches)))
                cache[key] = (model, mean, std, fisher)
            model, mean, std, fisher = cache[key]
            ev = load_event(args.dataset, event, files, work)
            # Record actual input files, including label files for the S1F reader.
            input_paths = ([work / 'data/S1Hand' / s for s, _ in files] + [work / 'data/LabelHand' / l for _, l in files]
                           if args.dataset == 's1f' else [Path(f) for f in files])
            record = dict(event=event, activation=event.split('-')[0] if args.dataset == 'mm' else event,
                checkpoint_sha256=digest(checkpoint), source_images={s: source_hash[s] for s, _ in selected},
                input_files={str(f): digest(f) for f in input_paths}, metrics={}, outputs={})
            for method in methods:
                start = time.monotonic()
                if method in METHODS or method in SENSITIVITIES:
                    hist, stats = score_arrays(model, SENSITIVITIES.get(method, method), ev, mean, std,
                                              method_config(method, config), fisher, steps, args.device)
                else:
                    legacy_args = argparse.Namespace(seed=0, model_seed=args.seed, steps=steps, runs=str(work / 'runs'))
                    hist = run_one(method, 'V0', event, ev, mean, std, legacy_args, args.device, ckpt=str(checkpoint))
                    stats = {'comparison_control': True}
                validate_histogram(hist, ev)
                safe = re.sub(r'[^A-Za-z0-9_-]', '_', event)
                output = args.out / f'{method}__{safe}.npz'
                if output.exists():
                    raise FileExistsError(output)
                np.savez_compressed(output, histogram=hist)
                record['outputs'][method] = dict(file=output.name, sha256=digest(output))
                record['metrics'][method] = dict(stats, seconds=time.monotonic() - start)
                print(event, method, record['metrics'][method], flush=True)
            manifest['records'].append(record)
            write_json(args.out / 'manifest.json', manifest)
            if args.dataset == 's1f':
                cache.clear()  # never retain 11 GPU models
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest['status'], manifest['error'] = 'failed', repr(exc)
        raise
    finally:
        write_json(args.out / 'manifest.json', manifest)
    print('BASELINES_V2_DONE', flush=True)


if __name__ == '__main__':
    main()
