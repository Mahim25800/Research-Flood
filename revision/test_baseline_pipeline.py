"""CPU integration test of the actual packaged worker with synthetic raster files.

Keeps fixtures/logs under revision/generated/test_artifacts; never uses research data.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import json
import subprocess
import sys
import uuid
import numpy as np
import torch
import rasterio
from rasterio.transform import from_origin
from build_baseline_runs import payload, ROOT


def run():
    torch.set_num_threads(1)
    path = ROOT / 'revision/generated/test_artifacts' / uuid.uuid4().hex
    work = path / 'work'
    work.mkdir(parents=True)
    for name, content in payload().items():
        (work / name).write_text(content, encoding='utf-8')
    sys.path.insert(0, str(work))
    from sen1floods11_unet_baseline import UNet
    (work / 'data/S1Hand').mkdir(parents=True)
    (work / 'data/flood_handlabeled').mkdir()
    rng = np.random.default_rng(123)
    pairs = []
    for event in ('SourceA', 'SourceB'):
        name = f'{event}_0_S1Hand.tif'
        raw = rng.normal(-15, 3, (2, 32, 32)).astype('float32')
        with rasterio.open(work / 'data/S1Hand' / name, 'w', driver='GTiff', width=32, height=32,
                           count=2, dtype='float32', crs='EPSG:32631', transform=from_origin(0, 1000, 10, 10)) as dst:
            dst.write(raw)
        pairs.append(f'{name},{event}_0_LabelHand.tif')
    (work / 'data/flood_handlabeled/flood_train_data.csv').write_text('\n'.join(pairs))
    event_dir = work / 'data_ks/train_GRD/synthetic_event'
    event_dir.mkdir(parents=True)
    for index in range(8):
        mask = np.zeros((32, 32), dtype='uint8')
        mask[:16] = 2
        mask[16:18] = 1
        np.savez_compressed(event_dir / f'{index}.npz',
            vv=rng.normal(-15, 3, (32, 32)).astype('float32'),
            vh=rng.normal(-20, 3, (32, 32)).astype('float32'),
            dem=np.zeros((32, 32), dtype='float32'), mask=mask, valid=np.ones((32, 32), dtype='uint8'))
    checkpoint = work / 'runs/V0_seed0'
    checkpoint.mkdir(parents=True)
    torch.manual_seed(123)
    model = UNet(cin=2)
    with torch.no_grad():
        model.head.bias.copy_(torch.tensor([-3., 3.]))
    torch.save(model.state_dict(), checkpoint / 'model.pt')
    output = path / 'outputs'
    command = [sys.executable, str(work / 'eval_baselines_v2.py'), '--dataset', 'ks', '--work', str(work),
               '--out', str(output), '--smoke', '--device', 'cpu']
    with open(path / 'pipeline.log', 'w') as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'), timeout=180)
    if completed.returncode:
        print((path / 'pipeline.log').read_text())
        raise RuntimeError(f'Pipeline failed, see {path}')
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'completed'
    assert len(manifest['records']) == 1
    record = manifest['records'][0]
    assert len(record['outputs']) == 9
    for item in record['outputs'].values():
        with np.load(output / item['file']) as data:
            hist = data['histogram']
            assert hist.shape == (2, 2, 1000)
            assert hist[0].sum() == 2 * 32 * 32
            assert hist[1].sum() == 2 * 30 * 32
    before = (output / 'manifest.json').read_bytes()
    # Existing artifacts must not be overwritten, even when arguments match.
    again = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'), timeout=60)
    assert again.returncode != 0
    assert before == (output / 'manifest.json').read_bytes()
    from verify_baseline_outputs import verify
    verified = verify(output, allow_smoke=True)
    assert verified['histogram_files'] == 9
    try:
        verify(output)
    except ValueError:
        pass
    else:
        raise AssertionError('Smoke output accepted as full experiment')
    report = dict(status='passed', dataset='synthetic only', methods=manifest['methods'],
                  histogram_pixel_counts='all 9 methods verified', refuses_overwrite=True,
                  artifact_directory=str(path), device='cpu')
    (path / 'test_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
