"""CPU-only regression checks; no data downloads, training runs or GPU access."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
import torch._dynamo  # initialise lazy imports before scoped sys.modules mocks
from torch import nn
import tta_px1 as tta
from tent_batch_patch import patch as patch_bundle


class TinyNet(nn.Module):
    def __init__(self, cin=2):
        super().__init__()
        self.conv = nn.Conv2d(cin, 4, 1)
        self.bn = nn.BatchNorm2d(4)
        self.head = nn.Conv2d(4, 2, 1)

    def forward(self, x):
        return self.head(self.bn(self.conv(x)))


class ValidityTests(unittest.TestCase):
    def run_event(self, method, label):
        torch.manual_seed(12)
        initial = TinyNet().state_dict()
        captured = []
        def factory(cin=2):
            model = TinyNet(cin)
            captured.append(model)
            return model
        module = types.ModuleType('sen1floods11_unet_baseline')
        module.UNet, module.IGNORE = factory, 255
        rng = np.random.default_rng(4)
        ev = dict(raw=[rng.normal(size=(2, 8, 8)).astype('float32')],
                  terr=[np.zeros((3, 8, 8), dtype='float32')],
                  lab=[np.full((8, 8), label, dtype='uint8')], clean=[False])
        args = types.SimpleNamespace(seed=0, model_seed=0, steps=2, runs='unused')
        with patch.dict(sys.modules, {'sen1floods11_unet_baseline': module}), \
             patch.object(torch, 'load', return_value=initial):
            hist = tta.run_one(method, 'V0', 'toy', ev, np.zeros(2, dtype='float32'), np.ones(2, dtype='float32'), args, 'cpu', ckpt='unused')
        return captured[0], hist, initial

    def test_batch_statistics_survive_eval(self):
        model, hist, initial = self.run_event('tent_batch', 1)
        self.assertFalse(model.training)
        self.assertFalse(model.bn.track_running_stats)
        self.assertIsNone(model.bn.running_mean)
        self.assertIsNone(model.bn.running_var)
        self.assertEqual(int(hist[0].sum()), 64)
        for name in ['conv.weight', 'conv.bias', 'head.weight', 'head.bias']:
            torch.testing.assert_close(model.state_dict()[name], initial[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(model.bn.bias.detach(), initial['bn.bias']))

    def test_labels_do_not_affect_adaptation(self):
        a, ha, _ = self.run_event('tent_batch', 0)
        b, hb, _ = self.run_event('tent_batch', 1)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
        np.testing.assert_array_equal(ha[0, 0], hb[0, 1])

    def test_legacy_tent_retains_running_statistics(self):
        model, _, _ = self.run_event('tent', 1)
        self.assertTrue(model.bn.track_running_stats)
        self.assertIsNotNone(model.bn.running_mean)

    def test_bundle_patch_is_idempotent_and_compiles(self):
        source = (ROOT / 'src/tta_px1.py').read_text(encoding='utf-8')
        self.assertEqual(patch_bundle(source), source)
        # Exercise a legacy bundle as well, preserving its historical method key.
        legacy = source.replace('ALL_METHODS += ["tent_batch"]  # separate key: never reuse historical tent histograms\n', '')
        legacy = legacy.replace('if method in ("adabn", "tent", "tent_batch"):', 'if method in ("adabn", "tent"):')
        start = legacy.index('    if pre in ("tent", "tent_batch"):')
        end = legacy.index('        params = ', start)
        legacy = legacy[:start] + '    if pre == "tent":\n        model.train()\n' + legacy[end:]
        result = patch_bundle(legacy)
        self.assertEqual(patch_bundle(result), result)
        self.assertIn('"tent_batch"', result)

    def test_kaggle_patch_chain(self):
        import r34_patch
        import rev_patch
        files = {name: (ROOT / 'src' / name).read_text(encoding='utf-8')
                 for name in set(r34_patch.PATCHES) | set(rev_patch.PATCHES)}
        for patches in (r34_patch.PATCHES, rev_patch.PATCHES):
            for name, replacements in patches.items():
                for old, new in replacements:
                    self.assertEqual(files[name].count(old), 1, (name, old[:60]))
                    files[name] = files[name].replace(old, new)
                compile(files[name], name, 'exec')
        self.assertIn('"tent_batch"', files['tta_px1.py'])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
