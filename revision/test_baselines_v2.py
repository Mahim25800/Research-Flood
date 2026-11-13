"""Mechanism regression tests. CPU only, entirely synthetic inputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch
import torch
from torch import nn
from baselines_v2 import Adapter, Config, METHODS, configure, entropy, estimate_source_fisher, sar_augment


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Conv2d(2, 4, 1)
        self.bn = nn.BatchNorm2d(4)
        self.head = nn.Conv2d(4, 2, 1)

    def forward(self, x):
        return self.head(self.bn(self.feature(x)))


class BaselineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.model = SmallNet()
        self.x = torch.randn(2, 2, 8, 8)
        self.fisher = estimate_source_fisher(self.model, [self.x, self.x * 0.7])

    def assert_state_equal(self, a, b):
        self.assertEqual(set(a), set(b))
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)

    def test_all_baselines_finite_and_source_unchanged(self):
        original = deepcopy(self.model.state_dict())
        for method in METHODS:
            adapter = Adapter(self.model, method, Config(views=2), self.fisher if method == 'eata_v2' else None)
            aug = sar_augment(torch.ones(2))
            adapter.step(self.x, aug)
            probability = adapter.predict(self.x, aug).softmax(1)
            self.assertTrue(torch.isfinite(probability).all())
            self.assert_state_equal(original, self.model.state_dict())
            self.assertIsNone(adapter.model.bn.running_mean)
            self.assertFalse(adapter.model.bn.track_running_stats)

    def test_tent_matches_one_manual_entropy_step(self):
        oracle = deepcopy(self.model)
        params = configure(oracle)
        optimizer = torch.optim.Adam(params.values(), lr=Config().lr)
        logits = oracle(self.x)
        loss = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
        loss.backward()
        optimizer.step()
        adapter = Adapter(self.model, 'tent_v2')
        adapter.step(self.x)
        self.assert_state_equal(oracle.state_dict(), adapter.model.state_dict())

    def test_fisher_source_only_nonnegative_and_no_mutation(self):
        original = deepcopy(self.model.state_dict())
        f = estimate_source_fisher(self.model, [self.x])
        self.assertTrue(all((v >= 0).all() and torch.isfinite(v).all() for v in f.values()))
        self.assertTrue(any((v > 0).any() for v in f.values()))
        self.assert_state_equal(original, self.model.state_dict())
        with self.assertRaises(ValueError):
            estimate_source_fisher(self.model, [])
        with self.assertRaises(ValueError):
            Adapter(self.model, 'eata_v2')

    def test_eata_rejects_redundant_pixels_without_optimizer_step(self):
        config = Config(entropy_margin=1.0, diversity_margin=0.0)
        adapter = Adapter(self.model, 'eata_v2', config, self.fisher)
        adapter.step(self.x)
        self.assertEqual(adapter.stats['updates'], 1)
        before = deepcopy(adapter.model.state_dict())
        adapter.step(self.x)
        self.assertEqual(adapter.stats['updates'], 1)
        self.assertEqual(adapter.stats['skipped'], 1)
        self.assert_state_equal(before, adapter.model.state_dict())

    def test_eata_fisher_penalty_and_weighted_entropy(self):
        config = Config(entropy_margin=1, diversity_margin=1)
        adapter = Adapter(self.model, 'eata_v2', config, self.fisher)
        with torch.no_grad():
            for p in adapter.parameters.values():
                p.add_(0.1)
        h = entropy(adapter.model(self.x))
        expected_entropy = (h * torch.exp(config.entropy_margin - h.detach())).mean()
        expected_penalty = sum((self.fisher[n] * (p - adapter.initial[n]).square()).sum()
                               for n, p in adapter.parameters.items()) * config.fisher_weight
        self.assertGreater(float(expected_penalty.detach()), 0)
        with patch.object(adapter, '_update') as update:
            adapter.step(self.x)
        torch.testing.assert_close(update.call_args.args[0], expected_entropy + expected_penalty)

    def test_only_bn_parameters_change_for_entropy_methods(self):
        for method in ('tent_v2', 'eata_v2', 'sar_v2'):
            adapter = Adapter(self.model, method, Config(entropy_margin=1, recovery_entropy=0),
                              self.fisher if method == 'eata_v2' else None)
            before = deepcopy(adapter.model.state_dict())
            adapter.step(self.x)
            for name, value in adapter.model.state_dict().items():
                if not name.startswith('bn.'):
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_source_holdout_filter_and_named_sensitivities(self):
        from eval_baselines_v2 import source_pairs, method_config
        pairs = [('Bolivia_1.tif', 'unused'), ('India_1.tif', 'unused')]
        self.assertEqual(source_pairs(pairs, 'Bolivia'), [pairs[1]])
        self.assertEqual(source_pairs(pairs), pairs)
        self.assertEqual(method_config('eata_v2_d09', Config()).diversity_margin, 0.9)
        self.assertLess(method_config('sar_v2_scaled', Config()).recovery_entropy, Config().recovery_entropy)

    def test_empty_reliability_filter_is_finite_and_does_not_update(self):
        for method in ('eata_v2', 'sar_v2'):
            adapter = Adapter(self.model, method, Config(entropy_margin=1e-12), self.fisher if method == 'eata_v2' else None)
            before = deepcopy(adapter.model.state_dict())
            adapter.step(self.x)
            self.assertEqual(adapter.stats['updates'], 0)
            self.assertEqual(adapter.stats['skipped'], 1)
            self.assert_state_equal(before, adapter.model.state_dict())

    def test_sar_recovers_model_optimizer_and_entropy_state(self):
        adapter = Adapter(self.model, 'sar_v2', Config(entropy_margin=1, recovery_entropy=1))
        before = deepcopy(adapter.model.state_dict())
        adapter.step(self.x)
        self.assertEqual(adapter.stats['recoveries'], 1)
        self.assertIsNone(adapter.entropy_average)
        self.assertEqual(adapter.optimizer.state_dict()['state'], {})
        self.assert_state_equal(before, adapter.model.state_dict())

    def test_sar_restores_perturbation_when_second_filter_empty(self):
        adapter = Adapter(self.model, 'sar_v2', Config(entropy_margin=1))
        before = deepcopy(adapter.model.state_dict())
        count = 0
        def changing_entropy(logits):
            nonlocal count
            count += 1
            h = entropy(logits)
            return h if count == 1 else h + 10
        with patch('baselines_v2.entropy', side_effect=changing_entropy):
            adapter.step(self.x)
        self.assertEqual(adapter.stats['skipped'], 1)
        self.assert_state_equal(before, adapter.model.state_dict())

    def test_sar_restores_perturbation_on_exception(self):
        adapter = Adapter(self.model, 'sar_v2', Config(entropy_margin=1))
        before = deepcopy(adapter.model.state_dict())
        with patch('baselines_v2.entropy', side_effect=[entropy(adapter.model(self.x)), RuntimeError('test')]):
            with self.assertRaises(RuntimeError):
                adapter.step(self.x)
        self.assert_state_equal(before, adapter.model.state_dict())

    def test_cotta_confidence_gate_and_logit_average(self):
        for threshold, expected in [(0.0, 0), (1.0, 3)]:
            adapter = Adapter(self.model, 'cotta_v2', Config(anchor_confidence=threshold, views=3))
            calls = []
            def aug(x):
                calls.append(1)
                return x.flip(-1), lambda y: y.flip(-1)
            output = adapter.predict(self.x, aug)
            self.assertEqual(len(calls), expected)
            torch.testing.assert_close(output, adapter.teacher(self.x), atol=1e-6, rtol=1e-6)

    def test_cotta_restore_and_teacher_update(self):
        adapter = Adapter(self.model, 'cotta_v2', Config(anchor_confidence=0, restore_probability=1))
        before = deepcopy(adapter.model.state_dict())
        adapter.step(self.x)
        self.assert_state_equal(before, adapter.model.state_dict())
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in adapter.teacher.state_dict().items() if v.is_floating_point()))
        self.assert_state_equal(before, adapter.anchor.state_dict())
        adapter.reset()
        self.assert_state_equal(before, adapter.teacher.state_dict())

    def test_predict_does_not_update_weights(self):
        for method in METHODS:
            adapter = Adapter(self.model, method, Config(views=2), self.fisher if method == 'eata_v2' else None)
            before = deepcopy(adapter.model.state_dict())
            adapter.predict(self.x, sar_augment(torch.ones(2)))
            self.assertEqual(adapter.stats['updates'], 0)
            self.assert_state_equal(before, adapter.model.state_dict())

    def test_sar_augmentation_inverse_preserves_spatial_pattern(self):
        aug = sar_augment(torch.tensor([2., 4.]))
        x = torch.arange(2 * 6 * 8, dtype=torch.float32).reshape(1, 2, 6, 8)
        for _ in range(8):
            view, undo = aug(x)
            difference = undo(view) - x
            torch.testing.assert_close(difference, difference[:, :, :1, :1].expand_as(x), atol=1e-5, rtol=0)
            torch.testing.assert_close(difference[:, 0] * 2, difference[:, 1] * 4, atol=2e-5, rtol=0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
