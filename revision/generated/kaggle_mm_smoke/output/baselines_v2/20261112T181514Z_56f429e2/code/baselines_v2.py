"""PX1 event-wise segmentation ports. See BASELINES_V2_PROTOCOL.md.

Independent implementation of mechanisms described by TENT, EATA, SAR and CoTTA.
No labels enter these adapters. Prediction never performs an optimiser update.
"""
from copy import deepcopy
from dataclasses import dataclass, asdict
import math

import torch
from torch import nn
from torch.nn import functional as F

METHODS = ('tent_v2', 'eata_v2', 'sar_v2', 'cotta_v2')


@dataclass(frozen=True)
class Config:
    lr: float = 1e-3
    teacher_lr: float = 1e-4
    entropy_margin: float = 0.4 * math.log(2)
    diversity_margin: float = 0.05
    fisher_weight: float = 2000.0
    rho: float = 0.05
    recovery_entropy: float = 0.2
    ema_decay: float = 0.99
    restore_probability: float = 0.01
    anchor_confidence: float = 0.9
    views: int = 32

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f'Nonfinite configuration: {key}')
        if self.lr <= 0 or self.teacher_lr <= 0 or self.views < 1:
            raise ValueError('Positive learning rates and view count required')
        if self.entropy_margin <= 0 or self.rho <= 0 or self.fisher_weight < 0:
            raise ValueError('Invalid loss settings')
        for value in (self.diversity_margin, self.ema_decay, self.restore_probability, self.anchor_confidence):
            if not 0 <= value <= 1:
                raise ValueError('Probability/decay/margin must be in [0,1]')


def entropy(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def configure(model, all_weights=False):
    model.train()
    model.requires_grad_(all_weights)
    found = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            found = True
            if not module.affine:
                raise ValueError('Affine BatchNorm required')
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            module.requires_grad_(True)
    if not found:
        raise ValueError('This version supports BatchNorm2d models only')
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def estimate_source_fisher(source_model, batches):
    """Mean squared batch pseudo-label gradient on SOURCE inputs; no model mutation.

    The caller supplies source-training crops, excluding the held-out source event.
    Returns CPU tensors, reusable for the same checkpoint/normalisation/source split.
    """
    model = deepcopy(source_model)
    parameters = configure(model)
    result = {name: torch.zeros_like(p) for name, p in parameters.items()}
    count = 0
    for x in batches:
        logits = model(x)
        loss = F.cross_entropy(logits, logits.detach().argmax(1))
        gradients = torch.autograd.grad(loss, tuple(parameters.values()))
        for name, grad in zip(parameters, gradients):
            if not torch.isfinite(grad).all():
                raise FloatingPointError('Nonfinite source Fisher gradient')
            result[name].add_(grad.detach().square())
        count += 1
    if count == 0:
        raise ValueError('EATA requires nonempty source Fisher batches')
    return {name: (value / count).cpu() for name, value in result.items()}


class Adapter:
    def __init__(self, source_model, method, config=None, fisher=None):
        if method not in METHODS:
            raise ValueError(f'Unknown baseline {method}')
        self.method, self.config = method, config or Config()
        self.model = deepcopy(source_model)
        self.parameters = configure(self.model, all_weights=method == 'cotta_v2')
        self.initial = deepcopy(self.model.state_dict())
        lr = self.config.teacher_lr if method == 'cotta_v2' else self.config.lr
        self.optimizer = torch.optim.Adam(self.parameters.values(), lr=lr)
        self.initial_optimizer = deepcopy(self.optimizer.state_dict())
        self.fisher = {}
        if method == 'eata_v2':
            if fisher is None or set(fisher) != set(self.parameters):
                raise ValueError('EATA requires Fisher for every adapted source parameter')
            for name, parameter in self.parameters.items():
                value = fisher[name].to(parameter)
                if value.shape != parameter.shape or not torch.isfinite(value).all() or (value < 0).any():
                    raise ValueError(f'Invalid Fisher tensor: {name}')
                self.fisher[name] = value.detach().clone()
        self.teacher = self.anchor = None
        if method == 'cotta_v2':
            self.teacher = deepcopy(self.model).requires_grad_(False)
            self.anchor = deepcopy(self.model).requires_grad_(False)
        self.probability_average = None
        self.entropy_average = None
        self.stats = dict(steps=0, updates=0, skipped=0, reliable_pixels=0,
                          selected_pixels=0, recoveries=0, augmented_calls=0, plain_calls=0)

    def reset(self):
        self.model.load_state_dict(self.initial)
        self.optimizer.load_state_dict(deepcopy(self.initial_optimizer))
        self.optimizer.zero_grad(set_to_none=True)
        self.probability_average = self.entropy_average = None
        if self.teacher is not None:
            self.teacher.load_state_dict(self.initial)
            self.anchor.load_state_dict(self.initial)

    def _update(self, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite adaptation loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.parameters.values()):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('Nonfinite adaptation gradient')
        self.optimizer.step()
        self.stats['updates'] += 1

    def _eata(self, x):
        logits = self.model(x)
        h = entropy(logits)
        reliable = h < self.config.entropy_margin
        probabilities = logits.softmax(1).movedim(1, -1)[reliable]
        losses = h[reliable]
        self.stats['reliable_pixels'] += losses.numel()
        if probabilities.numel() and self.probability_average is not None:
            similarity = F.cosine_similarity(probabilities.detach(), self.probability_average[None], dim=1)
            diverse = similarity.abs() < self.config.diversity_margin
            probabilities, losses = probabilities[diverse], losses[diverse]
        self.stats['selected_pixels'] += losses.numel()
        if losses.numel() == 0:
            self.stats['skipped'] += 1
            return
        current = probabilities.detach().mean(0)
        self.probability_average = current if self.probability_average is None else 0.9 * self.probability_average + 0.1 * current
        weighted = (losses * torch.exp(self.config.entropy_margin - losses.detach())).mean()
        penalty = sum((self.fisher[n] * (p - self.initial[n]).square()).sum() for n, p in self.parameters.items())
        self._update(weighted + self.config.fisher_weight * penalty)

    def _sar(self, x):
        h = entropy(self.model(x))
        selected = h < self.config.entropy_margin
        self.stats['reliable_pixels'] += int(selected.sum())
        if not selected.any():
            self.stats['skipped'] += 1
            return
        parameters = tuple(self.parameters.values())
        gradients = torch.autograd.grad(h[selected].mean(), parameters)
        norm = torch.stack([g.square().sum() for g in gradients]).sum().sqrt()
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite sharpness gradient')
        originals = [p.detach().clone() for p in parameters]
        self.optimizer.zero_grad(set_to_none=True)
        value = None
        try:
            with torch.no_grad():
                for p, grad in zip(parameters, gradients):
                    p.add_(grad * (self.config.rho / norm.clamp_min(1e-12)))
            h2 = entropy(self.model(x))
            keep = selected & (h2 < self.config.entropy_margin)
            self.stats['selected_pixels'] += int(keep.sum())
            if keep.any():
                loss = h2[keep].mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite sharpness loss')
                loss.backward()
                value = float(loss.detach())
        finally:
            # Restore exactly even if the second filter is empty or forward raises.
            with torch.no_grad():
                for p, original in zip(parameters, originals):
                    p.copy_(original)
        if value is None:
            self.stats['skipped'] += 1
            self.optimizer.zero_grad(set_to_none=True)
            return
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError('Nonfinite second sharpness gradient')
        self.optimizer.step()
        self.stats['updates'] += 1
        self.entropy_average = value if self.entropy_average is None else 0.9 * self.entropy_average + 0.1 * value
        if self.entropy_average < self.config.recovery_entropy:
            self.reset()
            self.stats['recoveries'] += 1

    @torch.no_grad()
    def _teacher_prediction(self, x, augment):
        confidence = self.anchor(x).softmax(1).amax(1).mean()
        if confidence >= self.config.anchor_confidence:
            self.stats['plain_calls'] += 1
            return self.teacher(x)
        if augment is None:
            raise ValueError('CoTTA low-confidence prediction requires spatially invertible SAR augmentation')
        total = None
        for _ in range(self.config.views):
            view, undo = augment(x)
            logits = undo(self.teacher(view))
            total = logits if total is None else total + logits
        self.stats['augmented_calls'] += 1
        return total / self.config.views

    def _cotta(self, x, augment):
        target = self._teacher_prediction(x, augment).softmax(1)
        self._update(-(target * self.model(x).log_softmax(1)).sum(1).mean())
        with torch.no_grad():
            for teacher, student in zip(self.teacher.parameters(), self.model.parameters()):
                teacher.lerp_(student, 1 - self.config.ema_decay)
            for name, p in self.parameters.items():
                restore = torch.rand_like(p) < self.config.restore_probability
                p.copy_(torch.where(restore, self.initial[name], p))

    def step(self, x, augment=None):
        if not torch.isfinite(x).all():
            raise ValueError('Nonfinite target input')
        self.stats['steps'] += 1
        self.model.train()
        with torch.enable_grad():
            if self.method == 'tent_v2':
                self._update(entropy(self.model(x)).mean())
            elif self.method == 'eata_v2':
                self._eata(x)
            elif self.method == 'sar_v2':
                self._sar(x)
            else:
                self._cotta(x, augment)

    @torch.no_grad()
    def predict(self, x, augment=None):
        self.model.eval()
        if self.teacher is not None:
            self.teacher.eval()
            self.anchor.eval()
            logits = self._teacher_prediction(x, augment)
        else:
            logits = self.model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite prediction')
        return logits


def sar_augment(std):
    """D4 geometry with exact inverse; +/-1.5 dB shared radiometric offset.

    Operates on normalised VV/VH only; radiometric offsets are scaled by source std.
    """
    def transform(x):
        if x.shape[1] != 2:
            raise ValueError('SAR augmentation requires VV/VH input')
        k = int(torch.randint(4, ()).item())
        flip = bool(torch.randint(2, ()).item())
        value = torch.rot90(x, k, (-2, -1))
        if flip:
            value = value.flip(-1)
        offset = (torch.rand((x.shape[0], 1, 1, 1), device=x.device) * 3 - 1.5)
        value = value + offset / std.to(x).reshape(1, 2, 1, 1)
        def inverse(logits):
            if flip:
                logits = logits.flip(-1)
            return torch.rot90(logits, -k, (-2, -1))
        return value, inverse
    return transform
