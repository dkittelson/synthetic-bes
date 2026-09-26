"""Model variants for the size sweep.

A variant is just a frozen config. The model class itself is the mentor's
`TemporalSpatialELMMapNet`; we only change its constructor arguments, plus an
optional BatchNorm -> GroupNorm swap for the normalisation ablation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

import train_synthetic_bes_v3_1_shared as ref


@dataclass(frozen=True)
class ModelConfig:
    name: str
    temporal_channels: tuple
    spatial_channels: tuple
    kernel_size: int = 7
    norm: str = "batch"  # "batch" or "group"

    def kwargs(self) -> dict:
        kw = dict(ref.MODEL_KWARGS)
        kw.update(
            temporal_channels=self.temporal_channels,
            spatial_channels=self.spatial_channels,
            temporal_kernel_size=self.kernel_size,
        )
        return kw


def _v(name, t, s, k=7, norm="batch"):
    return ModelConfig(name, tuple(t), tuple(s), k, norm)


# Ordered baseline-first, then small -> large, so a partial sweep is still useful.
_ORDER = [
    _v("baseline", (8, 16, 32), (64, 64)),
    _v("tiny", (4, 8, 8), (16, 16)),
    _v("s_xs", (8, 16, 32), (16, 16)),
    _v("small", (4, 8, 16), (32, 32)),
    _v("s_s", (8, 16, 32), (32, 32)),
    _v("t_xs", (4, 8, 16), (64, 64)),
    _v("base_k3", (8, 16, 32), (64, 64), k=3),
    _v("base_gn", (8, 16, 32), (64, 64), norm="group"),
    _v("base_k11", (8, 16, 32), (64, 64), k=11),
    _v("t_l", (16, 32, 64), (64, 64)),
    _v("s_l", (8, 16, 32), (128, 128)),
    _v("big", (16, 32, 64), (128, 128)),
    _v("s_xl", (8, 16, 32), (192, 192)),
    _v("xl", (16, 32, 64), (192, 192)),
]
VARIANTS = {c.name: c for c in _ORDER}
BASELINE = "baseline"


def replace_bn_with_gn(module: nn.Module, groups: int = 8) -> nn.Module:
    """Swap every BatchNorm1d/2d for GroupNorm (same trainable param count: 2*C)."""
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            c = child.num_features
            g = groups if c % groups == 0 else 1
            setattr(module, name, nn.GroupNorm(g, c))
        else:
            replace_bn_with_gn(child, groups)
    return module


def build_model(cfg: ModelConfig) -> nn.Module:
    model = ref.TemporalSpatialELMMapNet(**cfg.kwargs())
    if cfg.norm == "group":
        replace_bn_with_gn(model)
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
