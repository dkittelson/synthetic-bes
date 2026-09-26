import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bes_sweep import metrics
from bes_sweep.configs import VARIANTS, build_model
from bes_sweep.data import resolve_device


def test_eval_loss_is_repeatable_and_finite():
    """Regression: ref.evaluate_loss (non_blocking=True) gave NaN on MPS."""
    device = resolve_device()
    torch.manual_seed(0)
    model = build_model(VARIANTS["tiny"]).to(device)
    xs = torch.randn(256, 320, 8, 8)
    ys = torch.rand(256, 1, 2, 2)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xs, ys), batch_size=64)
    vals = [metrics.eval_loss(model, loader, nn.BCEWithLogitsLoss(), device) for _ in range(20)]
    assert np.all(np.isfinite(vals))
    assert np.ptp(vals) < 1e-6
