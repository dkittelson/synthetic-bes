import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train_synthetic_bes_v3_1_shared as ref
from bes_sweep.configs import VARIANTS, build_model, count_params


def test_baseline_param_count_matches_mentor_model():
    assert count_params(build_model(VARIANTS["baseline"])) == 60449


def test_baseline_state_dict_matches_reference_model():
    ours = build_model(VARIANTS["baseline"]).state_dict()
    theirs = ref.TemporalSpatialELMMapNet(**ref.MODEL_KWARGS).state_dict()
    assert ours.keys() == theirs.keys()
    assert all(ours[k].shape == theirs[k].shape for k in ours)


def test_groupnorm_keeps_param_count():
    assert count_params(build_model(VARIANTS["base_gn"])) == 60449


@pytest.mark.parametrize("name", list(VARIANTS))
def test_forward_shape(name):
    model = build_model(VARIANTS[name]).eval()
    out = model(torch.randn(2, 320, 8, 8))
    assert out.shape == (2, 1, 2, 2)
