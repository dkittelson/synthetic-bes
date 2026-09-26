import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import explained_variance_score, r2_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train_synthetic_bes_v3_1_shared as ref
from bes_sweep import metrics

rng = np.random.default_rng(0)
Y = rng.choice([0.0, 0.5, 1.0], size=2000)
S = np.clip(Y * 0.7 + rng.normal(0.15, 0.2, size=Y.shape), 0.01, 0.99)


def test_r2_and_ev_match_sklearn():
    assert np.isclose(metrics.r2(Y, S), r2_score(Y, S))
    assert np.isclose(metrics.explained_variance(Y, S), explained_variance_score(Y, S))


def test_r2_is_one_minus_brier_over_variance():
    assert np.isclose(metrics.r2(Y, S), 1 - metrics.brier(Y, S) / np.var(Y))


def test_brier_and_log_loss_explicit_formula():
    assert np.isclose(metrics.brier(Y, S), np.mean((S - Y) ** 2))
    expected = -np.mean(Y * np.log(S) + (1 - Y) * np.log(1 - S))
    assert np.isclose(metrics.log_loss(Y, S), expected)


def test_auc_matches_sklearn_on_hard_subset():
    mask = Y != 0.5
    auc = ref.compute_roc_from_scores(Y, S)[3]
    assert np.isclose(auc, roc_auc_score((Y[mask] >= 0.75).astype(int), S[mask]))


def test_perfect_score_has_r2_one():
    assert np.isclose(metrics.r2(Y, Y), 1.0)
