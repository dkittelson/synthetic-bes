"""Metrics for the sweep. Pure numpy except `evaluate`.

Two conventions, deliberately kept separate (state both in the table caption):
  * Ranking / thresholded metrics (AUC, accuracy, F1, ...) follow the mentor:
    windows labelled 0.5 are dropped, label >= 0.75 is positive, and the
    decision threshold is the Youden-J optimum chosen on VAL only.
  * Calibration / regression metrics (Brier, log loss, R2, explained variance)
    compare the score `mean(sigmoid(2x2 map))` with the SOFT label (0, 0.5, 1)
    and keep every window, including the 0.5 ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import train_synthetic_bes_v3_1_shared as ref

EPS = 1e-7


def brier(y, s) -> float:
    y, s = np.asarray(y, float), np.asarray(s, float)
    return float(np.mean((s - y) ** 2))


def log_loss(y, s) -> float:
    """Cross-entropy with soft targets, scores clipped away from 0/1."""
    y = np.asarray(y, float)
    s = np.clip(np.asarray(s, float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(s) + (1 - y) * np.log(1 - s)))


def r2(y, s) -> float:
    """1 - SSE/SST. On a fixed set this equals 1 - Brier/Var(y)."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - s) ** 2) / sst)


def explained_variance(y, s) -> float:
    """1 - Var(y - s)/Var(y). Unlike R2 it ignores a constant bias in s."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    return float(1.0 - np.var(y - s) / np.var(y))


def event_auc_summary(table: pd.DataFrame) -> dict:
    """Per-event AUC (0.5 windows dropped), like the notebook. Events lacking
    both classes are skipped and counted."""
    aucs = []
    for _, g in table.groupby("event"):
        try:
            aucs.append(ref.compute_roc_from_scores(g["target"].values, g["score"].values)[3])
        except RuntimeError:
            continue
    aucs = np.asarray(aucs)
    if len(aucs) == 0:
        return dict(event_auc_mean=float("nan"), event_auc_median=float("nan"),
                    event_auc_min=float("nan"), event_auc_n=0)
    return dict(event_auc_mean=float(aucs.mean()), event_auc_median=float(np.median(aucs)),
                event_auc_min=float(aucs.min()), event_auc_n=int(len(aucs)))


def soft_metrics(y, s) -> dict:
    return dict(brier=brier(y, s), log_loss=log_loss(y, s), r2=r2(y, s),
                explained_var=explained_variance(y, s))


def eval_loss(model, loader, loss_fn, device) -> float:
    """Mean loss over a loader.

    Deliberately NOT ref.evaluate_loss: that copies batches with
    non_blocking=True, which is fine with CUDA pinned memory but on MPS with
    ordinary tensors it intermittently yields NaN/-inf (verified: same model and
    data, 30 repeats, some NaN). Blocking copies are stable.
    """
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            total += float(loss_fn(model(xb), yb).item()) * xb.shape[0]
            n += xb.shape[0]
    return total / max(n, 1)


def evaluate(model, loader, refs, device):
    """Score a NON-shuffled loader (rows are matched to `refs` by position).

    Returns (y_soft, score, table); same contract as ref.collect_scores, but
    with blocking device copies for the MPS reason given in `eval_loss`.
    """
    model.eval()
    ys, scores = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pm = torch.sigmoid(model(xb.to(device)))
            scores.append(pm.float().cpu().numpy().mean(axis=(1, 2, 3)))
            ys.append(yb.numpy().mean(axis=(1, 2, 3)))
    y, score = np.concatenate(ys), np.concatenate(scores)
    assert len(score) == len(refs), (len(score), len(refs))
    table = pd.DataFrame(
        [dict(event=r[0], start=int(r[1]), target=float(r[2]), range_name=r[3],
              dt_from_on_ms=float(r[4]), score=float(score[i])) for i, r in enumerate(refs)]
    )
    return y, score, table


def summarize(val, test) -> dict:
    """val/test are (y, score, table) triples. Threshold comes from val only."""
    v_y, v_s, v_t = val
    t_y, t_s, t_t = test
    v_fpr, v_tpr, v_thr, v_auc = ref.compute_roc_from_scores(v_y, v_s)
    _, _, _, t_auc = ref.compute_roc_from_scores(t_y, t_s)
    thr = ref.choose_youden_threshold(v_fpr, v_tpr, v_thr)
    out = dict(val_auc=float(v_auc), test_auc=float(t_auc), threshold=float(thr))
    for k, v in ref.binary_metrics(t_y, t_s, thr).items():
        out[f"test_{k}"] = v
    for k, v in soft_metrics(v_y, v_s).items():
        out[f"val_{k}"] = v
    for k, v in soft_metrics(t_y, t_s).items():
        out[f"test_{k}"] = v
    for k, v in event_auc_summary(t_t).items():
        out[f"test_{k}"] = v
    return out
