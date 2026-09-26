#!/usr/bin/env python3
"""
Portable single-process/single-GPU trainer for synthetic_bes_v3_1_shared.h5.

No absolute path is hard-coded. By default this script expects the shared H5
to be in the same directory as this script.

Run:
    python train_synthetic_bes_v3_1_shared.py

Optional:
    python train_synthetic_bes_v3_1_shared.py --data my_shared_file.h5 --epochs 40
"""

from pathlib import Path
from collections import OrderedDict
from contextlib import nullcontext
import argparse
import json
import random
import time

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, total=None, desc=None, leave=True, **kwargs):
        return iterable if iterable is not None else range(total or 0)


HERE = Path(__file__).resolve().parent

GRID_SHAPE = (8, 8)
WINDOW_SIZE = 320
RANDOM_SEED = 1234

LABEL_RANGES = [
    dict(name="far_pre",       reference="onset", dt0_ms=-3.7, dt1_ms=-3.1, target=0.0, max_windows=32),
    dict(name="early_pre_elm", reference="onset", dt0_ms=-2.8, dt1_ms=-2.2, target=0.5, max_windows=64),
    dict(name="pre_elm",       reference="onset", dt0_ms=-1.8, dt1_ms=-0.5, target=1.0, max_windows=64),
    dict(name="recovery",      reference="end",   dt0_ms=+0.5, dt1_ms=+1.4, target=0.0, max_windows=32),
]

MODEL_KWARGS = dict(
    input_time=WINDOW_SIZE,
    grid_shape=GRID_SHAPE,
    temporal_channels=(8, 16, 32),
    spatial_channels=(64, 64),
    temporal_kernel_size=7,
    dropout=0.10,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=HERE / "synthetic_bes_v3_1_shared.h5")
    p.add_argument("--checkpoint", type=Path, default=HERE / "synthetic_bes_v3_1_shared_model.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--window-stride", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cache-events", type=int, default=16)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or e.g. 'cuda:0'")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TemporalSpatialELMMapNet(nn.Module):
    def __init__(
        self,
        input_time=320,
        grid_shape=(8, 8),
        temporal_channels=(8, 16, 32),
        spatial_channels=(64, 64),
        temporal_kernel_size=7,
        dropout=0.10,
    ):
        super().__init__()
        self.input_time = int(input_time)
        self.grid_shape = tuple(grid_shape)

        t_layers = []
        in_ch = 1
        pad = temporal_kernel_size // 2
        for out_ch in temporal_channels:
            t_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=temporal_kernel_size, padding=pad),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ]
            if dropout > 0:
                t_layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.temporal_net = nn.Sequential(*t_layers)

        s_layers = []
        in_ch = temporal_channels[-1]
        for out_ch in spatial_channels:
            s_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            if dropout > 0:
                s_layers.append(nn.Dropout2d(dropout))
            in_ch = out_ch
        self.spatial_net = nn.Sequential(*s_layers)
        self.out_head = nn.Conv2d(spatial_channels[-1], 1, kernel_size=1)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected (B,T,H,W), got {tuple(x.shape)}")
        B, T, H, W = x.shape
        if T != self.input_time or (H, W) != self.grid_shape:
            raise ValueError(
                f"Expected (B,{self.input_time},{self.grid_shape[0]},{self.grid_shape[1]}), "
                f"got {tuple(x.shape)}"
            )
        z = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, 1, T)
        z = self.temporal_net(z).mean(dim=-1)
        Ct = z.shape[1]
        z = z.view(B, H, W, Ct).permute(0, 3, 1, 2).contiguous()
        z = self.spatial_net(z)
        return self.out_head(z)


def detect_elm_bounds(target):
    y = np.asarray(target)
    idx = np.flatnonzero(y > 0)
    if len(idx) == 0:
        raise ValueError("No actual ELM samples in event target.")
    onset = int(idx[0])
    end = int(idx[-1] + 1)
    return onset, end


def ms_to_samples(dt_ms, sample_dt_us):
    return float(dt_ms) * 1000.0 / float(sample_dt_us)


def candidate_centers_from_target(
    target,
    reference_index,
    dt0_ms,
    dt1_ms,
    sample_dt_us,
    window_stride,
):
    T = len(target)
    half_left = WINDOW_SIZE // 2
    half_right = WINDOW_SIZE - half_left

    a = reference_index + ms_to_samples(dt0_ms, sample_dt_us)
    b = reference_index + ms_to_samples(dt1_ms, sample_dt_us)
    lo = int(np.ceil(min(a, b) - 1e-12))
    hi = int(np.floor(max(a, b) + 1e-12))

    lo = max(lo, half_left)
    hi = min(hi, T - half_right)

    if hi < lo:
        return np.asarray([], dtype=np.int64)

    return np.arange(lo, hi + 1, int(window_stride), dtype=np.int64)


def build_window_refs(h5_path, split, sample_dt_us, window_stride, seed):
    refs = []
    rng = np.random.default_rng(seed + {"train": 0, "val": 1, "test": 2}[split])

    with h5py.File(h5_path, "r") as f:
        event_names = sorted(f[split].keys())
        for event_name in tqdm(event_names, desc=f"index {split}"):
            y = np.asarray(f[split][event_name]["target"])
            onset, end = detect_elm_bounds(y)

            for spec in LABEL_RANGES:
                ref_idx = onset if spec["reference"] == "onset" else end
                centers = candidate_centers_from_target(
                    y,
                    ref_idx,
                    spec["dt0_ms"],
                    spec["dt1_ms"],
                    sample_dt_us,
                    window_stride,
                )

                maxw = spec.get("max_windows")
                if maxw is not None and len(centers) > int(maxw):
                    centers = np.sort(
                        rng.choice(centers, size=int(maxw), replace=False)
                    )

                for c in centers:
                    start = int(c - WINDOW_SIZE // 2)
                    refs.append(
                        (
                            event_name,
                            start,
                            float(spec["target"]),
                            str(spec["name"]),
                            float((c - onset) * sample_dt_us / 1000.0),
                        )
                    )

    return refs


class SharedWindowDataset(Dataset):
    def __init__(self, h5_path, split, refs, cache_events=16):
        self.h5_path = str(Path(h5_path).resolve())
        self.split = str(split)
        self.refs = list(refs)
        self.cache_events = int(cache_events)
        self._cache = OrderedDict()
        self._h5 = None

    def __len__(self):
        return len(self.refs)

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _event_input(self, event_name):
        if event_name in self._cache:
            x = self._cache.pop(event_name)
            self._cache[event_name] = x
            return x

        x = np.asarray(
            self._file()[self.split][event_name]["input"],
            dtype=np.float32,
        )
        self._cache[event_name] = x
        while len(self._cache) > self.cache_events:
            self._cache.popitem(last=False)
        return x

    def __getitem__(self, idx):
        event_name, start, target, range_name, dt_from_on_ms = self.refs[int(idx)]
        x_event = self._event_input(event_name)
        x = np.asarray(
            x_event[start:start + WINDOW_SIZE],
            dtype=np.float32,
        )
        y = np.full((1, 2, 2), target, dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass


def auc_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def compute_roc_from_scores(y_cont, scores):
    y_cont = np.asarray(y_cont, dtype=float)
    scores = np.asarray(scores, dtype=float)
    mask = (
        np.isfinite(y_cont)
        & np.isfinite(scores)
        & (~np.isclose(y_cont, 0.5))
    )
    y = (y_cont[mask] >= 0.75).astype(int)
    s = scores[mask]

    if len(np.unique(y)) < 2:
        raise RuntimeError("ROC requires positive and negative examples.")

    thresholds = np.r_[np.inf, np.sort(np.unique(s))[::-1], -np.inf]
    P = max(int(np.sum(y == 1)), 1)
    N = max(int(np.sum(y == 0)), 1)
    tpr, fpr = [], []

    for th in thresholds:
        pred = s >= th
        tpr.append(np.sum(pred & (y == 1)) / P)
        fpr.append(np.sum(pred & (y == 0)) / N)

    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    order = np.argsort(fpr, kind="stable")
    return (
        fpr[order],
        tpr[order],
        thresholds[order],
        auc_trapezoid(tpr[order], fpr[order]),
    )


def choose_youden_threshold(fpr, tpr, thresholds):
    finite = np.isfinite(thresholds)
    idxs = np.flatnonzero(finite)
    if len(idxs) == 0:
        return 0.5
    j = tpr[idxs] - fpr[idxs]
    return float(thresholds[idxs[int(np.argmax(j))]])


def binary_metrics(y_cont, scores, threshold):
    y_cont = np.asarray(y_cont, dtype=float)
    scores = np.asarray(scores, dtype=float)
    mask = (
        np.isfinite(y_cont)
        & np.isfinite(scores)
        & (~np.isclose(y_cont, 0.5))
    )
    y = (y_cont[mask] >= 0.75).astype(int)
    pred = (scores[mask] >= threshold).astype(int)

    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))

    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return dict(
        threshold=float(threshold),
        tp=tp, tn=tn, fp=fp, fn=fn,
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        specificity=float(specificity),
        f1=float(f1),
    )


def resolve_device(s):
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def autocast_context(device, amp_enabled):
    if not amp_enabled or device.type != "cuda":
        return nullcontext()
    if torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def evaluate_loss(model, loader, loss_fn, device, amp_enabled):
    model.eval()
    loss_sum = 0.0
    n = 0
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device, dtype=torch.float32, non_blocking=True)
            yb = yb.to(device, dtype=torch.float32, non_blocking=True)
            with autocast_context(device, amp_enabled):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            loss_sum += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]
    return loss_sum / max(n, 1)


def collect_scores(model, loader, refs, device, amp_enabled):
    model.eval()
    ys, scores = [], []
    offset = 0
    rows = []

    with torch.inference_mode():
        for xb, yb in tqdm(loader, desc="scoring"):
            xb = xb.to(device, dtype=torch.float32, non_blocking=True)
            with autocast_context(device, amp_enabled):
                pm = torch.sigmoid(model(xb))
            score = pm.float().cpu().numpy().mean(axis=(1, 2, 3))
            y = yb.numpy().mean(axis=(1, 2, 3))

            ys.append(y)
            scores.append(score)

            for i in range(len(score)):
                event_name, start, target, range_name, dt_from_on_ms = refs[offset + i]
                rows.append(
                    dict(
                        event=event_name,
                        start=int(start),
                        target=float(target),
                        range_name=range_name,
                        dt_from_on_ms=float(dt_from_on_ms),
                        score=float(score[i]),
                    )
                )
            offset += len(score)

    return np.concatenate(ys), np.concatenate(scores), pd.DataFrame(rows)


def main():
    args = parse_args()
    args.data = args.data.resolve()
    args.checkpoint = args.checkpoint.resolve()
    set_seed(args.seed)

    if not args.data.exists():
        raise FileNotFoundError(args.data)

    device = resolve_device(args.device)
    amp_enabled = (not args.no_amp) and device.type == "cuda"

    with h5py.File(args.data, "r") as f:
        sample_dt_us = float(f.attrs.get("sample_dt_us", 1.0))
        preprocess_mode = str(f.attrs.get("preprocess_mode", "raw"))
        split_counts = {
            s: len(f[s].keys())
            for s in ("train", "val", "test")
        }

    refs = {
        split: build_window_refs(
            args.data,
            split,
            sample_dt_us,
            args.window_stride,
            args.seed,
        )
        for split in ("train", "val", "test")
    }

    datasets = {
        split: SharedWindowDataset(
            args.data,
            split,
            refs[split],
            cache_events=args.cache_events,
        )
        for split in ("train", "val", "test")
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        ),
    }

    print("=" * 80)
    print("Portable Synthetic BES V3.1 training")
    print("Data      :", args.data)
    print("Device    :", device)
    print("Events    :", split_counts)
    print("Windows   :", {k: len(v) for k, v in refs.items()})
    print("Batch     :", args.batch_size)
    print("AMP BF16 :", amp_enabled and device.type == "cuda" and torch.cuda.is_bf16_supported())
    print("=" * 80)

    model = TemporalSpatialELMMapNet(**MODEL_KWARGS).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = np.inf
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
        n = 0

        for xb, yb in tqdm(loaders["train"], desc=f"epoch {epoch:03d} train"):
            xb = xb.to(device, dtype=torch.float32, non_blocking=True)
            yb = yb.to(device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                logits = model(xb)
                loss = loss_fn(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            loss_sum += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]

        train_loss = loss_sum / max(n, 1)
        val_loss = evaluate_loss(
            model, loaders["val"], loss_fn, device, amp_enabled
        )

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        history.append(
            dict(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_loss=best_val,
                elapsed_s=time.time() - t0,
            )
        )

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train={train_loss:.6e} val={val_loss:.6e} "
            f"best={best_val:.6e} bad_epochs={bad_epochs}/{args.patience}"
        )

        if bad_epochs >= args.patience:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("No trained state was produced.")

    model.load_state_dict(best_state)
    model.to(device)

    val_y, val_score, val_table = collect_scores(
        model, loaders["val"], refs["val"], device, amp_enabled
    )
    test_y, test_score, test_table = collect_scores(
        model, loaders["test"], refs["test"], device, amp_enabled
    )

    val_fpr, val_tpr, val_thr, val_auc = compute_roc_from_scores(
        val_y, val_score
    )
    test_fpr, test_tpr, test_thr, test_auc = compute_roc_from_scores(
        test_y, test_score
    )
    decision_threshold = choose_youden_threshold(
        val_fpr, val_tpr, val_thr
    )

    val_metrics = binary_metrics(
        val_y, val_score, decision_threshold
    )
    test_metrics = binary_metrics(
        test_y, test_score, decision_threshold
    )

    checkpoint = {
        "model_state_dict": best_state,
        "config": {
            "checkpoint_type": "portable_shared_h5_training",
            "model_kwargs": MODEL_KWARGS,
            "input_time": WINDOW_SIZE,
            "grid_shape": GRID_SHAPE,
            "sample_dt_us": sample_dt_us,
            "fs_hz": 1e6 / sample_dt_us,
            "preprocess_mode": preprocess_mode,
            "label_ranges": LABEL_RANGES,
            "window_stride": args.window_stride,
            "decision_threshold": float(decision_threshold),
            "target_map_convention": "scalar forecast target broadcast to 2x2 map",
            "roc_score_convention": "mean(sigmoid(2x2 map)); target=0.5 excluded",
        },
        "best_val_loss": float(best_val),
        "history": history,
    }
    torch.save(checkpoint, args.checkpoint)

    history_df = pd.DataFrame(history)
    history_df.to_csv(HERE / "history_synthetic_bes_v3_1_shared.csv", index=False)

    val_table.to_csv(
        HERE / "validation_window_scores_synthetic_bes_v3_1_shared.csv",
        index=False,
    )
    test_table["predicted_positive"] = (
        test_table["score"] >= decision_threshold
    )
    test_table.to_csv(
        HERE / "test_window_scores_synthetic_bes_v3_1_shared.csv",
        index=False,
    )

    metrics = {
        "validation_auc": float(val_auc),
        "test_auc": float(test_auc),
        "decision_threshold": float(decision_threshold),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "n_events": split_counts,
        "n_windows": {k: len(v) for k, v in refs.items()},
    }
    (HERE / "metrics_synthetic_bes_v3_1_shared.json").write_text(
        json.dumps(metrics, indent=2)
    )

    np.savez_compressed(
        HERE / "roc_synthetic_bes_v3_1_shared.npz",
        val_y=val_y,
        val_score=val_score,
        val_fpr=val_fpr,
        val_tpr=val_tpr,
        val_thresholds=val_thr,
        val_auc=np.asarray(val_auc),
        test_y=test_y,
        test_score=test_score,
        test_fpr=test_fpr,
        test_tpr=test_tpr,
        test_thresholds=test_thr,
        test_auc=np.asarray(test_auc),
        decision_threshold=np.asarray(decision_threshold),
    )

    plt.figure(figsize=(6, 5.5))
    plt.plot(val_fpr, val_tpr, lw=2, label=f"validation AUC={val_auc:.3f}")
    plt.plot(test_fpr, test_tpr, lw=2, label=f"test AUC={test_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(HERE / "roc_synthetic_bes_v3_1_shared.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="train")
    plt.plot(history_df["epoch"], history_df["val_loss"], marker="o", label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("BCEWithLogitsLoss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(HERE / "history_synthetic_bes_v3_1_shared.png", dpi=180)
    plt.close()

    print("\nTraining complete.")
    print("Checkpoint:", args.checkpoint)
    print(f"Validation AUC={val_auc:.4f}")
    print(f"Test AUC={test_auc:.4f}")
    print(f"Decision threshold={decision_threshold:.4f}")


if __name__ == "__main__":
    main()
