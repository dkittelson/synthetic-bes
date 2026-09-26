"""Train one (variant, seed) job. Idempotent: skips if metrics.json exists.

    python -m bes_sweep.train_one --variant baseline --seed 0
    python -m bes_sweep.train_one --variant tiny --max-events 5 --epochs 1   # smoke test

Everything a job produces lives in runs/<variant>/seed<K>/. metrics.json is
written LAST and atomically (tmp file + os.replace), so its existence means
"this job finished completely". That is what makes re-running a sweep safe.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import train_synthetic_bes_v3_1_shared as ref
from bes_sweep import data, metrics
from bes_sweep.configs import VARIANTS, build_model, count_params

ROOT = Path(__file__).resolve().parent.parent


def run_dir(runs_dir: Path, variant: str, seed: int) -> Path:
    return Path(runs_dir) / variant / f"seed{seed}"


def default_runs_dir(max_events) -> Path:
    # Smoke runs use a subset of events; keep them out of the real results.
    return ROOT / ("runs_smoke" if max_events else "runs")


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--window-stride", type=int, default=8)
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--data", type=Path, default=data.DEFAULT_H5)
    p.add_argument("--runs-dir", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    runs_dir = args.runs_dir or default_runs_dir(args.max_events)
    out = run_dir(runs_dir, args.variant, args.seed)
    if (out / "metrics.json").exists() and not args.force:
        print(f"[skip] {args.variant} seed{args.seed}: metrics.json exists")
        return
    out.mkdir(parents=True, exist_ok=True)

    ref.set_seed(args.seed)
    device = data.resolve_device(args.device)
    cfg = VARIANTS[args.variant]
    model = build_model(cfg).to(device)
    n_params = count_params(model)
    print(f"[{args.variant} seed{args.seed}] device={device} params={n_params:,}", flush=True)

    # Window sampling seed is fixed (not args.seed) so every variant and seed
    # sees identical windows; args.seed only changes init + batch order.
    loaders, refs = data.make_loaders(args.data, ref.RANDOM_SEED, args.batch_size,
                                      args.window_stride, args.max_events)

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val, best_state, best_epoch, bad = np.inf, None, 0, 0
    history = []
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        data.synchronize(device)
        t0 = time.perf_counter()
        model.train()
        loss_sum, n = 0.0, 0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]
        data.synchronize(device)
        train_s = time.perf_counter() - t0

        val_loss = metrics.eval_loss(model, loaders["val"], loss_fn, device)
        if not np.isfinite(val_loss):
            raise RuntimeError(f"non-finite val loss at epoch {epoch}: {val_loss}")
        if val_loss < best_val:
            best_val, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        history.append(dict(epoch=epoch, train_loss=loss_sum / n, val_loss=val_loss,
                            best_val_loss=best_val, train_s=train_s))
        print(f"  epoch {epoch:02d} train={loss_sum/n:.4f} val={val_loss:.4f} "
              f"best={best_val:.4f} bad={bad}/{args.patience} ({train_s:.0f}s)", flush=True)
        if bad >= args.patience:
            print("  early stop", flush=True)
            break
    total_s = time.perf_counter() - t_start

    model.load_state_dict(best_state)
    val = metrics.evaluate(model, loaders["val"], refs["val"], device)
    test = metrics.evaluate(model, loaders["test"], refs["test"], device)
    summary = metrics.summarize(val, test)

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    test[2].to_csv(out / "test_scores.csv", index=False)
    torch.save(best_state, out / "model.pt")
    result = dict(
        variant=args.variant, seed=args.seed, params=n_params,
        epochs_run=len(history), best_epoch=best_epoch, best_val_loss=float(best_val),
        train_s_per_epoch=float(np.mean([h["train_s"] for h in history])),
        total_s=total_s, device=str(device), max_events=args.max_events,
        config=dict(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                    lr=args.lr, weight_decay=args.weight_decay),
        **summary,
    )
    atomic_write_json(out / "metrics.json", result)
    print(f"[done] {args.variant} seed{args.seed}: test_auc={summary['test_auc']:.4f} "
          f"r2={summary['test_r2']:.4f} in {total_s/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
