"""Data loading for the sweep.

The mentor's `SharedWindowDataset` reads gzip-compressed HDF5 per window
through a 16-event LRU cache. With shuffled access that cache misses
constantly and the epoch is dominated by decompression. The decoded train
split is only ~1.2 GB, so we decode every event once into RAM and slice
windows from the in-memory arrays.

Window indexing (`ref.build_window_refs`) is reused unchanged so the exact
same windows and soft labels are used as in the mentor's trainer.
"""

from __future__ import annotations

import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import train_synthetic_bes_v3_1_shared as ref

HERE = Path(__file__).resolve().parent.parent
DEFAULT_H5 = HERE / "synthetic_bes_v3_1_shared.h5"
SPLITS = ("train", "val", "test")


def resolve_device(name: str = "auto") -> torch.device:
    """cuda -> mps -> cpu. The mentor's version only knows cuda/cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Block until queued GPU work finishes, so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def event_names(h5_path: Path, split: str, max_events: int | None = None) -> list[str]:
    with h5py.File(h5_path, "r") as f:
        names = sorted(f[split].keys())
    return names[:max_events] if max_events else names


def load_split(h5_path: Path, split: str, max_events: int | None = None) -> dict[str, np.ndarray]:
    """Decode every event of a split into RAM: {event_name: float32 (T, 8, 8)}."""
    names = event_names(h5_path, split, max_events)
    events = {}
    with h5py.File(h5_path, "r") as f:
        for name in names:
            events[name] = np.asarray(f[split][name]["input"], dtype=np.float32)
    return events


def build_refs(h5_path: Path, split: str, seed: int, window_stride: int = 8,
               max_events: int | None = None) -> list[tuple]:
    """Mentor's window index, optionally restricted to the first N sorted events.

    `ref.build_window_refs` walks events in sorted order and draws from one RNG,
    so the first N events' windows are identical whether or not the rest are
    present. That makes `--max-events` a faithful subset for smoke tests.
    """
    with h5py.File(h5_path, "r") as f:
        sample_dt_us = float(f.attrs.get("sample_dt_us", 1.0))
    refs = ref.build_window_refs(h5_path, split, sample_dt_us, window_stride, seed)
    if max_events:
        keep = set(event_names(h5_path, split, max_events))
        refs = [r for r in refs if r[0] in keep]
    return refs


class PreloadedWindowDataset(Dataset):
    """Same (x, y) contract as ref.SharedWindowDataset, but slices from RAM."""

    def __init__(self, events: dict[str, np.ndarray], refs: list[tuple]):
        self.events = events
        self.refs = list(refs)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, idx: int):
        event_name, start, target, _range_name, _dt = self.refs[idx]
        x = self.events[event_name][start:start + ref.WINDOW_SIZE]
        y = np.full((1, 2, 2), target, dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y)


def make_loaders(h5_path: Path, seed: int, batch_size: int, window_stride: int = 8,
                 max_events: int | None = None, splits=SPLITS):
    """Returns (loaders, refs). Val/test are never shuffled: score tables are
    joined back to `refs` by position."""
    loaders, refs = {}, {}
    for split in splits:
        refs[split] = build_refs(h5_path, split, seed, window_stride, max_events)
        ds = PreloadedWindowDataset(load_split(h5_path, split, max_events), refs[split])
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=0, pin_memory=False,
        )
    return loaders, refs


def benchmark_loaders(h5_path: Path = DEFAULT_H5, n_batches: int = 200, batch_size: int = 128):
    """Time the mentor's HDF5-backed dataset vs the preloaded one on shuffled train batches."""
    refs = build_refs(h5_path, "train", ref.RANDOM_SEED)
    results = {}

    t0 = time.perf_counter()
    events = load_split(h5_path, "train")
    results["preload_decode_s"] = time.perf_counter() - t0
    results["preload_gb"] = sum(v.nbytes for v in events.values()) / 1e9

    for label, ds in (
        ("h5_lru16", ref.SharedWindowDataset(h5_path, "train", refs, cache_events=16)),
        ("preloaded", PreloadedWindowDataset(events, refs)),
    ):
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        it = iter(loader)
        t0 = time.perf_counter()
        for _ in range(n_batches):
            next(it)
        results[f"{label}_s_per_batch"] = (time.perf_counter() - t0) / n_batches
    return results


if __name__ == "__main__":
    r = benchmark_loaders()
    print(f"decode train split into RAM: {r['preload_decode_s']:.1f}s ({r['preload_gb']:.2f} GB)")
    print(f"mentor h5 loader : {r['h5_lru16_s_per_batch']*1000:.1f} ms/batch "
          f"-> {r['h5_lru16_s_per_batch']*1050/60:.1f} min/epoch of loading")
    print(f"preloaded loader : {r['preloaded_s_per_batch']*1000:.1f} ms/batch "
          f"-> {r['preloaded_s_per_batch']*1050/60:.1f} min/epoch of loading")
    print(f"speedup          : {r['h5_lru16_s_per_batch']/r['preloaded_s_per_batch']:.0f}x")
