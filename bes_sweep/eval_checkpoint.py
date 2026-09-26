"""Evaluate any checkpoint with the sweep's metric set (no training).

    python -m bes_sweep.eval_checkpoint synthetic_bes_v3_1_pretrained_portable.pt

Used as a gate: the pretrained baseline should give val AUC ~0.944 and test
AUC ~0.939 (numbers from the mentor's notebook). If this matches, data.py and
metrics.py are trustworthy before any training run.
"""

import argparse
import json
from pathlib import Path

import torch

import train_synthetic_bes_v3_1_shared as ref
from bes_sweep import data, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--data", type=Path, default=data.DEFAULT_H5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-events", type=int, default=None)
    args = p.parse_args()

    device = data.resolve_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model_kwargs = cfg.get("model_kwargs", ref.MODEL_KWARGS)
    model = ref.TemporalSpatialELMMapNet(**model_kwargs)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.to(device).eval()

    loaders, refs = data.make_loaders(args.data, ref.RANDOM_SEED, args.batch_size,
                                      max_events=args.max_events, splits=("val", "test"))
    val = metrics.evaluate(model, loaders["val"], refs["val"], device)
    test = metrics.evaluate(model, loaders["test"], refs["test"], device)
    print(json.dumps(metrics.summarize(val, test), indent=2))


if __name__ == "__main__":
    main()
