# Synthetic BES: model-size sweep

Reproduces a "training results per model" table (params, params %, metrics with
deltas vs baseline) for the ELM-forecast network in
`train_synthetic_bes_v3_1_shared.py`, on `synthetic_bes_v3_1_shared.h5`.
The mentor's script is left untouched and imported as a library.

## Run it

```bash
pip install -r requirements_synthetic_bes_v3_1_shared.txt scikit-learn pytest
python -m pytest tests -q                                   # 23 tests
python -m bes_sweep.eval_checkpoint synthetic_bes_v3_1_pretrained_portable.pt   # pipeline gate
python -m bes_sweep.train_one --variant tiny --max-events 5 --epochs 1          # smoke test
python -m bes_sweep.train_one --variant baseline --seed 0                       # one job
python -m bes_sweep.sweep --seeds 0                         # all variants (resumable)
python -m bes_sweep.make_table                              # -> results.{md,csv,tex}
```

Run from this folder (the modules import the mentor's script by name).

## Layout

| file | role |
|---|---|
| `bes_sweep/configs.py` | the 14 variants (`VARIANTS`), model builder, BN to GroupNorm swap |
| `bes_sweep/data.py` | decode each split into RAM once; reuse the mentor's window indexing |
| `bes_sweep/metrics.py` | AUC/F1 (mentor's convention) plus Brier, log loss, R2, EV, event AUC |
| `bes_sweep/train_one.py` | one (variant, seed) job to `runs/<variant>/seed<K>/`; skips if done |
| `bes_sweep/sweep.py` | loop over jobs, one subprocess each; safe to re-run after interruption |
| `bes_sweep/make_table.py` | reduce all `metrics.json` files to the table |

A job writes `metrics.json` last and atomically, so its existence means the job
finished. That makes the sweep resumable and the table a pure function of `runs/`.

## Metric conventions

- **AUC, accuracy, F1**: windows labelled 0.5 are dropped, label >= 0.75 is
  positive, threshold = Youden-J optimum chosen on **val** only (mentor's convention).
- **Brier, log loss, R2, explained variance**: score = mean(sigmoid(2x2 map))
  against the **soft** label (0 / 0.5 / 1), all windows kept. On a fixed set
  R2 = 1 - Brier / Var(y), so they carry the same information.
- **Event AUC**: mean of per-event AUCs on test.
- Deltas vs baseline: percentage points for AUC/F1, relative % otherwise.

**Open question for the mentor:** is R2 meant to be computed this way (score vs
soft label)? It is a one-line change in `metrics.summarize`.

## Findings worth knowing

1. **Data loading was the bottleneck.** The mentor's per-window gzip HDF5 loader
   with a 16-event cache costs 379 ms/batch on shuffled train batches (about 6.6
   min/epoch of pure loading). Preloading the decoded split (1.2 GB, 2 s to
   decode) costs 1.1 ms/batch, a 330x speedup. Reproduce with `python -m bes_sweep.data`.
2. **`non_blocking=True` is unsafe on MPS.** The mentor's `evaluate_loss` and
   `collect_scores` copy batches with `non_blocking=True`. On Apple MPS this
   intermittently produced NaN/-inf validation loss for an identical model and
   data (verified over repeated runs). `bes_sweep/metrics.py` uses blocking
   copies; `tests/test_eval_stability.py` guards it. Fine on CUDA with pinned memory.
3. **Parallel jobs on one Mac GPU are not worth it.** Two jobs at once gave 1.14x
   aggregate throughput but slowed each job's epoch by 25-45%, which would corrupt
   the train-time column. The sweep runs one job at a time.
4. **Pipeline gate.** The pretrained checkpoint evaluated through this code gives
   val AUC 0.9440 / test AUC 0.9385, matching the mentor's notebook (0.944 / 0.939).
5. **Baseline retrained here scores higher than the pretrained checkpoint**
   (test AUC 0.994 vs 0.9385 on identical test windows). Checked for leakage: no
   event is duplicated across splits, and per-phase scores are sensible. Most likely
   the provided checkpoint is simply less trained. Worth asking the mentor.

Training setup: AdamW, lr 1e-3, weight decay 1e-5, batch 128, grad-clip 5,
up to 30 epochs, early stop patience 6 on val loss (mentor's script uses 40 / 8).
Window sampling seed is fixed so every variant sees identical windows; the seed
only changes initialisation and batch order. Timing is Apple M4 Pro, MPS.

## Results

See `results.md` (generated).
