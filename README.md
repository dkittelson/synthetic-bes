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
4. **`caffeinate -i` does not stop macOS "Maintenance Sleep".** During the overnight
   sweep the machine cycled through short scheduled sleep/wake periods (`pmset -g log`
   shows repeated `Entering Sleep state due to 'Maintenance Sleep'` entries) that
   `-i`, which only blocks *idle* sleep, does not prevent. The two runs that happened
   to span that window (`base_gn`, `base_k11`) show inflated wall time in their raw
   `total_s`, even though their per-epoch `train_s` (in `history.csv`) stayed normal
   and consistent throughout, meaning the compute itself was not corrupted, only the
   idle time got counted. `make_table.py` reports training time as the sum of
   per-epoch `train_s`, not the raw wall clock, for exactly this reason. Next time,
   use `caffeinate -s` (keeps the whole system awake on AC power) for unattended runs.
4. **Pipeline gate.** The pretrained checkpoint evaluated through this code gives
   val AUC 0.9440 / test AUC 0.9385, matching the mentor's notebook (0.944 / 0.939).
5. **Baseline retrained here scores higher than the pretrained checkpoint**
   (test AUC 0.994 vs 0.9385 on identical test windows). Checked for leakage: no
   event is duplicated across splits, and per-phase scores are sensible. Most likely
   the provided checkpoint is simply less trained. Worth asking the mentor.

Training setup: AdamW, lr 1e-3, weight decay 1e-5, batch 128, grad-clip 5,
up to 30 epochs, early stop patience 6 on val loss (mentor's script uses 40 / 8).
Window sampling seed is fixed so every variant sees identical windows; the seed
only changes initialisation and batch order. Timing is Apple M4 Pro, MPS,
single seed per variant (regenerate `results.md` for the current numbers).

## Results


| Model | Params | Params % | Test AUC ↑ | Δpp | Event AUC ↑ | Acc ↑ | F1 ↑ | Δpp | Brier ↓ | Δ% | LogLoss ↓ | R2 ↑ | Δ% | EV ↑ | Val loss ↓ | Best ep | Train min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xl | 462,081 | +664.4% | 0.9952 | +0.11 | 0.9956 | 0.9683 | 0.9679 | +0.09 | 0.0376 | -8.5% | 0.3541 | 0.7742 | +2.8% | 0.7748 | 0.3570 | 5 | 52.2 |
| s_xl | 393,121 | +550.3% | 0.9954 | +0.13 | 0.9966 | 0.9692 | 0.9690 | +0.21 | 0.0394 | -4.1% | 0.3587 | 0.7634 | +1.4% | 0.7639 | 0.3690 | 4 | 24.4 |
| big | 240,449 | +297.8% | 0.9950 | +0.09 | 0.9953 | 0.9692 | 0.9690 | +0.21 | 0.0393 | -4.6% | 0.3608 | 0.7644 | +1.5% | 0.7662 | 0.3664 | 7 | 61.9 |
| s_l | 189,921 | +214.2% | 0.9953 | +0.12 | 0.9960 | 0.9679 | 0.9680 | +0.10 | 0.0424 | +3.2% | 0.3651 | 0.7453 | -1.0% | 0.7495 | 0.3684 | 4 | 23.6 |
| t_l | 92,545 | +53.1% | 0.9955 | +0.15 | 0.9967 | 0.9664 | 0.9664 | -0.06 | 0.0425 | +3.2% | 0.3658 | 0.7452 | -1.1% | 0.7464 | 0.3746 | 3 | 31.9 |
| base_k11 | 63,041 | +4.3% | 0.9939 | -0.02 | 0.9941 | 0.9663 | 0.9664 | -0.05 | 0.0399 | -3.0% | 0.3634 | 0.7606 | +1.0% | 0.7615 | 0.3645 | 19 | 44.9 |
| base_gn | 60,449 | +0.0% | 0.9954 | +0.13 | 0.9963 | 0.9700 | 0.9700 | +0.31 | 0.0393 | -4.4% | 0.3583 | 0.7641 | +1.4% | 0.7651 | 0.3662 | 10 | 31.7 |
| baseline | 60,449 | +0.0% | 0.9941 | +0.00 | 0.9944 | 0.9669 | 0.9670 | +0.00 | 0.0411 | +0.0% | 0.3646 | 0.7532 | +0.0% | 0.7546 | 0.3640 | 7 | 15.4 |
| base_k3 | 57,857 | -4.3% | 0.9920 | -0.21 | 0.9920 | 0.9583 | 0.9586 | -0.84 | 0.0464 | +12.8% | 0.3810 | 0.7216 | -4.2% | 0.7263 | 0.3881 | 8 | 15.5 |
| t_xs | 47,761 | -21.0% | 0.9951 | +0.10 | 0.9954 | 0.9658 | 0.9659 | -0.11 | 0.0415 | +1.0% | 0.3650 | 0.7508 | -0.3% | 0.7515 | 0.3748 | 10 | 8.2 |
| s_s | 23,361 | -61.4% | 0.9957 | +0.16 | 0.9972 | 0.9685 | 0.9684 | +0.15 | 0.0408 | -0.8% | 0.3637 | 0.7552 | +0.3% | 0.7559 | 0.3661 | 6 | 14.1 |
| small | 15,281 | -74.7% | 0.9941 | +0.01 | 0.9941 | 0.9644 | 0.9647 | -0.23 | 0.0425 | +3.3% | 0.3680 | 0.7451 | -1.1% | 0.7465 | 0.3730 | 19 | 12.8 |
| s_xs | 11,729 | -80.6% | 0.9942 | +0.01 | 0.9964 | 0.9601 | 0.9598 | -0.72 | 0.0426 | +3.6% | 0.3681 | 0.7442 | -1.2% | 0.7443 | 0.3704 | 10 | 18.8 |
| tiny | 4,329 | -92.8% | 0.9911 | -0.30 | 0.9930 | 0.9547 | 0.9550 | -1.19 | 0.0470 | +14.3% | 0.3838 | 0.7179 | -4.7% | 0.7185 | 0.3867 | 12 | 7.7 |

*Results per model on the held-out test split, sorted by descending parameter count. AUC, accuracy and F1 exclude windows labelled 0.5 (positive = label >= 0.75) and use the Youden-J threshold chosen on validation. Brier, log loss, R2 and explained variance compare mean(sigmoid(2x2 map)) with the soft label (0, 0.5, 1) over all windows. Event AUC is the mean of per-event AUCs. Deltas are vs the baseline row: percentage points for AUC/F1, relative % otherwise. Single seed.*
