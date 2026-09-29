| Model | Params | Params % | Test AUC ↑ | Δpp | R2 ↑ | Δ% | Train min |
|---|---|---|---|---|---|---|---|
| xl | 462,081 | +664.4% | 0.9952 | +0.11 | 0.7742 | +2.8% | 52.2 |
| s_xl | 393,121 | +550.3% | 0.9954 | +0.13 | 0.7634 | +1.4% | 24.4 |
| big | 240,449 | +297.8% | 0.9950 | +0.09 | 0.7644 | +1.5% | 61.9 |
| s_l | 189,921 | +214.2% | 0.9953 | +0.12 | 0.7453 | -1.0% | 23.6 |
| t_l | 92,545 | +53.1% | 0.9955 | +0.15 | 0.7452 | -1.1% | 31.9 |
| base_k11 | 63,041 | +4.3% | 0.9939 | -0.02 | 0.7606 | +1.0% | 44.9 |
| base_gn | 60,449 | +0.0% | 0.9954 | +0.13 | 0.7641 | +1.4% | 31.7 |
| baseline | 60,449 | +0.0% | 0.9941 | +0.00 | 0.7532 | +0.0% | 15.4 |
| base_k3 | 57,857 | -4.3% | 0.9920 | -0.21 | 0.7216 | -4.2% | 15.5 |
| t_xs | 47,761 | -21.0% | 0.9951 | +0.10 | 0.7508 | -0.3% | 8.2 |
| s_s | 23,361 | -61.4% | 0.9957 | +0.16 | 0.7552 | +0.3% | 14.1 |
| small | 15,281 | -74.7% | 0.9941 | +0.01 | 0.7451 | -1.1% | 12.8 |
| s_xs | 11,729 | -80.6% | 0.9942 | +0.01 | 0.7442 | -1.2% | 18.8 |
| tiny | 4,329 | -92.8% | 0.9911 | -0.30 | 0.7179 | -4.7% | 7.7 |

*Results per model on the held-out test split, sorted by descending parameter count. Test AUC excludes windows labelled 0.5 (positive = label >= 0.75) and uses the Youden-J threshold chosen on validation. R2 compares mean(sigmoid(2x2 map)) with the soft label (0, 0.5, 1) over all windows, including the 0.5 ones. Full metrics (accuracy, F1, Brier, log loss, explained variance, event AUC) are in each run's runs/<model>/seed0/metrics.json. Deltas are vs the baseline row: percentage points for AUC, relative % for R2. Single seed.*
