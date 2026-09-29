"""Reduce runs/*/seed*/metrics.json to the results table.

    python -m bes_sweep.make_table            # writes results.{csv,md,tex}
    python -m bes_sweep.make_table --runs-dir runs_smoke --out-dir /tmp/x

Deltas are vs the `baseline` row's mean:
  * AUC / F1 / accuracy : percentage points (relative % explodes near 0 and is
    hard to read for bounded scores).
  * everything else     : relative %.
Arrows in the header say which direction is better; deltas are signed raw
values, never sign-flipped.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bes_sweep.configs import BASELINE
from bes_sweep.train_one import ROOT

# key in metrics.json -> (header, higher_is_better, decimals)
# Kept deliberately short: Test AUC is the headline ranking metric and R2 is
# the headline calibration metric, and the two tell different parts of the
# story (AUC is nearly flat across model sizes; R2 is where size matters).
# The full metric set (accuracy, F1, Brier, log loss, event AUC, ...) is
# still in every run's metrics.json for anyone who wants it.
COLS = {
    "test_auc": ("Test AUC", True, 4),
    "test_r2": ("R2", True, 4),
}
PP_COLS = {"test_auc"}  # delta in percentage points (relative % is unreadable near 1.0)
DELTA_OF = ["test_auc", "test_r2"]  # which columns get a delta column


def load_runs(runs_dir: Path) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(Path(runs_dir).glob("*/seed*/metrics.json"))]
    if not rows:
        raise SystemExit(f"no metrics.json under {runs_dir}")
    return pd.DataFrame(rows)


def honest_train_minutes(runs_dir: Path, variant: str, seed) -> float:
    """Sum of per-epoch train_s from history.csv, NOT metrics.json's total_s.

    On an unattended overnight sweep, macOS can enter periodic "Maintenance
    Sleep" that `caffeinate -i` does not block (that flag only prevents idle
    sleep from inactivity). Two runs in this sweep spanned that window and
    each sleep/wake cycle added minutes of wall time to total_s without any
    extra epochs or slower per-epoch compute. Per-epoch train_s stayed stable
    throughout, so summing it is the honest number; total_s is not.
    """
    hist = Path(runs_dir) / variant / f"seed{seed}" / "history.csv"
    return pd.read_csv(hist)["train_s"].sum() / 60.0


def aggregate(df: pd.DataFrame, runs_dir: Path) -> pd.DataFrame:
    """One row per variant: mean over seeds, plus std and seed count."""
    df = df.copy()
    df["train_min"] = [honest_train_minutes(runs_dir, r.variant, r.seed) for r in df.itertuples()]
    keys = list(COLS) + ["params", "train_min"]
    g = df.groupby("variant")
    mean, std = g[keys].mean(), g[keys].std(ddof=1)
    out = mean.copy()
    for k in keys:
        out[k + "_std"] = std[k]
    out["n_seeds"] = g.size()
    return out.reset_index().sort_values("params", ascending=False).reset_index(drop=True)


def build(agg: pd.DataFrame):
    base = agg[agg.variant == BASELINE]
    base = base.iloc[0] if len(base) else None
    header = ["Model", "Params", "Params %"]
    for k in COLS:
        header.append(COLS[k][0] + (" ↑" if COLS[k][1] else " ↓"))
        if k in DELTA_OF:
            header.append("Δpp" if k in PP_COLS else "Δ%")
    header += ["Train min"]

    rows = []
    for _, r in agg.iterrows():
        cells = [r.variant, f"{int(r.params):,}"]
        cells.append("+0.0%" if base is None else f"{(r.params / base.params - 1) * 100:+.1f}%")
        for k, (_, _, d) in COLS.items():
            multi = r.n_seeds > 1 and not np.isnan(r[k + "_std"])
            cells.append(f"{r[k]:.{d}f}" + (f" ± {r[k + '_std']:.{d}f}" if multi else ""))
            if k in DELTA_OF:
                if base is None:
                    cells.append("")
                elif k in PP_COLS:
                    cells.append(f"{(r[k] - base[k]) * 100:+.2f}")
                else:
                    cells.append(f"{(r[k] / base[k] - 1) * 100:+.1f}%")
        cells += [f"{r.train_min:.1f}"]
        rows.append(cells)
    return header, rows


def to_markdown(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def to_latex(header, rows, caption) -> str:
    esc = lambda s: s.replace("%", r"\%").replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$") \
        .replace("Δ", r"$\Delta$").replace("±", r"$\pm$").replace("_", r"\_").replace("R2", "$R^2$")
    body = "\n".join(" & ".join(esc(c) for c in r) + r" \\" for r in rows)
    return (
        "\\begin{table}[t]\n\\centering\\small\n"
        f"\\begin{{tabular}}{{l{'r' * (len(header) - 1)}}}\n\\hline\n"
        + " & ".join(esc(h) for h in header) + " \\\\\n\\hline\n" + body + "\n\\hline\n"
        f"\\end{{tabular}}\n\\caption{{{caption}}}\n\\end{{table}}\n"
    )


def caption(agg: pd.DataFrame) -> str:
    seeds = int(agg.n_seeds.max())
    return (
        "Results per model on the held-out test split, sorted by descending parameter count. "
        "Test AUC excludes windows labelled 0.5 (positive = label >= 0.75) and uses the Youden-J "
        "threshold chosen on validation. R2 compares mean(sigmoid(2x2 map)) with the soft label "
        "(0, 0.5, 1) over all windows, including the 0.5 ones. Full metrics (accuracy, F1, Brier, "
        "log loss, explained variance, event AUC) are in each run's runs/<model>/seed0/metrics.json. "
        "Deltas are vs the baseline row: percentage points for AUC, relative % for R2. "
        + (f"Values are mean ± std over up to {seeds} seeds." if seeds > 1 else "Single seed.")
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    p.add_argument("--out-dir", type=Path, default=ROOT)
    args = p.parse_args()

    agg = aggregate(load_runs(args.runs_dir), args.runs_dir)
    header, rows = build(agg)
    cap = caption(agg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=header).to_csv(args.out_dir / "results.csv", index=False)
    (args.out_dir / "results.md").write_text(to_markdown(header, rows) + f"\n\n*{cap}*\n")
    (args.out_dir / "results.tex").write_text(to_latex(header, rows, cap))
    print(to_markdown(header, rows))
    if BASELINE not in set(agg.variant):
        print("\n(note: no baseline row, so deltas are blank)")


if __name__ == "__main__":
    main()
