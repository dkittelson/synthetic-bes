"""Run every (variant, seed) job, one subprocess each.

    python -m bes_sweep.sweep                       # all variants, seed 0
    python -m bes_sweep.sweep --variants baseline tiny xl --seeds 0 1 2
    python -m bes_sweep.sweep --parallel 2          # only if you measured a gain

Jobs already finished (metrics.json present) are skipped, so an interrupted
sweep resumes by re-running the same command. One process per job gives clean
GPU memory and means one crash does not take down the rest.
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bes_sweep.configs import VARIANTS
from bes_sweep.train_one import ROOT, default_runs_dir, run_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--max-events", type=int, default=None)
    args = p.parse_args()

    runs_dir = default_runs_dir(args.max_events)
    jobs = [(v, s) for s in args.seeds for v in args.variants]  # seed-major: full seed 0 first

    def run(job):
        v, s = job
        out = run_dir(runs_dir, v, s)
        if (out / "metrics.json").exists():
            return job, 0, "skipped"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "bes_sweep.train_one", "--variant", v, "--seed", str(s),
               "--epochs", str(args.epochs), "--patience", str(args.patience)]
        if args.max_events:
            cmd += ["--max-events", str(args.max_events)]
        t0 = time.time()
        with open(out / "train.log", "w") as log:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        return job, rc, f"{(time.time() - t0) / 60:.1f} min"

    print(f"{len(jobs)} jobs, parallel={args.parallel}, runs_dir={runs_dir}", flush=True)
    failed = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for (v, s), rc, info in pool.map(run, jobs):
            print(f"{'ok  ' if rc == 0 else 'FAIL'} {v} seed{s} ({info})", flush=True)
            if rc != 0:
                failed.append((v, s))
    if failed:
        print("failed:", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
