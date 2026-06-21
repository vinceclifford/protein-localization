#!/usr/bin/env python3
"""
Projection-dataset reconstruction comparison (d_c = 48).

Pretrains the unsupervised covariance projections on each dataset
(union / deeploc / meltome) for each of the three canonical seeds, caches every
checkpoint, and writes a reconstruction-loss summary (rel_err mean ± std).

Each (dataset, seed) run gets its own output dir so checkpoints never overwrite:
    runs/cov_unsup_compare/<dataset>_seed<seed>/cov_unsup_dc48.pt

Usage:
    python scripts/run_projection_comparison.py
    python scripts/run_projection_comparison.py --seeds 657 921 969 --dc 48
"""
import argparse
import csv
import datetime
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                    if (p / 'configs').is_dir() and (p / 'models').is_dir())
PRETRAIN_SCRIPT = "train_cov_unsup.py"
DATASET_CONFIGS = {
    'union':   PROJECT_ROOT / 'configs' / 'cov_unsup_pretrain' / 'union.yaml',
    'deeploc': PROJECT_ROOT / 'configs' / 'cov_unsup_pretrain' / 'deeploc.yaml',
    'meltome': PROJECT_ROOT / 'configs' / 'cov_unsup_pretrain' / 'meltome.yaml',
}
DEFAULT_SEEDS = [657, 921, 969]


def run_cmd(cmd, log_path: Path) -> int:
    print(f'  -> {" ".join(str(c) for c in cmd)}')
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "w") as logf:
        proc = subprocess.Popen([str(c) for c in cmd], cwd=PROJECT_ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=env, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
            logf.flush()
        proc.wait()
    return proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_CONFIGS),
                        choices=list(DATASET_CONFIGS))
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--dc', type=int, default=48)
    parser.add_argument('--pca_init', action='store_true', default=True)
    args = parser.parse_args()

    out_root = PROJECT_ROOT / 'runs' / 'cov_unsup_compare'
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in args.datasets:
        cfg = DATASET_CONFIGS[dataset]
        for seed in args.seeds:
            print(f"\n=== {dataset}  seed={seed}  d_c={args.dc} ===")
            run_dir = out_root / f"{dataset}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-u", PRETRAIN_SCRIPT,
                   "--config", cfg, "--proj_dim", args.dc, "--seed", seed,
                   "--output_dir", run_dir]
            if args.pca_init:
                cmd.append("--pca_init")
            rc = run_cmd(cmd, run_dir / "pretrain.log")

            ckpt = run_dir / f"cov_unsup_dc{args.dc}.pt"
            rel_err = loss = None
            if ckpt.exists():
                c = torch.load(ckpt, map_location="cpu")
                rel_err = c.get("rel_err")
                loss = c.get("loss")
            rows.append({"dataset": dataset, "seed": seed, "d_c": args.dc,
                         "rel_err": rel_err, "loss": loss,
                         "checkpoint": str(ckpt), "rc": rc})

    summary = out_root / "reconstruction_summary.csv"
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "seed", "d_c", "rel_err",
                                          "loss", "checkpoint", "rc"])
        w.writeheader()
        w.writerows(rows)

    print("\n=== reconstruction rel_err (mean +/- std over seeds, d_c=%d) ===" % args.dc)
    for dataset in args.datasets:
        errs = [r["rel_err"] for r in rows
                if r["dataset"] == dataset and r["rel_err"] is not None]
        if not errs:
            print(f"  {dataset:10s}  (no successful runs)")
            continue
        m = statistics.mean(errs)
        s = statistics.stdev(errs) if len(errs) > 1 else 0.0
        per_seed = ", ".join(f"{e:.4f}" for e in errs)
        print(f"  {dataset:10s}  {m:.4f} +/- {s:.4f}   [{per_seed}]   n={len(errs)}")
    print(f"\nsummary written to {summary}")


if __name__ == "__main__":
    main()
