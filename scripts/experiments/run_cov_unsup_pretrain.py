#!/usr/bin/env python3
"""
Pretrain the unsupervised covariance projections for every d_c in the size sweep.

Runs train_cov_unsup.py once per d_c on the union of all task train splits,
producing one frozen checkpoint per d_c that both downstream tasks reuse:

    runs/cov_unsup_pretrained/cov_unsup_dc{d_c}.pt

Usage:
    python scripts/run_cov_unsup_pretrain.py
    python scripts/run_cov_unsup_pretrain.py --dcs 8 32 --seed 123
    python scripts/run_cov_unsup_pretrain.py --config configs/cov_unsup_pretrain.yaml
"""
import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                    if (p / 'configs').is_dir() and (p / 'models').is_dir())
PRETRAIN_SCRIPT = "train_cov_unsup.py"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cov_unsup_pretrain" / "union.yaml"
DC_VALUES = [8, 16, 24, 32, 48]


def run_cmd(cmd, log_path: Path) -> int:
    print(f'  → {" ".join(str(c) for c in cmd)}')
    print(f"    log: {log_path}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
            logf.flush()

        proc.wait()

    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dcs", type=int, nargs="+", default=DC_VALUES)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pca_init", action="store_true",
                        help="PCA-initialize L, R then refine with SGD.")
    parser.add_argument("--pca_only", action="store_true",
                        help="PCA-initialize and save, no SGD refinement.")
    args = parser.parse_args()

    log_dir = PROJECT_ROOT / "runs" / "cov_unsup_pretrained" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = []
    for dc in args.dcs:
        print(f"\n=== pretrain d_c={dc} ===")
        log_path = log_dir / f"cov_unsup_dc{dc}.log"

        cmd = [sys.executable, "-u", PRETRAIN_SCRIPT, "--config", args.config, "--proj_dim", dc]
        if args.seed is not None:
            cmd.extend(["--seed", args.seed])
        if args.pca_only:
            cmd.append("--pca_only")
        elif args.pca_init:
            cmd.append("--pca_init")

        t0 = datetime.datetime.now()
        rc = run_cmd(cmd, log_path)
        minutes = (datetime.datetime.now() - t0).total_seconds() / 60.0

        checkpoint = PROJECT_ROOT / "runs" / "cov_unsup_pretrained" / f"cov_unsup_dc{dc}.pt"
        status = "ok" if rc == 0 and checkpoint.exists() else f"FAILED(rc={rc})"
        summary_lines.append(f"d_c={dc}  {status}  time={minutes:.1f}min  checkpoint={checkpoint}")

        if rc != 0:
            print(f"\nd_c={dc} failed; stopping.")
            break

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
