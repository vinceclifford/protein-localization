#!/usr/bin/env python3
"""
Run the full pooling/dimension sweep sequentially.

Supports multiple tasks and seeds in one invocation:

Usage:
    python scripts/run_sweep.py
    python scripts/run_sweep.py --tasks loc meltome --seeds 123 969 309
    python scripts/run_sweep.py --methods mean cov hybrid --dcs 8 16 32
    python scripts/run_sweep.py --tag overnight
"""
import argparse
import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIGS = {
    'meltome': {
        'cov_unsup': PROJECT_ROOT / 'configs' / 'meltome' / 'cov_unsup.yaml',
    },
}
TRAIN_SCRIPTS = {
    'meltome': 'train_meltome.py',
}
PRETRAIN_SCRIPTS = {
    'meltome': 'train_cov_unsup_meltome.py',
}
# Methods that do not sweep over d_c (no projection dimension)
NO_DC_METHODS = {'mean', 'la'}
DC_VALUES = [8]


def build_run_plan(methods, dcs):
    plan = []
    for method in methods:
        if method in NO_DC_METHODS:
            plan.append((method, None))
        else:
            for dc in dcs:
                plan.append((method, dc))
    return plan


def make_config(base_path: Path, dst_path: Path, method: str, dc: int | None,
                seed: int, experiment_name: str) -> None:
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    cfg['seed'] = seed
    cfg['experiment_name'] = experiment_name

    if "model_parameters" not in cfg or cfg["model_parameters"] is None:
        cfg["model_parameters"] = {}

    cfg["model_parameters"]["pooling"] = "cov_unsup"
    cfg["model_parameters"]["proj_dim"] = dc
    cfg["model_parameters"]["cov_unsup_checkpoint"] = str(checkpoint_path)
    cfg["model_parameters"]["freeze_cov_projections"] = True

    if dc is not None:
        cfg['model_parameters']['proj_dim'] = dc
    with open(dst_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def run_one(config_path: Path, log_path: Path, train_script: str = 'train.py') -> int:
    cmd = [sys.executable, '-u', train_script, '--config', str(config_path)]
    print(f'  → {" ".join(cmd)}')
    print(f'    log: {log_path}')
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    with open(log_path, 'w') as logf:
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
        proc.wait()
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[123],
                        help='One or more random seeds, e.g. --seeds 123 969 309')
    parser.add_argument('--tag', default=None,
                        help='Sweep folder name (default: timestamped)')
    parser.add_argument('--methods', nargs='+', default=['mean', 'cov', 'hybrid', 'la', 'la_cov'],
                        choices=['mean', 'cov', 'hybrid', 'la', 'la_cov'])
    parser.add_argument('--dcs', nargs='+', type=int, default=DC_VALUES)
    parser.add_argument('--tasks', nargs='+', default=['loc'],
                        choices=['loc', 'meltome'],
                        help='One or more tasks, e.g. --tasks loc meltome')
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--pretrain_batch_size', type=int, default=256)
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)
    parser.add_argument('--pretrain_patience', type=int, default=10)
    parser.add_argument('--pretrain_log_iterations', type=int, default=10)
    args = parser.parse_args()

    tag = args.tag or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = PROJECT_ROOT / "sweeps" / tag

    config_dir = sweep_dir / "configs"
    log_dir = sweep_dir / "logs"
    pretrain_dir = sweep_dir / "cov_unsup_pretrained"

    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    pretrain_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sweep dir: {sweep_dir}")
    print(f"Seeds:     {args.seeds}")
    print(f"d_c:       {args.dcs}")
    print(f"Method:    cov_unsup")
    print()

    total = len(args.seeds) * len(args.dcs)
    summary_lines = []
    run_idx = 0

    for seed in args.seeds:
        for dc in args.dcs:
            run_idx += 1

            tag_str = f"cov_unsup_dc{dc}"
            exp_name = f"meltome_{tag_str}_seed{seed}"

            checkpoint_path = pretrain_dir / f"cov_unsup_dc{dc}_seed{seed}.pt"

            pretrain_log_path = log_dir / f"{exp_name}_pretrain.log"
            downstream_config_path = config_dir / f"{exp_name}.yaml"
            downstream_log_path = log_dir / f"{exp_name}_train.log"

            print(f"[{run_idx:3d}/{total}] seed={seed}  d_c={dc}")

            #Train L and R projections with unsupervised covariance reconstruction loss
            pretrain_cmd = [
                sys.executable,
                "-u", PRETRAIN_SCRIPTS,
                "--seed", seed,
                "--proj_dim", dc,
                "--num_epochs", args.pretrain_epochs,
                "--batch_size", args.pretrain_batch_size,
                "--lr", args.pretrain_lr,
                "--patience", args.pretrain_patience,
                "--log_iterations", args.pretrain_log_iterations,
                "--output_dir", pretrain_dir,
            ]

            t0 = datetime.datetime.now()
            rc_pretrain = run_one(pretrain_cmd, pretrain_log_path)
            pretrain_elapsed = (datetime.datetime.now() - t0).total_seconds() / 60.0

            # train_cov_unsup_meltome.py currently saves as cov_unsup_dc{dc}.pt.
            # Rename it to include the seed so multiple seeds do not overwrite each other.
            raw_checkpoint_path = pretrain_dir / f"cov_unsup_dc{dc}.pt"

            if rc_pretrain == 0:
                if not raw_checkpoint_path.exists():
                    rc_pretrain = 999
                    print(f"ERROR: expected checkpoint not found: {raw_checkpoint_path}")
                else:
                    raw_checkpoint_path.replace(checkpoint_path)

            if rc_pretrain != 0:
                line = (
                    f"meltome  {tag_str:20s}  seed={seed}  "
                    f"pretrain_rc={rc_pretrain:3d}  train_rc=SKIP  "
                    f"pretrain_time={pretrain_elapsed:6.1f}min"
                )
                summary_lines.append(line)
                print(f"    FAILED pretrain rc={rc_pretrain}")
                with open(sweep_dir / "summary.txt", "w") as f:
                    f.write("\n".join(summary_lines) + "\n")
                continue

            # Make downstream config pointing to the pretrained checkpoint
            make_config(
                BASE_CONFIGS["meltome"]["cov_unsup"],
                downstream_config_path,
                method="cov_unsup",
                dc=dc,
                seed=seed,
                experiment_name=exp_name,
            )

            # Train downstream model with pretrained L/R 
            downstream_cmd = [
                sys.executable,
                "-u", TRAIN_SCRIPTS["meltome"],
                "--config", downstream_config_path,
            ]
            t1 = datetime.datetime.now()
            rc_train = run_one(downstream_cmd, downstream_log_path)
            train_elapsed = (datetime.datetime.now() - t1).total_seconds() / 60.0

            total_elapsed = pretrain_elapsed + train_elapsed
            status = "ok" if rc_train == 0 else f"FAILED train rc={rc_train}"

            line = (
                f"meltome  {tag_str:20s}  seed={seed}  "
                f"pretrain_rc={rc_pretrain:3d}  train_rc={rc_train:3d}  "
                f"time={total_elapsed:6.1f}min"
            )
            summary_lines.append(line)

            print(f"    {status}  ({total_elapsed:.1f} min total)")

            with open(sweep_dir / "summary.txt", "w") as f:
                f.write("\n".join(summary_lines) + "\n")

    print('Complete.')
    print('Summary:')
    print('\n'.join(summary_lines))
    print(f'\nNext steps:')
    print(f'  python scripts/collect_results.py --sweep {sweep_dir}')
    print(f'  python scripts/plot_sweep.py --sweep {sweep_dir}')


if __name__ == '__main__':
    main()
