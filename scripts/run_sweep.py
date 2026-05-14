#!/usr/bin/env python3
"""
Run the full pooling/dimension sweep sequentially.

Sweep:
    - mean baseline (no d_c)
    - covariance pooling, d_c in {8, 16, 24, 32, 48}
    - hybrid pooling,     d_c in {8, 16, 24, 32, 48}
    = 11 runs total, all at seed 123.

Each run is launched as a subprocess so its stdout streams live to a per-run log
file under sweeps/<sweep_tag>/logs/. The base YAML configs are duplicated to
temporary configs with the proj_dim and experiment_name overridden.

Usage:
    python scripts/run_sweep.py
    python scripts/run_sweep.py --seed 123 --tag overnight
    python scripts/run_sweep.py --methods mean cov           # subset
    python scripts/run_sweep.py --dcs 8 16 32                # subset of d_c
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
    'mean':   PROJECT_ROOT / 'configs' / 'mean.yaml',
    'cov':    PROJECT_ROOT / 'configs' / 'cov.yaml',
    'hybrid': PROJECT_ROOT / 'configs' / 'hybrid.yaml',
    'la_cov': PROJECT_ROOT / 'configs' / 'la_cov.yaml',
}
DC_VALUES = [8, 16, 24, 32, 48]


def build_run_plan(methods, dcs):
    plan = []
    for method in methods:
        if method == 'mean':
            plan.append(('mean', None))
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
    if dc is not None:
        cfg['model_parameters']['proj_dim'] = dc
    with open(dst_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def run_one(config_path: Path, log_path: Path) -> int:
    cmd = [sys.executable, '-u', 'train.py', '--config', str(config_path)]
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
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--tag', default=None,
                        help='Sweep folder name (default: timestamped)')
    parser.add_argument('--methods', nargs='+', default=['mean', 'cov', 'hybrid'],
                        choices=['mean', 'cov', 'hybrid', 'la_cov'])
    parser.add_argument('--dcs', nargs='+', type=int, default=DC_VALUES)
    args = parser.parse_args()

    tag = args.tag or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_dir = PROJECT_ROOT / 'sweeps' / tag
    (sweep_dir / 'configs').mkdir(parents=True, exist_ok=True)
    (sweep_dir / 'logs').mkdir(parents=True, exist_ok=True)
    print(f'Sweep dir: {sweep_dir}')

    plan = build_run_plan(args.methods, args.dcs)
    print(f'Plan: {len(plan)} runs')
    for method, dc in plan:
        tag_str = method if dc is None else f'{method}_dc{dc}'
        print(f'  - {tag_str}')

    summary_lines = []
    for i, (method, dc) in enumerate(plan, 1):
        tag_str = method if dc is None else f'{method}_dc{dc}'
        exp_name = f'{tag_str}_seed{args.seed}'
        config_path = sweep_dir / 'configs' / f'{tag_str}.yaml'
        log_path = sweep_dir / 'logs' / f'{tag_str}.log'

        make_config(BASE_CONFIGS[method], config_path, method, dc, args.seed, exp_name)

        print(f'\n[{i}/{len(plan)}] {tag_str}')
        t0 = datetime.datetime.now()
        rc = run_one(config_path, log_path)
        elapsed = (datetime.datetime.now() - t0).total_seconds() / 60.0
        status = 'ok' if rc == 0 else f'FAILED rc={rc}'
        line = f'{tag_str:20s}  seed={args.seed}  rc={rc:3d}  time={elapsed:6.1f}min'
        summary_lines.append(line)
        print(f'    {status}  ({elapsed:.1f} min)')

        with open(sweep_dir / 'summary.txt', 'w') as f:
            f.write('\n'.join(summary_lines) + '\n')

    print('\nSweep complete.')
    print('Summary:')
    print('\n'.join(summary_lines))
    print(f'\nNext steps:')
    print(f'  python scripts/collect_results.py --sweep {sweep_dir}')
    print(f'  python scripts/plot_sweep.py --sweep {sweep_dir}')


if __name__ == '__main__':
    main()
