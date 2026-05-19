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
    'loc': {
        'mean':   PROJECT_ROOT / 'configs' / 'subcellular_localization' / 'mean.yaml',
        'cov':    PROJECT_ROOT / 'configs' / 'subcellular_localization' / 'cov.yaml',
        'hybrid': PROJECT_ROOT / 'configs' / 'subcellular_localization' / 'hybrid.yaml',
        'la':     PROJECT_ROOT / 'configs' / 'subcellular_localization' / 'la.yaml',
        'la_cov': PROJECT_ROOT / 'configs' / 'subcellular_localization' / 'la_cov.yaml',
    },
    'meltome': {
        'mean':   PROJECT_ROOT / 'configs' / 'meltome' / 'mean.yaml',
        'cov':    PROJECT_ROOT / 'configs' / 'meltome' / 'cov.yaml',
        'hybrid': PROJECT_ROOT / 'configs' / 'meltome' / 'hybrid.yaml',
        'la':     PROJECT_ROOT / 'configs' / 'meltome' / 'la.yaml',
        'la_cov': PROJECT_ROOT / 'configs' / 'meltome' / 'la_cov.yaml',
    },
}
TRAIN_SCRIPTS = {
    'loc':     'train_subcellular_localization.py',
    'meltome': 'train_meltome.py',
}
# Methods that do not sweep over d_c (no projection dimension)
NO_DC_METHODS = {'mean', 'la'}
DC_VALUES = [8, 16, 24, 32, 48]


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
    args = parser.parse_args()

    tag = args.tag or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_dir = PROJECT_ROOT / 'sweeps' / tag
    (sweep_dir / 'configs').mkdir(parents=True, exist_ok=True)
    (sweep_dir / 'logs').mkdir(parents=True, exist_ok=True)
    print(f'Sweep dir: {sweep_dir}')
    print(f'Tasks:   {args.tasks}')
    print(f'Seeds:   {args.seeds}')
    print(f'Methods: {args.methods}')
    print(f'd_c:     {args.dcs}')

    method_plan = build_run_plan(args.methods, args.dcs)
    total = len(args.tasks) * len(args.seeds) * len(method_plan)
    print(f'\nTotal runs: {total}  ({len(args.tasks)} tasks × {len(args.seeds)} seeds × {len(method_plan)} method/dc combos)\n')

    summary_lines = []
    run_idx = 0
    for task in args.tasks:
        task_configs = BASE_CONFIGS[task]
        train_script = TRAIN_SCRIPTS[task]
        for seed in args.seeds:
            for method, dc in method_plan:
                run_idx += 1
                tag_str  = method if dc is None else f'{method}_dc{dc}'
                exp_name = f'{task}_{tag_str}_seed{seed}'
                config_path = sweep_dir / 'configs' / f'{task}_{tag_str}_seed{seed}.yaml'
                log_path    = sweep_dir / 'logs'    / f'{task}_{tag_str}_seed{seed}.log'

                make_config(task_configs[method], config_path, method, dc, seed, exp_name)

                print(f'[{run_idx:3d}/{total}] task={task}  seed={seed}  {tag_str}')
                t0 = datetime.datetime.now()
                rc = run_one(config_path, log_path, train_script)
                elapsed = (datetime.datetime.now() - t0).total_seconds() / 60.0
                status = 'ok' if rc == 0 else f'FAILED rc={rc}'
                line = f'{task}  {tag_str:20s}  seed={seed}  rc={rc:3d}  time={elapsed:6.1f}min'
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
