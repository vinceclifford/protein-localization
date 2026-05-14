#!/usr/bin/env python3
"""
Walk runs/ for all PoolingFFN experiments matching a sweep tag, parse their
evaluation_test_set_after_train.txt files, and write a CSV summary.

Usage:
    python scripts/collect_results.py --sweep sweeps/20251205_180000
    python scripts/collect_results.py --pattern '*seed123*'
"""
import argparse
import csv
import os
import re
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_eval_file(path: Path) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            m = re.match(r'^(Accuracy|Accuracy stderr|MCC|MCC stderr|F1|F1 stderr): ([\-0-9.]+)', line)
            if m:
                key = m.group(1).lower().replace(' ', '_')
                out[key] = float(m.group(2))
    return out


def find_runs(sweep_dir: Path | None) -> list[Path]:
    runs_root = PROJECT_ROOT / 'runs'
    if not sweep_dir:
        return sorted(list(runs_root.glob('PoolingFFN_*'))
                      + list(runs_root.glob('LightAttentionCov_*')))
    cfg_dir = sweep_dir / 'configs'
    if not cfg_dir.exists():
        raise FileNotFoundError(f'no configs/ in {sweep_dir}')
    wanted_names = set()
    for cfg in cfg_dir.glob('*.yaml'):
        with open(cfg) as f:
            data = yaml.safe_load(f)
        wanted_names.add(data.get('experiment_name', ''))
    matched = []
    all_runs = sorted(list(runs_root.glob('PoolingFFN_*'))
                      + list(runs_root.glob('LightAttentionCov_*')))
    for run in all_runs:
        for name in wanted_names:
            if f'_{name}_' in run.name:
                matched.append(run)
                break
    return matched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=Path, default=None,
                        help='Sweep dir under sweeps/. If omitted, collects all PoolingFFN runs.')
    parser.add_argument('--out', type=Path, default=None,
                        help='Output CSV path (default: <sweep>/results.csv or results.csv)')
    args = parser.parse_args()

    runs = find_runs(args.sweep)
    if not runs:
        print('No matching runs found.')
        return

    rows = []
    for run in runs:
        eval_file = run / 'evaluation_test_set_after_train.txt'
        if not eval_file.exists():
            print(f'skip (no test eval): {run.name}')
            continue
        # parse the train_arguments.yaml saved in the run
        ta_path = run / 'train_arguments.yaml'
        config = {}
        if ta_path.exists():
            with open(ta_path) as f:
                config = yaml.safe_load(f)
        model_type = config.get('model_type', '')
        mp = config.get('model_parameters', {})
        if model_type == 'LightAttentionCov':
            method = 'la_cov'
        else:
            method = mp.get('pooling', '')
        proj_dim = mp.get('proj_dim', 0)
        seed = config.get('seed', '')
        metrics = parse_eval_file(eval_file)
        rows.append({
            'run_dir': run.name,
            'method': method,
            'proj_dim': proj_dim if method in ('cov', 'hybrid', 'la_cov') else '',
            'seed': seed,
            'test_acc': metrics.get('accuracy'),
            'test_acc_stderr': metrics.get('accuracy_stderr'),
            'test_mcc': metrics.get('mcc'),
            'test_mcc_stderr': metrics.get('mcc_stderr'),
            'test_f1': metrics.get('f1'),
            'test_f1_stderr': metrics.get('f1_stderr'),
        })

    if not rows:
        print('No completed evaluations found.')
        return

    out_path = args.out or ((args.sweep / 'results.csv') if args.sweep else PROJECT_ROOT / 'results.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f'Wrote {len(rows)} rows to {out_path}')

    print()
    print(f'{"method":8s} {"d_c":>5s} {"seed":>5s} {"Q10":>7s} {"±":>7s}')
    for r in sorted(rows, key=lambda x: (x['method'], x['proj_dim'] or 0)):
        print(f'{r["method"]:8s} {str(r["proj_dim"]):>5s} {str(r["seed"]):>5s} '
              f'{r["test_acc"]:7.2f} {r["test_acc_stderr"]:7.2f}')


if __name__ == '__main__':
    main()
