#!/usr/bin/env python3
"""
Walk runs/ for all PoolingFFN experiments matching a sweep tag, parse their
evaluation_test_set_after_train.txt files, and write a CSV summary.

Usage:
    python scripts/collect_results.py --sweep sweeps/20251205_180000
    python scripts/collect_results.py --pattern '*seed123*'
    python scripts/collect_results.py --sweep sweeps/... --task meltome
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import yaml


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                    if (p / 'configs').is_dir() and (p / 'models').is_dir())


def parse_eval_file_loc(path: Path) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            m = re.match(r'^(Accuracy|Accuracy stderr|MCC|MCC stderr|F1|F1 stderr): ([\-0-9.]+)', line)
            if m:
                key = m.group(1).lower().replace(' ', '_')
                out[key] = float(m.group(2))
    return out


def parse_eval_file_meltome(path: Path) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            m = re.match(r'^(Spearman R|Spearman stderr|MSE|MAE): ([\-0-9.eE+]+)', line)
            if m:
                key = m.group(1).lower().replace(' ', '_')
                out[key] = float(m.group(2))
    return out


def detect_task(path: Path) -> str:
    """Detect whether an eval file is from localization or meltome based on its content."""
    with open(path) as f:
        content = f.read()
    return 'meltome' if 'Spearman R' in content else 'loc'


def _all_pooling_runs(runs_root: Path) -> list[Path]:
    """All run dirs under runs_root, recursively (handles deeploc/meltome subdirs)."""
    return sorted([p for p in runs_root.rglob('PoolingFFN_*') if p.is_dir()]
                  + [p for p in runs_root.rglob('LightAttentionCov_*') if p.is_dir()])


def find_runs(sweep_dir: Path | None, runs_root: Path) -> list[Path]:
    if not sweep_dir:
        return _all_pooling_runs(runs_root)
    cfg_dir = sweep_dir / 'configs'
    if not cfg_dir.exists():
        raise FileNotFoundError(f'no configs/ in {sweep_dir}')
    wanted_names = set()
    for cfg in cfg_dir.glob('*.yaml'):
        with open(cfg) as f:
            data = yaml.safe_load(f)
        wanted_names.add(data.get('experiment_name', ''))
    matched = []
    for run in _all_pooling_runs(runs_root):
        for name in wanted_names:
            if f'_{name}_' in run.name:
                matched.append(run)
                break
    return matched


def parse_layer(name: str) -> str:
    """Extract the layer token from a layer-sweep run dir name, '' if absent."""
    m = re.search(r'_layer([A-Za-z0-9]+)_', name)
    return m.group(1) if m else ''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=Path, default=None,
                        help='Sweep dir under sweeps/. If omitted, collects all PoolingFFN runs.')
    parser.add_argument('--out', type=Path, default=None,
                        help='Output CSV path (default: <sweep>/results.csv or results.csv)')
    parser.add_argument('--runs-dir', type=Path, default=None,
                        help='Directory to search for run dirs (recursively). '
                             'Default: <repo>/runs. Point at e.g. '
                             'checkpoints/heads/supervised/deeploc.')
    parser.add_argument('--task', default=None, choices=['loc', 'meltome'],
                        help='Force task type. Auto-detected from eval file content if omitted.')
    args = parser.parse_args()

    runs_root = args.runs_dir or (PROJECT_ROOT / 'runs')
    runs = find_runs(args.sweep, runs_root)
    if not runs:
        print('No matching runs found.')
        return

    rows = []
    for run in runs:
        eval_file = run / 'evaluation_test_set_after_train.txt'
        if not eval_file.exists():
            print(f'skip (no test eval): {run.name}')
            continue
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
        layer = parse_layer(run.name)

        task = args.task or detect_task(eval_file)
        if task == 'meltome':
            metrics = parse_eval_file_meltome(eval_file)
            rows.append({
                'run_dir': run.name,
                'task': 'meltome',
                'method': method,
                'layer': layer,
                'proj_dim': proj_dim if method in ('cov', 'cov_unsup', 'hybrid', 'la_cov') else '',
                'seed': seed,
                'test_spearman': metrics.get('spearman_r'),
                'test_spearman_stderr': metrics.get('spearman_stderr'),
                'test_mse': metrics.get('mse'),
                'test_mae': metrics.get('mae'),
            })
        else:
            metrics = parse_eval_file_loc(eval_file)
            rows.append({
                'run_dir': run.name,
                'task': 'loc',
                'method': method,
                'layer': layer,
                'proj_dim': proj_dim if method in ('cov', 'cov_unsup', 'hybrid', 'la_cov') else '',
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
    all_keys = sorted({k for r in rows for k in r.keys()})
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in all_keys})
    print(f'Wrote {len(rows)} rows to {out_path}')

    tasks_in_rows = sorted({r.get('task', 'loc') for r in rows})
    for task in tasks_in_rows:
        task_rows = sorted((r for r in rows if r.get('task', 'loc') == task),
                           key=lambda x: (x.get('layer') or '', x['method'], x.get('proj_dim') or 0))
        metric_name = 'SpearmanR' if task == 'meltome' else 'Q10'
        metric_col = 'test_spearman' if task == 'meltome' else 'test_acc'
        stderr_col = 'test_spearman_stderr' if task == 'meltome' else 'test_acc_stderr'
        print()
        print(f'=== {task} ===')
        print(f'{"layer":>8s} {"method":8s} {"d_c":>5s} {"seed":>5s} {metric_name:>10s} {"±":>8s}')
        for r in task_rows:
            val = r.get(metric_col)
            err = r.get(stderr_col)
            val_s = f'{val:10.4f}' if val is not None else '       nan'
            err_s = f'{err:8.4f}' if err is not None else '     nan'
            print(f'{(r.get("layer") or "-"):>8s} {r["method"]:8s} '
                  f'{str(r["proj_dim"]):>5s} {str(r["seed"]):>5s} {val_s} {err_s}')


if __name__ == '__main__':
    main()
