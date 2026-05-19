#!/usr/bin/env python3
"""
Plot the dimension sweep from results.csv.

Auto-detects task from the CSV columns:
    - loc task:     plots test_acc  (Q10 accuracy, %)
    - meltome task: plots test_spearman (Spearman R)

Produces two charts:
    sweep_dc_curve.png   - metric vs d_c with a line per method, mean as horizontal baseline
    sweep_bars.png       - one bar per (method, d_c)

Usage:
    python scripts/plot_sweep.py --sweep sweeps/20251205_180000
    python scripts/plot_sweep.py --csv results.csv
    python scripts/plot_sweep.py --sweep sweeps/... --task meltome
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

METHOD_COLORS = {'mean': '#2ca02c', 'cov': '#1f77b4', 'hybrid': '#d62728', 'la_cov': '#9467bd'}


def read_csv(path: Path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['proj_dim'] = int(r['proj_dim']) if r.get('proj_dim') else None
        for col in ('test_acc', 'test_acc_stderr', 'test_spearman', 'test_spearman_stderr',
                    'test_mse', 'test_mae'):
            if col in r:
                r[col] = float(r[col]) if r[col] else None
    return rows


def detect_task(rows: list, forced_task: str | None) -> str:
    if forced_task:
        return forced_task
    tasks = {r.get('task', '') for r in rows}
    if 'meltome' in tasks:
        return 'meltome'
    # fall back to column presence
    if any(r.get('test_spearman') is not None for r in rows):
        return 'meltome'
    return 'loc'


def _metric_key(task: str):
    if task == 'meltome':
        return 'test_spearman', 'test_spearman_stderr'
    return 'test_acc', 'test_acc_stderr'


def _axis_label(task: str) -> str:
    return 'Spearman R' if task == 'meltome' else 'Test Q10 accuracy (%)'


def plot_dc_curve(rows, task: str, out_path: Path):
    metric_key, stderr_key = _metric_key(task)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    for method in ('cov', 'hybrid', 'la_cov'):
        pts = sorted(
            [r for r in rows if r['method'] == method and r['proj_dim'] is not None and r.get(metric_key) is not None],
            key=lambda r: r['proj_dim'],
        )
        if not pts:
            continue
        xs = [r['proj_dim'] for r in pts]
        ys = [r[metric_key] for r in pts]
        es = [r.get(stderr_key) or 0.0 for r in pts]
        ax.errorbar(xs, ys, yerr=es, marker='o', label=method, color=METHOD_COLORS[method],
                    capsize=3, linewidth=2)
    mean_rows = [r for r in rows if r['method'] == 'mean' and r.get(metric_key) is not None]
    if mean_rows:
        m = mean_rows[0]
        val = m[metric_key]
        err = m.get(stderr_key) or 0.0
        all_dims = [r['proj_dim'] for r in rows if r['proj_dim'] is not None]
        x_min = (min(all_dims) - 2) if all_dims else 6
        x_max = (max(all_dims) + 2) if all_dims else 50
        ax.axhline(val, linestyle='--', color='gray', label=f'mean baseline ({val:.4f})')
        ax.fill_between([x_min, x_max], val - err, val + err, color='gray', alpha=0.15)
    ax.set_xlabel('d_c (projection dimension)')
    ax.set_ylabel(_axis_label(task))
    title_task = 'Meltome (FLIP)' if task == 'meltome' else 'DeepLoc'
    ax.set_title(f'{title_task} — pooling method vs d_c')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'wrote {out_path}')


def plot_bars(rows, task: str, out_path: Path):
    metric_key, stderr_key = _metric_key(task)
    rows = sorted(rows, key=lambda r: (r['method'], r['proj_dim'] or -1))
    labels, vals, errs, colors = [], [], [], []
    for r in rows:
        if r.get(metric_key) is None:
            continue
        lbl = r['method'] if r['method'] == 'mean' else f"{r['method']}\nd_c={r['proj_dim']}"
        labels.append(lbl)
        vals.append(r[metric_key])
        errs.append(r.get(stderr_key) or 0.0)
        colors.append(METHOD_COLORS.get(r['method'], 'gray'))

    if not vals:
        print('No data to plot.')
        return

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(vals)), 5), dpi=150)
    xs = range(len(vals))
    ax.bar(xs, vals, yerr=errs, color=colors, capsize=4, edgecolor='black', linewidth=0.5)
    fmt = '.4f' if task == 'meltome' else '.1f'
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e * 1.1 + (0.002 if task == 'meltome' else 0.2),
                format(v, fmt), ha='center', va='bottom', fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_ylabel(_axis_label(task))
    title_task = 'Meltome (FLIP)' if task == 'meltome' else 'DeepLoc'
    ax.set_title(f'{title_task} — pooling sweep at seed 123')
    margin = 0.05 if task == 'meltome' else 5
    ax.set_ylim(bottom=max(0, min(vals) - margin), top=max(vals) + margin)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'wrote {out_path}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=Path, default=None)
    parser.add_argument('--csv', type=Path, default=None)
    parser.add_argument('--task', default=None, choices=['loc', 'meltome'],
                        help='Force task type. Auto-detected from CSV columns if omitted.')
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
        out_dir = csv_path.parent
    elif args.sweep:
        csv_path = args.sweep / 'results.csv'
        out_dir = args.sweep
    else:
        csv_path = PROJECT_ROOT / 'results.csv'
        out_dir = PROJECT_ROOT

    rows = read_csv(csv_path)
    if not rows:
        print('No rows in csv.')
        return

    task = detect_task(rows, args.task)
    print(f'Task: {task}')

    plot_dc_curve(rows, task, out_dir / 'sweep_dc_curve.png')
    plot_bars(rows, task, out_dir / 'sweep_bars.png')


if __name__ == '__main__':
    main()
