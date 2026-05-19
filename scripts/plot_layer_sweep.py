#!/usr/bin/env python3
"""
Plot layer × method results from run_layer_sweep.py.

Reads layer_results.csv and produces:
    layer_curve.png  — metric vs layer, one line per pooling method

Usage:
    python scripts/plot_layer_sweep.py --csv sweeps/layer_<tag>/layer_results.csv
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


METHOD_COLORS  = {'mean': '#2ca02c', 'cov': '#1f77b4', 'hybrid': '#d62728', 'la_cov': '#9467bd'}
METHOD_MARKERS = {'mean': 's',       'cov': 'o',       'hybrid': '^',       'la_cov': 'D'}


def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['layer'] = int(r['layer'])
        for col in ('spearman_r', 'spearman_stderr', 'accuracy', 'accuracy_stderr', 'mcc', 'mse'):
            if r.get(col):
                try:
                    r[col] = float(r[col])
                except ValueError:
                    r[col] = None
    return rows


def detect_task(rows: list[dict], forced: str | None) -> str:
    if forced:
        return forced
    return 'meltome' if any(r.get('spearman_r') is not None for r in rows) else 'loc'


def plot(rows: list[dict], task: str, out_path: Path) -> None:
    if task == 'meltome':
        metric_key, stderr_key = 'spearman_r', 'spearman_stderr'
        ylabel = 'Spearman R (test)'
        title  = 'ProtX layer sweep — pooling method comparison'
    else:
        metric_key, stderr_key = 'accuracy', 'accuracy_stderr'
        ylabel = 'Q10 accuracy (test, %)'
        title  = 'ProtX layer sweep — pooling method comparison'

    methods = sorted({r['method'] for r in rows})
    all_layers = sorted({r['layer'] for r in rows})

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    for method in methods:
        pts = sorted([r for r in rows if r['method'] == method], key=lambda r: r['layer'])
        xs = [r['layer'] for r in pts if r.get(metric_key) is not None]
        ys = [r[metric_key] for r in pts if r.get(metric_key) is not None]
        es = [r.get(stderr_key) or 0.0 for r in pts if r.get(metric_key) is not None]
        if not xs:
            continue
        ax.errorbar(
            xs, ys, yerr=es,
            marker=METHOD_MARKERS.get(method, 'o'),
            color=METHOD_COLORS.get(method, 'gray'),
            label=method, capsize=3, linewidth=2, markersize=7,
        )

    ax.set_xlabel('ProtX transformer layer')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(all_layers)
    ax.set_xticklabels([f'L{l}\n({"early" if l <= 6 else "middle" if l <= 14 else "late" if l < 24 else "last"})' for l in all_layers])
    ax.legend(title='Pooling')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'wrote {out_path}')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',  type=Path, required=True,
                        help='layer_results.csv from run_layer_sweep.py')
    parser.add_argument('--task', default=None, choices=['loc', 'meltome'])
    args = parser.parse_args()

    rows = read_csv(args.csv)
    if not rows:
        print('No rows in csv.')
        return

    task = detect_task(rows, args.task)
    out_path = args.csv.with_name('layer_curve.png')
    plot(rows, task, out_path)


if __name__ == '__main__':
    main()
