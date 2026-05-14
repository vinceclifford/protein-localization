#!/usr/bin/env python3
"""
Plot the dimension sweep from results.csv.

Produces two charts:
    sweep_dc_curve.png   - Q10 vs d_c with a line per method, mean as horizontal baseline
    sweep_bars.png       - one bar per (method, d_c)

Usage:
    python scripts/plot_sweep.py --sweep sweeps/20251205_180000
    python scripts/plot_sweep.py --csv results.csv
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_csv(path: Path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['test_acc'] = float(r['test_acc']) if r['test_acc'] else None
        r['test_acc_stderr'] = float(r['test_acc_stderr']) if r['test_acc_stderr'] else None
        r['proj_dim'] = int(r['proj_dim']) if r['proj_dim'] else None
    return rows


def plot_dc_curve(rows, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    methods = ['cov', 'hybrid', 'la_cov']
    colors = {'cov': '#1f77b4', 'hybrid': '#d62728', 'la_cov': '#9467bd'}
    for method in methods:
        pts = sorted([r for r in rows if r['method'] == method and r['proj_dim'] is not None],
                     key=lambda r: r['proj_dim'])
        if not pts:
            continue
        xs = [r['proj_dim'] for r in pts]
        ys = [r['test_acc'] for r in pts]
        es = [r['test_acc_stderr'] for r in pts]
        ax.errorbar(xs, ys, yerr=es, marker='o', label=method, color=colors[method],
                    capsize=3, linewidth=2)
    mean_rows = [r for r in rows if r['method'] == 'mean']
    if mean_rows:
        m = mean_rows[0]
        ax.axhline(m['test_acc'], linestyle='--', color='gray',
                   label=f"mean baseline ({m['test_acc']:.2f}%)")
        ax.fill_between([min([r['proj_dim'] or 8 for r in rows]) - 2,
                         max([r['proj_dim'] or 48 for r in rows]) + 2],
                        m['test_acc'] - m['test_acc_stderr'],
                        m['test_acc'] + m['test_acc_stderr'],
                        color='gray', alpha=0.15)
    ax.set_xlabel('d_c (projection dimension)')
    ax.set_ylabel('Test Q10 accuracy (%)')
    ax.set_title('DeepLoc setDeepLoc — pooling method vs d_c')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'wrote {out_path}')


def plot_bars(rows, out_path: Path):
    rows = sorted(rows, key=lambda r: (r['method'], r['proj_dim'] or -1))
    labels = []
    accs = []
    errs = []
    colors = []
    cmap = {'mean': '#2ca02c', 'cov': '#1f77b4', 'hybrid': '#d62728', 'la_cov': '#9467bd'}
    for r in rows:
        lbl = r['method'] if r['method'] == 'mean' else f"{r['method']}\n d_c={r['proj_dim']}"
        labels.append(lbl)
        accs.append(r['test_acc'])
        errs.append(r['test_acc_stderr'])
        colors.append(cmap.get(r['method'], 'gray'))

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(rows)), 5), dpi=150)
    xs = range(len(rows))
    ax.bar(xs, accs, yerr=errs, color=colors, capsize=4, edgecolor='black', linewidth=0.5)
    for i, (a, e) in enumerate(zip(accs, errs)):
        ax.text(i, a + 0.2, f'{a:.1f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=0, fontsize=9)
    ax.set_ylabel('Test Q10 accuracy (%)')
    ax.set_title('DeepLoc setDeepLoc — pooling sweep at seed 123')
    ax.set_ylim(bottom=max(0, min(accs) - 5), top=max(accs) + 3)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'wrote {out_path}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=Path, default=None)
    parser.add_argument('--csv', type=Path, default=None)
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

    plot_dc_curve(rows, out_dir / 'sweep_dc_curve.png')
    plot_bars(rows, out_dir / 'sweep_bars.png')


if __name__ == '__main__':
    main()
