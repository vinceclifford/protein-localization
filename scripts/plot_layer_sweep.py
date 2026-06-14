#!/usr/bin/env python3
"""
Plot layer × method results from run_layer_sweep.py.

Visualization principles:
    * Y-axis defaults to full range (0–100 for Q10, 0–1 for Spearman) so visual
      magnitudes are honest. Use --zoom for a tight close-up variant.
    * "Stacked" layer is plotted as a separate marker on the right edge, not
      mixed with the integer-layer line.
    * Multi-seed input is averaged with cross-seed stdev error bars.
    * Slide-ready fonts.

Usage:
    python scripts/plot_layer_sweep.py --csv sweeps/layer_<tag>/layer_results.csv
    python scripts/plot_layer_sweep.py --csv ... --zoom
    python scripts/plot_layer_sweep.py --csv ... --paper-loc 76 --paper-meltome 0.65
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl


METHOD_COLORS  = {'mean': '#2ca02c', 'cov': '#1f77b4', 'hybrid': '#d62728',
                   'la': '#ff7f0e',   'la_cov': '#9467bd'}
METHOD_MARKERS = {'mean': 's',       'cov': 'o',       'hybrid': '^',
                   'la': 'v',         'la_cov': 'D'}

# Y-axis floors. Standard view uses a meaningful reference floor (above
# random/majority baseline); zoom view uses a tight floor for small gaps.
# Top of axis is always data-driven.
TASK_FLOOR      = {'loc': 50.0, 'meltome': 0.50}
TASK_ZOOM_FLOOR = {'loc': 75.0, 'meltome': 0.65}

SLIDE_RC = {
    'font.size':       12,
    'axes.titlesize':  16,
    'axes.labelsize':  14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi':      150,
}


def parse_layer(v):
    if v is None or v == '':
        return None
    s = str(v).strip()
    if s.lower() == 'stacked':
        return 'stacked'
    try:
        return int(s)
    except ValueError:
        return s


def read_csv(path: Path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['layer'] = parse_layer(r.get('layer'))
        for col in ('spearman_r', 'spearman_stderr', 'accuracy',
                    'accuracy_stderr', 'mcc', 'mse',
                    'test_acc', 'test_acc_stderr', 'test_spearman',
                    'test_spearman_stderr'):
            v = r.get(col)
            if v in (None, ''):
                r[col] = None
                continue
            try:
                r[col] = float(v)
            except (ValueError, TypeError):
                r[col] = None
    return rows


def _is_nan(v):
    try:
        return v != v
    except Exception:
        return False


def stat(values):
    vals = [v for v in values if isinstance(v, (int, float)) and not _is_nan(v)]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def detect_task(rows, forced):
    if forced:
        return forced
    tasks = {r.get('task') for r in rows if r.get('task')}
    if 'meltome' in tasks: return 'meltome'
    if 'loc' in tasks:     return 'loc'
    return 'meltome' if any(r.get('spearman_r') is not None for r in rows) else 'loc'


def metric_keys(task):
    if task == 'meltome':
        return 'spearman_r', 'spearman_stderr', 'Spearman R (test)'
    return 'accuracy', 'accuracy_stderr', 'Q10 accuracy (test, %)'


def _task_title(task):
    return 'Meltome (FLIP human_cell)' if task == 'meltome' else 'DeepLoc setDeepLoc'


def _ylim(values, task, zoom):
    vals = [v for v in values if v is not None]
    pad = 0.04 if task == 'meltome' else 4.0
    floor_pref = (TASK_ZOOM_FLOOR if zoom else TASK_FLOOR)[task]
    if not vals:
        return (floor_pref, floor_pref + (0.3 if task == 'meltome' else 30))
    lo = min(floor_pref, min(vals) - pad)
    hi = max(vals) + pad
    return (lo, hi)


def _paper_baseline(task, args):
    if task == 'loc' and args.paper_loc is not None:
        return args.paper_loc, f'paper baseline (≈{args.paper_loc:g}% Q10)'
    if task == 'meltome' and args.paper_meltome is not None:
        return args.paper_meltome, f'paper baseline (≈{args.paper_meltome:g})'
    return None, None


def plot(rows, task, out_path, zoom, args):
    metric_key, stderr_key, ylabel = metric_keys(task)
    rows_task = [r for r in rows if r.get(metric_key) is not None
                 and (r.get('task') == task or 'task' not in r)]
    if not rows_task:
        print(f'No rows with {metric_key} for task {task}.')
        return

    grouped = defaultdict(list)
    for r in rows_task:
        grouped[(r['method'], r['layer'])].append(r[metric_key])

    methods = sorted({m for m, _ in grouped.keys()})
    int_layers = sorted({l for _, l in grouped.keys() if isinstance(l, int)})
    has_stacked = any(l == 'stacked' for _, l in grouped.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    all_y = []

    for method in methods:
        xs, ys, es = [], [], []
        for L in int_layers:
            vals = grouped.get((method, L), [])
            m, s = stat(vals)
            if m is None:
                continue
            xs.append(L); ys.append(m); es.append(s)
        if xs:
            ax.errorbar(xs, ys, yerr=es,
                        marker=METHOD_MARKERS.get(method, 'o'),
                        color=METHOD_COLORS.get(method, 'gray'),
                        label=method, capsize=4, linewidth=2.5, markersize=9)
            all_y.extend(ys)

        if has_stacked:
            vals = grouped.get((method, 'stacked'), [])
            m, s = stat(vals)
            if m is not None:
                stacked_x = (int_layers[-1] if int_layers else 24) + 6
                ax.errorbar([stacked_x], [m], yerr=[s],
                            marker=METHOD_MARKERS.get(method, 'o'),
                            color=METHOD_COLORS.get(method, 'gray'),
                            capsize=4, markersize=11, linestyle='none',
                            markerfacecolor='none', markeredgewidth=2.5)
                all_y.append(m)

    xticks = list(int_layers)
    xlabels = [f'L{l}' for l in int_layers]
    if has_stacked and int_layers:
        stacked_x = int_layers[-1] + 6
        xticks.append(stacked_x)
        xlabels.append('stacked\n(6+12+18+24)')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    ax.set_xlabel('ProtT5 transformer layer')
    ax.set_ylabel(ylabel)
    n_seeds = max(len(v) for v in grouped.values()) if grouped else 0
    ax.set_title(f'{_task_title(task)} — pooling vs ProtT5 layer')

    ax.set_ylim(_ylim(all_y, task, zoom))

    ref_val, ref_lbl = _paper_baseline(task, args)
    if ref_val is not None:
        ax.axhline(ref_val, linestyle=':', color='black', alpha=0.6,
                   linewidth=1.5, label=ref_lbl)

    ax.legend(title='Pooling', loc='lower right', framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    note = ('Error bars: stdev across seeds. '
            'Open marker on the right = stacked (6+12+18+24) embedding.'
            if any(l == 'stacked' for _, l in grouped.keys())
            else 'Error bars: stdev across seeds.')
    fig.text(0.01, 0.005, note, fontsize=8, style='italic', color='#555555',
             ha='left', va='bottom')
    fig.savefig(out_path)
    print(f'wrote {out_path}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--task', default=None, choices=['loc', 'meltome'])
    parser.add_argument('--zoom', action='store_true',
                        help='Also produce a zoomed-in version (suffix "_zoom.png").')
    parser.add_argument('--paper-loc', type=float, default=None,
                        help='Q10 baseline (%) to draw as a horizontal line, e.g. 76.')
    parser.add_argument('--paper-meltome', type=float, default=None,
                        help='Spearman R baseline to draw as a horizontal line, e.g. 0.65.')
    args = parser.parse_args()

    mpl.rcParams.update(SLIDE_RC)

    rows = read_csv(args.csv)
    if not rows:
        print('No rows in csv.')
        return

    tasks_in = sorted({r.get('task') for r in rows if r.get('task')})
    if not tasks_in:
        tasks_in = [detect_task(rows, args.task)]

    for task in tasks_in:
        suffix = f'_{task}' if len(tasks_in) > 1 else ''
        plot(rows, task, args.csv.with_name(f'layer_curve{suffix}.png'),
             zoom=False, args=args)
        if args.zoom:
            plot(rows, task, args.csv.with_name(f'layer_curve{suffix}_zoom.png'),
                 zoom=True, args=args)


if __name__ == '__main__':
    main()
