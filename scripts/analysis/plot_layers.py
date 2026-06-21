"""
Plot the layer sweep from the master results.csv.

Uses only rows with a non-empty `layer` column (the layer sweep: mean / cov /
hybrid at a fixed d_c across ProtT5 layers 06/12/18/24 and the Stacked = all-layers
representation). The d_c sweep rows (empty `layer`) are ignored here — plot those
with scripts/analysis/plot_dc.py. Both read the same results.csv.

Produces, per task:
    layer_curve_averaged[_<task>].png   metric vs layer, one line per method,
                                        averaged across seeds (stdev error bars)
    layer_summary[_<task>].txt          per (method, layer) mean ± std table

Usage:
    python scripts/analysis/plot_layers.py --csv results/results.csv --out results/figures/layer_sweep
    python scripts/analysis/plot_layers.py --csv results/results.csv --out ... --seeds 657 921 969
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                    if (p / 'configs').is_dir() and (p / 'models').is_dir())

# Methods that appear in the layer sweep, in plot order.
METHOD_ORDER = ['mean', 'cov', 'hybrid']
METHOD_COLOR = {
    'mean':   '#2ca02c',   # green
    'cov':    '#1f77b4',   # blue
    'hybrid': '#d62728',   # red
}
# Layer tokens in x-axis order. 'Stacked' = all layers concatenated, placed last.
LAYER_ORDER = ['06', '12', '18', '24', 'Stacked']

TASK_FLOOR = {'loc': 50.0, 'meltome': 0.50}

SLIDE_RC = {
    'font.size':       12,
    'axes.titlesize':  14,
    'axes.labelsize':  14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi':      150,
}


def read_csv(path: Path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['seed'] = int(r['seed']) if r.get('seed') else None
        for col in ('test_acc', 'test_acc_stderr',
                    'test_spearman', 'test_spearman_stderr'):
            if col in r:
                r[col] = float(r[col]) if r[col] not in (None, '') else None
    return rows


def _stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def _metric_key(task: str) -> str:
    return 'test_spearman' if task == 'meltome' else 'test_acc'


def _axis_label(task: str) -> str:
    return 'Spearman R (test)' if task == 'meltome' else 'Test Q10 accuracy (%)'


def _task_title(task: str) -> str:
    return 'Meltome (FLIP human_cell)' if task == 'meltome' else 'DeepLoc setDeepLoc'


def _fmt(task: str) -> str:
    return '.3f' if task == 'meltome' else '.1f'


def _layer_index(layer: str) -> int:
    return LAYER_ORDER.index(layer) if layer in LAYER_ORDER else len(LAYER_ORDER)


def _suffix(task: str, multi_task: bool) -> str:
    return f'_{task}' if multi_task else ''


def plot_layer_curve(rows, task, out_dir, multi_task):
    metric_key = _metric_key(task)
    fig, ax = plt.subplots(figsize=(9, 6))
    all_y = []
    layers_present = sorted({r['layer'] for r in rows if r.get('layer')}, key=_layer_index)
    x_pos = {lyr: i for i, lyr in enumerate(layers_present)}

    for method in METHOD_ORDER:
        by_layer = defaultdict(list)
        for r in rows:
            if r['method'] == method and r.get('layer') and r.get(metric_key) is not None:
                by_layer[r['layer']].append(r[metric_key])
        if not by_layer:
            continue
        lyrs = sorted(by_layer, key=_layer_index)
        xs = [x_pos[l] for l in lyrs]
        means = [_stat(by_layer[l])[0] for l in lyrs]
        stds = [_stat(by_layer[l])[1] for l in lyrs]
        n = max(len(v) for v in by_layer.values())
        ax.errorbar(xs, means, yerr=stds, marker='o', label=f'{method} (n={n})',
                    color=METHOD_COLOR.get(method, 'gray'), capsize=4,
                    linewidth=2.5, markersize=9)
        all_y.extend(means)

    if not all_y:
        plt.close(fig)
        return

    pad = 0.04 if task == 'meltome' else 4.0
    lo = min(TASK_FLOOR[task], min(all_y) - pad)
    hi = max(all_y) + pad
    ax.set_ylim(lo, hi)
    ax.set_xticks(range(len(layers_present)))
    ax.set_xticklabels([f'layer {l}' if l != 'Stacked' else l for l in layers_present])
    ax.set_xlabel('ProtT5 layer')
    ax.set_ylabel(_axis_label(task))
    ax.set_title(f'{_task_title(task)} — metric vs layer (avg over seeds)')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.text(0.01, 0.005, 'Error bars: stdev across seeds. Fixed d_c; Stacked = all layers concatenated.',
             fontsize=8, style='italic', color='#555555', ha='left', va='bottom')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = out_dir / f'layer_curve_averaged{_suffix(task, multi_task)}.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'wrote {out}')


def write_summary(rows, task, out_dir, multi_task):
    metric_key = _metric_key(task)
    fmt = _fmt(task)
    metric_name = 'Spearman R' if task == 'meltome' else 'Q10'
    lines = [f'=== Layer sweep: {task} ({metric_name}) ===',
             f"{'method':8s} {'layer':>8s} {'n':>3s} {'mean':>10s} {'std':>9s}  per-seed",
             '-' * 78]
    grouped = defaultdict(list)
    for r in rows:
        if r.get('layer') and r.get(metric_key) is not None:
            grouped[(r['method'], r['layer'])].append(r)
    for key in sorted(grouped, key=lambda k: (METHOD_ORDER.index(k[0]) if k[0] in METHOD_ORDER else 9,
                                              _layer_index(k[1]))):
        vals = [r[metric_key] for r in grouped[key]]
        m, s = _stat(vals)
        per_seed = ', '.join(f'{r["seed"]}:{format(r[metric_key], fmt)}'
                             for r in sorted(grouped[key], key=lambda r: r['seed'] or 0))
        lines.append(f'{key[0]:8s} {key[1]:>8s} {len(vals):>3d} '
                     f'{format(m, ">10" + fmt)} {format(s, ">9" + fmt)}  {per_seed}')
    out = out_dir / f'layer_summary{_suffix(task, multi_task)}.txt'
    out.write_text('\n'.join(lines) + '\n')
    print(f'wrote {out}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=PROJECT_ROOT / 'results' / 'results.csv')
    parser.add_argument('--out', type=Path, default=None,
                        help='Output dir (default: <csv dir>/figures/layer_sweep).')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Only use these seeds (e.g. --seeds 657 921 969).')
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f'{args.csv} not found. Run scripts/collect_results.py first.')
    out_dir = args.out or (args.csv.parent / 'figures' / 'layer_sweep')
    out_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(SLIDE_RC)

    rows = read_csv(args.csv)
    rows = [r for r in rows if (r.get('layer') or '').strip()]
    if not rows:
        print('No layer-sweep rows (non-empty `layer`) in CSV. Nothing to plot.')
        return

    if args.seeds:
        keep = set(args.seeds)
        before = len(rows)
        rows = [r for r in rows if r['seed'] in keep]
        print(f'Filtered to seeds {sorted(keep)}: {before} -> {len(rows)} rows')

    tasks_present = sorted({r.get('task', 'loc') for r in rows})
    multi_task = len(tasks_present) > 1
    print(f'Loaded {len(rows)} layer-sweep rows; tasks: {tasks_present}')
    print(f'Output dir: {out_dir}\n')

    for task in tasks_present:
        task_rows = [r for r in rows if r.get('task', 'loc') == task]
        plot_layer_curve(task_rows, task, out_dir, multi_task)
        write_summary(task_rows, task, out_dir, multi_task)

    print('\nDone.')


if __name__ == '__main__':
    main()
