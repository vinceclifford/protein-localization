"""
Produce a comprehensive set of plots from a sweep's results.csv.

Generates per-task variants of:
    bars_seed_<S>.png         one bar chart per seed
    bars_averaged.png         bars averaged across seeds (mean ± stdev across seeds)
    dc_curve_averaged.png     metric vs d_c, averaged across seeds
    dc_curve_per_seed.png     metric vs d_c with one panel per seed
    head_to_head.png          head-to-head at a single d_c (default 32)
    per_seed_scatter.png      scatter of seeds per (method, d_c) with mean bars
    summary_table.txt / .md   tabular summary

Visualization principles applied:
    * Bar y-axes always start at 0 (no truncation) so visual magnitudes are honest.
    * Optional `--zoom` flag produces a second set of plots with a tight y-range,
      labelled "_zoom" so they can be presented alongside the full-range version
      when you want to highlight small differences.
    * Consistent y-axis limits across all plots of the same task.
    * Larger fonts and figure sizes tuned for slide presentations.
    * Optional paper-baseline reference lines via --paper-loc / --paper-meltome.

Plots the d_c sweep only (rows with an empty `layer`); layer-sweep rows are
plotted by scripts/analysis/plot_layers.py. Both read the same master results.csv.

Usage:
    python scripts/analysis/plot_dc.py --sweep sweeps/<tag>
    python scripts/analysis/plot_dc.py --csv results/results.csv --out results/figures/dc_sweep
    python scripts/analysis/plot_dc.py --sweep ... --zoom            # also produce zoomed
    python scripts/analysis/plot_dc.py --sweep ... --paper-loc 76 --paper-meltome 0.65
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

METHOD_ORDER = ['mean', 'cov', 'cov_unsup', 'hybrid', 'la', 'la_cov']
METHOD_COLOR = {
    'mean':      '#2ca02c',   # green
    'cov':       '#1f77b4',   # blue
    'cov_unsup': '#9467bd',   # purple
    'hybrid':    '#d62728',   # red
    'la':        '#ff7f0e',   # orange
    'la_cov':    '#8c564b',   # brown
}
DC_VALUES = [8, 16, 24, 32, 48]

# Y-axis floors per task.
#   "standard" view: a meaningful reference floor (above random/majority baseline)
#                    that keeps differences readable without truncating to a lie.
#   "zoom" view:     a tight floor for close-up comparison of small gaps.
# The top of the axis is always data-driven (tallest bar/point + padding).
TASK_FLOOR      = {'loc': 50.0, 'meltome': 0.50}
TASK_ZOOM_FLOOR = {'loc': 75.0, 'meltome': 0.65}

# Slide-ready font and figure defaults
SLIDE_RC = {
    'font.size':       12,
    'axes.titlesize':  14,
    'axes.labelsize':  14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi':      150,
}


def _footnote(fig, text):
    """Small italic note at the bottom-left for methodology details."""
    fig.text(0.01, 0.005, text, fontsize=8, style='italic', color='#555555',
             ha='left', va='bottom')


def read_csv(path: Path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['proj_dim'] = int(r['proj_dim']) if r.get('proj_dim') else None
        r['seed'] = int(r['seed']) if r.get('seed') else None
        for col in ('test_acc', 'test_acc_stderr', 'test_mcc', 'test_mcc_stderr',
                    'test_f1', 'test_f1_stderr',
                    'test_spearman', 'test_spearman_stderr', 'test_mse', 'test_mae'):
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


def _metric_key(task: str):
    if task == 'meltome':
        return 'test_spearman', 'test_spearman_stderr'
    return 'test_acc', 'test_acc_stderr'


def _axis_label(task: str) -> str:
    return 'Spearman R (test)' if task == 'meltome' else 'Test Q10 accuracy (%)'


def _task_title(task: str) -> str:
    return 'Meltome (FLIP human_cell)' if task == 'meltome' else 'DeepLoc setDeepLoc'


def _fmt(task: str) -> str:
    return '.3f' if task == 'meltome' else '.1f'


def cell_label(method, dc):
    if dc is None:
        return method
    return f'{method}\n d_c={dc}'


def group_rows(rows, key_fn):
    out = defaultdict(list)
    for r in rows:
        out[key_fn(r)].append(r)
    return out


def _suffix(task: str, multi_task: bool, zoom: bool) -> str:
    parts = []
    if multi_task: parts.append(task)
    if zoom: parts.append('zoom')
    return ('_' + '_'.join(parts)) if parts else ''


def _ylim(values, task, zoom):
    """Y-limits: a task-appropriate floor with a data-driven top.

    The floor is the preferred reference (50/0.5 standard, 75/0.65 zoom) unless
    the data dips below it, in which case we lower the floor so nothing is cut.
    """
    vals = [v for v in values if v is not None]
    pad = 0.04 if task == 'meltome' else 4.0
    floor_pref = (TASK_ZOOM_FLOOR if zoom else TASK_FLOOR)[task]
    if not vals:
        return (floor_pref, floor_pref + (0.3 if task == 'meltome' else 30))
    lo = min(floor_pref, min(vals) - pad)
    hi = max(vals) + pad
    return (lo, hi)


def _paper_baseline(task: str, args):
    if task == 'loc' and args.paper_loc is not None:
        return args.paper_loc, f'paper MLP+ProtT5 (≈{args.paper_loc:g}% Q10)'
    if task == 'meltome' and args.paper_meltome is not None:
        return args.paper_meltome, f'paper mean+MLP+ESM-1b (≈{args.paper_meltome:g})'
    return None, None


# ────────────────────────────────────────────────────────────────────────────
# Plot 1: per-seed bar chart
# ────────────────────────────────────────────────────────────────────────────
def plot_bars_per_seed(rows, task, out_dir, multi_task, args, zoom=False):
    metric_key, stderr_key = _metric_key(task)
    fmt = _fmt(task)
    seeds = sorted({r['seed'] for r in rows if r['seed'] is not None})
    for seed in seeds:
        seed_rows = [r for r in rows if r['seed'] == seed and r.get(metric_key) is not None]
        if not seed_rows:
            continue
        seed_rows = sorted(seed_rows, key=lambda r: (METHOD_ORDER.index(r['method']),
                                                       r['proj_dim'] or -1))
        labels, vals, errs, colors = [], [], [], []
        for r in seed_rows:
            labels.append(cell_label(r['method'], r['proj_dim']))
            vals.append(r[metric_key])
            errs.append(r.get(stderr_key) or 0.0)
            colors.append(METHOD_COLOR.get(r['method'], 'gray'))

        ylo, yhi = _ylim(vals, task, zoom)
        fig, ax = plt.subplots(figsize=(max(10, 0.92 * len(seed_rows)), 6.5))
        ax.set_ylim(ylo, yhi)
        xs = range(len(seed_rows))
        ax.bar(xs, vals, yerr=errs, color=colors, capsize=4,
               edgecolor='black', linewidth=0.5)
        # value labels above bars
        for i, v in enumerate(vals):
            ax.text(i, v + 0.012 * (yhi - ylo),
                    format(v, fmt), ha='center', va='bottom', fontsize=10)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlim(-0.6, len(seed_rows) - 0.4)
        ax.set_ylabel(_axis_label(task))
        ax.set_title(f'{_task_title(task)} — pooling sweep (seed {seed})')

        # paper baseline reference
        ref_val, ref_lbl = _paper_baseline(task, args)
        if ref_val is not None:
            ax.axhline(ref_val, linestyle=':', color='black', alpha=0.6,
                       linewidth=1.5, label=ref_lbl)
            ax.legend(loc='lower right')

        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        _footnote(fig, 'Error bars: bootstrap test stderr (200 resamples).')
        out = out_dir / f'bars_seed_{seed}{_suffix(task, multi_task, zoom)}.png'
        fig.savefig(out)
        plt.close(fig)
        print(f'wrote {out}')


# ────────────────────────────────────────────────────────────────────────────
# Plot 2: averaged across seeds
# ────────────────────────────────────────────────────────────────────────────
def plot_bars_averaged(rows, task, out_dir, multi_task, args, zoom=False):
    metric_key, _ = _metric_key(task)
    fmt = _fmt(task)
    rows = [r for r in rows if r.get(metric_key) is not None]
    if not rows:
        return
    grouped = group_rows(rows, lambda r: (r['method'], r['proj_dim']))
    entries = sorted(grouped.keys(),
                     key=lambda k: (METHOD_ORDER.index(k[0]), k[1] or -1))
    labels, means, stds, colors, ns = [], [], [], [], []
    for key in entries:
        accs = [r[metric_key] for r in grouped[key]]
        m, s = _stat(accs)
        if m is None:
            continue
        labels.append(cell_label(*key))
        means.append(m)
        stds.append(s)
        colors.append(METHOD_COLOR.get(key[0], 'gray'))
        ns.append(len(accs))

    if not means:
        return
    ylo, yhi = _ylim(means, task, zoom)
    fig, ax = plt.subplots(figsize=(max(10, 0.92 * len(means)), 6.5))
    ax.set_ylim(ylo, yhi)
    xs = range(len(means))
    ax.bar(xs, means, yerr=stds, color=colors, capsize=4,
           edgecolor='black', linewidth=0.5)
    n_label_y = ylo + 0.06 * (yhi - ylo)
    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m + 0.012 * (yhi - ylo),
                format(m, fmt), ha='center', va='bottom',
                fontsize=10, fontweight='bold')
        ax.text(i, n_label_y, f'n={n}', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlim(-0.6, len(means) - 0.4)
    ax.set_ylabel(_axis_label(task))
    ax.set_title(f'{_task_title(task)} — pooling sweep (avg over seeds)')

    ref_val, ref_lbl = _paper_baseline(task, args)
    if ref_val is not None:
        ax.axhline(ref_val, linestyle=':', color='black', alpha=0.6,
                   linewidth=1.5, label=ref_lbl)
        ax.legend(loc='lower right')

    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _footnote(fig, 'Error bars: stdev across seeds. Number on each bar = seed count.')
    out = out_dir / f'bars_averaged{_suffix(task, multi_task, zoom)}.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'wrote {out}')


# ────────────────────────────────────────────────────────────────────────────
# Plot 3: dimension curve averaged
# ────────────────────────────────────────────────────────────────────────────
def plot_dc_curve_averaged(rows, task, out_dir, multi_task, args, zoom=False):
    metric_key, _ = _metric_key(task)
    fig, ax = plt.subplots(figsize=(9, 6))
    drew_anything = False
    all_y = []

    for method in ['cov', 'cov_unsup', 'hybrid', 'la_cov']:
        pts = [r for r in rows if r['method'] == method and r['proj_dim'] is not None
                and r.get(metric_key) is not None]
        if not pts:
            continue
        by_dc = group_rows(pts, lambda r: r['proj_dim'])
        xs = sorted(by_dc)
        means = [_stat([r[metric_key] for r in by_dc[x]])[0] for x in xs]
        stds  = [_stat([r[metric_key] for r in by_dc[x]])[1] for x in xs]
        n = len(next(iter(by_dc.values())))
        ax.errorbar(xs, means, yerr=stds, marker='o',
                    label=f'{method} (n={n})', color=METHOD_COLOR[method],
                    capsize=4, linewidth=2.5, markersize=9)
        all_y.extend(means)
        drew_anything = True

    for baseline in ('mean', 'la'):
        pts = [r for r in rows if r['method'] == baseline and r.get(metric_key) is not None]
        if not pts:
            continue
        m, s = _stat([r[metric_key] for r in pts])
        style = '--' if baseline == 'mean' else ':'
        fmt_s = '.3f' if task == 'meltome' else '.2f'
        ax.axhline(m, linestyle=style, color=METHOD_COLOR[baseline],
                   label=f'{baseline} ({format(m, fmt_s)}±{format(s, fmt_s)}, n={len(pts)})',
                   linewidth=2.5)
        ax.fill_between([min(DC_VALUES) - 4, max(DC_VALUES) + 4],
                        m - s, m + s, color=METHOD_COLOR[baseline], alpha=0.10)
        all_y.append(m)
        drew_anything = True

    if not drew_anything:
        plt.close(fig)
        return

    ax.set_ylim(_ylim(all_y, task, zoom))

    ax.set_xlabel('d_c (projection dimension)')
    ax.set_ylabel(_axis_label(task))
    ax.set_title(f'{_task_title(task)} — metric vs d_c (avg over seeds)')
    ax.set_xticks(DC_VALUES)
    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _footnote(fig, 'Error bars / shaded band: stdev across seeds.')
    out = out_dir / f'dc_curve_averaged{_suffix(task, multi_task, zoom)}.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'wrote {out}')


# ────────────────────────────────────────────────────────────────────────────
# Plot 4: dc curve per seed
# ────────────────────────────────────────────────────────────────────────────
def plot_dc_curve_per_seed(rows, task, out_dir, multi_task, args, zoom=False):
    metric_key, stderr_key = _metric_key(task)
    seeds = sorted({r['seed'] for r in rows if r['seed'] is not None})
    if not seeds:
        return
    n_seeds = len(seeds)
    fig, axes = plt.subplots(1, n_seeds, figsize=(5.5 * n_seeds, 5.5), sharey=True)
    if n_seeds == 1:
        axes = [axes]

    all_y = []
    for ax, seed in zip(axes, seeds):
        seed_rows = [r for r in rows if r['seed'] == seed]
        for method in ['cov', 'cov_unsup', 'hybrid', 'la_cov']:
            pts = sorted([r for r in seed_rows
                          if r['method'] == method and r['proj_dim'] is not None
                          and r.get(metric_key) is not None],
                          key=lambda r: r['proj_dim'])
            if not pts:
                continue
            xs = [r['proj_dim'] for r in pts]
            ys = [r[metric_key] for r in pts]
            es = [r.get(stderr_key) or 0.0 for r in pts]
            ax.errorbar(xs, ys, yerr=es, marker='o', label=method,
                         color=METHOD_COLOR[method], capsize=3, linewidth=2)
            all_y.extend(ys)

        for baseline_method, style in (('mean', '--'), ('la', ':')):
            base = [r for r in seed_rows if r['method'] == baseline_method
                     and r.get(metric_key) is not None]
            if base:
                m = base[0][metric_key]
                fmt_s = '{:.3f}' if task == 'meltome' else '{:.1f}'
                ax.axhline(m, linestyle=style, color=METHOD_COLOR[baseline_method],
                           label=f'{baseline_method} ({fmt_s.format(m)})')
                all_y.append(m)

        ax.set_xticks(DC_VALUES)
        ax.set_xlabel('d_c')
        if ax is axes[0]:
            ax.set_ylabel(_axis_label(task))
        ax.set_title(f'seed {seed}')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10, loc='lower right')

    for ax in axes:
        ax.set_ylim(_ylim(all_y, task, zoom))

    fig.suptitle(f'{_task_title(task)} — metric vs d_c, per seed', fontsize=15, y=1.01)
    fig.tight_layout()
    out = out_dir / f'dc_curve_per_seed{_suffix(task, multi_task, zoom)}.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


# ────────────────────────────────────────────────────────────────────────────
# Plot 5: head-to-head at fixed d_c
# ────────────────────────────────────────────────────────────────────────────
def plot_head_to_head(rows, task, out_dir, multi_task, args, dc=32, zoom=False):
    metric_key, _ = _metric_key(task)
    fmt = _fmt(task)
    selected = []
    for method in METHOD_ORDER:
        if method in ('mean', 'la'):
            pts = [r for r in rows if r['method'] == method and r.get(metric_key) is not None]
        else:
            pts = [r for r in rows if r['method'] == method
                    and r['proj_dim'] == dc
                    and r.get(metric_key) is not None]
        if not pts:
            continue
        accs = [r[metric_key] for r in pts]
        m, s = _stat(accs)
        if m is None:
            continue
        selected.append((method, m, s, len(pts)))

    if not selected:
        return

    labels = [m for m, *_ in selected]
    means  = [m for _, m, *_ in selected]
    stds   = [s for *_, s, _ in selected]
    colors = [METHOD_COLOR.get(m, 'gray') for m in labels]
    ns     = [n for *_, n in selected]

    ylo, yhi = _ylim(means, task, zoom)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_ylim(ylo, yhi)
    xs = range(len(labels))
    ax.bar(xs, means, yerr=stds, color=colors, capsize=5,
           edgecolor='black', linewidth=0.6, width=0.65)

    n_label_y = ylo + 0.08 * (yhi - ylo)
    for i, (m, s, n) in enumerate(zip(means, stds, ns)):
        ax.text(i, m + 0.015 * (yhi - ylo),
                format(m, fmt), ha='center', va='bottom',
                fontsize=14, fontweight='bold')
        ax.text(i, n_label_y, f'±{format(s, fmt)}\nn={n}',
                ha='center', va='center',
                fontsize=11, color='white', fontweight='bold')

    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel(_axis_label(task))
    ax.set_title(f'Head-to-head — {_task_title(task)} (d_c={dc})')

    ref_val, ref_lbl = _paper_baseline(task, args)
    if ref_val is not None:
        ax.axhline(ref_val, linestyle=':', color='black', alpha=0.7,
                   linewidth=2, label=ref_lbl)
        ax.legend(loc='lower right', fontsize=11)

    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _footnote(fig, 'cov/hybrid/la_cov at the stated d_c. Error bars: stdev across seeds.')
    out = out_dir / f'head_to_head_dc{dc}{_suffix(task, multi_task, zoom)}.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'wrote {out}')


# ────────────────────────────────────────────────────────────────────────────
# Plot 6: per-seed scatter
# ────────────────────────────────────────────────────────────────────────────
def plot_seed_scatter(rows, task, out_dir, multi_task, args, zoom=False):
    metric_key, _ = _metric_key(task)
    rows = [r for r in rows if r.get(metric_key) is not None]
    if not rows:
        return
    grouped = group_rows(rows, lambda r: (r['method'], r['proj_dim']))
    entries = sorted(grouped.keys(),
                     key=lambda k: (METHOD_ORDER.index(k[0]), k[1] or -1))

    fig, ax = plt.subplots(figsize=(max(10, 0.92 * len(entries)), 6.5))
    seeds_sorted = sorted({r['seed'] for r in rows if r['seed'] is not None})
    seed_markers = {s: m for s, m in zip(seeds_sorted, 'osDv^P*X')}

    all_y = []
    for i, key in enumerate(entries):
        pts = grouped[key]
        for r in pts:
            ax.scatter(i, r[metric_key],
                        color=METHOD_COLOR.get(key[0], 'gray'),
                        marker=seed_markers.get(r['seed'], 'o'),
                        s=100, edgecolor='black', linewidth=0.8,
                        alpha=0.9, zorder=3)
            all_y.append(r[metric_key])
        m, _ = _stat([r[metric_key] for r in pts])
        if m is not None:
            ax.hlines(m, i - 0.3, i + 0.3, colors=METHOD_COLOR.get(key[0], 'gray'),
                       linewidth=3, zorder=2)

    labels = [cell_label(*k) for k in entries]
    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlim(-0.6, len(entries) - 0.4)
    ax.set_ylabel(_axis_label(task))
    ax.set_title(f'{_task_title(task)} — per-seed scatter')

    ax.set_ylim(_ylim(all_y, task, zoom))

    ax.grid(axis='y', alpha=0.3)

    handles = [plt.Line2D([], [], marker=seed_markers[s], linestyle='None',
                            color='gray', markeredgecolor='black',
                            label=f'seed {s}', markersize=11)
                for s in seeds_sorted]
    ax.legend(handles=handles, loc='lower right', fontsize=11)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _footnote(fig, 'Markers = individual seeds. Horizontal bar = mean across seeds.')
    out = out_dir / f'per_seed_scatter{_suffix(task, multi_task, zoom)}.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'wrote {out}')


# ────────────────────────────────────────────────────────────────────────────
# Text summary (unchanged)
# ────────────────────────────────────────────────────────────────────────────
def write_summary(rows, out_dir: Path):
    by_task = group_rows(rows, lambda r: r.get('task', 'loc'))
    plain_lines, md_lines = [], []
    for task, task_rows in sorted(by_task.items()):
        metric_key, _ = _metric_key(task)
        fmt = _fmt(task)
        metric_name = 'Spearman R' if task == 'meltome' else 'Q10'
        plain_lines.append(f'\n=== Task: {task} ({metric_name}) ===')
        plain_lines.append(f"{'method':10s} {'d_c':>5s} {'n':>3s} "
                            f"{'mean':>10s} {'std':>9s}  per-seed")
        plain_lines.append('-' * 80)
        md_lines.append(f'\n## Task: {task} ({metric_name})\n')
        md_lines.append(f'| method | d_c | n | mean | std | per-seed |')
        md_lines.append('|---|---:|---:|---:|---:|---|')

        grouped = group_rows(task_rows, lambda r: (r['method'], r['proj_dim']))
        entries = sorted(grouped.keys(),
                         key=lambda k: (METHOD_ORDER.index(k[0]), k[1] or -1))
        for method, dc in entries:
            accs = [r[metric_key] for r in grouped[(method, dc)] if r.get(metric_key) is not None]
            m, s = _stat(accs)
            if m is None:
                continue
            per_seed = ', '.join(f'{r["seed"]}:{format(r[metric_key], fmt)}'
                                  for r in sorted(grouped[(method, dc)],
                                                  key=lambda r: r['seed'] or 0)
                                  if r.get(metric_key) is not None)
            dc_str = '' if dc is None else str(dc)
            plain_lines.append(f'{method:10s} {dc_str:>5s} {len(accs):>3d} '
                                f'{format(m, ">10" + fmt)} {format(s, ">9" + fmt)}  {per_seed}')
            md_lines.append(f'| {method} | {dc_str} | {len(accs)} | {format(m, fmt)} | {format(s, fmt)} | {per_seed} |')

    (out_dir / 'summary_table.txt').write_text('\n'.join(plain_lines) + '\n')
    (out_dir / 'summary_table.md').write_text('\n'.join(md_lines) + '\n')
    print(f'wrote summary_table.txt and summary_table.md')


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=Path, default=None)
    parser.add_argument('--csv', type=Path, default=None)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--head-to-head-dc', type=int, default=32)
    parser.add_argument('--zoom', action='store_true',
                        help='Also produce a zoomed-in version of each plot '
                             '(suffix "_zoom").')
    parser.add_argument('--paper-loc', type=float, default=None,
                        help='Q10 baseline to draw as horizontal reference (e.g. 76).')
    parser.add_argument('--paper-meltome', type=float, default=None,
                        help='Spearman R baseline to draw as horizontal reference (e.g. 0.65).')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Only use these seeds (e.g. --seeds 657 921 969). '
                             'Restricts every method to the same seed set so n matches.')
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
        out_dir = args.out or csv_path.parent
    elif args.sweep:
        csv_path = args.sweep / 'results.csv'
        out_dir = args.out or args.sweep
    else:
        csv_path = PROJECT_ROOT / 'results.csv'
        out_dir = args.out or PROJECT_ROOT

    if not csv_path.exists():
        raise FileNotFoundError(
            f'{csv_path} not found. Run scripts/collect_results.py first.')
    out_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(SLIDE_RC)

    rows = read_csv(csv_path)
    if not rows:
        print('No rows in CSV.')
        return

    # This script plots the d_c sweep only. Drop layer-sweep rows (non-empty
    # `layer`); use scripts/plot_layers.py for those. Both share results.csv.
    before = len(rows)
    rows = [r for r in rows if not (r.get('layer') or '').strip()]
    if len(rows) != before:
        print(f'Ignoring {before - len(rows)} layer-sweep row(s) '
              f'(use analysis/plot_layers.py): {before} -> {len(rows)}')

    if args.seeds:
        keep = set(args.seeds)
        before = len(rows)
        rows = [r for r in rows if r['seed'] in keep]
        print(f'Filtered to seeds {sorted(keep)}: {before} -> {len(rows)} rows')

    tasks_present = sorted({r.get('task', 'loc') for r in rows})
    multi_task = len(tasks_present) > 1
    print(f'Loaded {len(rows)} rows from {csv_path}')
    print(f'Tasks: {tasks_present}')
    print(f'Output dir: {out_dir}')
    print(f'Zoom variants: {args.zoom}\n')

    for task in tasks_present:
        task_rows = [r for r in rows if r.get('task', 'loc') == task]
        plot_bars_per_seed(task_rows, task, out_dir, multi_task, args, zoom=False)
        plot_bars_averaged(task_rows, task, out_dir, multi_task, args, zoom=False)
        plot_dc_curve_averaged(task_rows, task, out_dir, multi_task, args, zoom=False)
        plot_dc_curve_per_seed(task_rows, task, out_dir, multi_task, args, zoom=False)
        plot_head_to_head(task_rows, task, out_dir, multi_task, args,
                          dc=args.head_to_head_dc, zoom=False)
        plot_seed_scatter(task_rows, task, out_dir, multi_task, args, zoom=False)

        if args.zoom:
            plot_bars_per_seed(task_rows, task, out_dir, multi_task, args, zoom=True)
            plot_bars_averaged(task_rows, task, out_dir, multi_task, args, zoom=True)
            plot_dc_curve_averaged(task_rows, task, out_dir, multi_task, args, zoom=True)
            plot_dc_curve_per_seed(task_rows, task, out_dir, multi_task, args, zoom=True)
            plot_head_to_head(task_rows, task, out_dir, multi_task, args,
                              dc=args.head_to_head_dc, zoom=True)
            plot_seed_scatter(task_rows, task, out_dir, multi_task, args, zoom=True)

    write_summary(rows, out_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
