"""
Produce a comprehensive set of plots from a sweep's results.csv.

Generates:
    bars_seed_<S>.png         one bar chart per seed
    bars_averaged.png         bars averaged across seeds (mean ± std error bars)
    dc_curve_averaged.png     metric vs d_c, averaged across seeds, error bars
    dc_curve_per_seed.png     metric vs d_c with one panel per seed
    head_to_head.png          head-to-head at d_c=32 + LA + la_cov + mean
    per_seed_scatter.png      scatter of seeds per (method, d_c) with mean bars
    summary_table.txt         plain-text tabular summary
    summary_table.md          markdown-friendly version

Handles both 'loc' (Q10 accuracy) and 'meltome' (Spearman R) tasks. If both are
present in the CSV, produces a per-task set of plots, suffixed with the task name.

Usage:
    python scripts/plot_all.py --sweep sweeps/overnight_full_dc_sweep
    python scripts/plot_all.py --csv path/to/results.csv --out path/to/dir
"""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

METHOD_ORDER = ['mean', 'cov', 'hybrid', 'la', 'la_cov']
METHOD_COLOR = {
    'mean':   '#2ca02c',
    'cov':    '#1f77b4',
    'hybrid': '#d62728',
    'la':     '#ff7f0e',
    'la_cov': '#9467bd',
}
DC_VALUES = [8, 16, 24, 32, 48]


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
    return 'Spearman R' if task == 'meltome' else 'Test Q10 accuracy (%)'


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


def _task_suffix(task: str, multi_task: bool) -> str:
    return f'_{task}' if multi_task else ''


# ────────────────────────────────────────────────────────────────────────────
# Plot 1: per-seed bar chart
# ────────────────────────────────────────────────────────────────────────────
def plot_bars_per_seed(rows, task: str, out_dir: Path, multi_task: bool):
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

        fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(seed_rows)), 5), dpi=150)
        xs = range(len(seed_rows))
        ax.bar(xs, vals, yerr=errs, color=colors, capsize=4,
               edgecolor='black', linewidth=0.5)
        margin = 0.005 if task == 'meltome' else 0.2
        for i, v in enumerate(vals):
            ax.text(i, v + margin, format(v, fmt), ha='center', va='bottom', fontsize=9)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(_axis_label(task))
        title_task = 'Meltome (FLIP)' if task == 'meltome' else 'DeepLoc setDeepLoc'
        ax.set_title(f'{title_task} — pooling sweep at seed {seed}')
        margin2 = 0.05 if task == 'meltome' else 3
        ax.set_ylim(bottom=max(0, min(vals) - margin2), top=max(vals) + margin2 / 2)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        out = out_dir / f'bars_seed_{seed}{_task_suffix(task, multi_task)}.png'
        fig.savefig(out)
        plt.close(fig)
        print(f'wrote {out}')


# ────────────────────────────────────────────────────────────────────────────
# Plot 2: averaged across seeds
# ────────────────────────────────────────────────────────────────────────────
def plot_bars_averaged(rows, task: str, out_dir: Path, multi_task: bool):
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
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(means)), 5), dpi=150)
    xs = range(len(means))
    ax.bar(xs, means, yerr=stds, color=colors, capsize=4,
           edgecolor='black', linewidth=0.5)
    label_margin = 0.005 if task == 'meltome' else 0.2
    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m + label_margin, format(m, fmt), ha='center', va='bottom', fontsize=9)
        ax.text(i, max(0, m - label_margin * 4), f'n={n}', ha='center', va='top',
                fontsize=7, color='white', fontweight='bold')
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(_axis_label(task))
    title_task = 'Meltome (FLIP)' if task == 'meltome' else 'DeepLoc setDeepLoc'
    ax.set_title(f'{title_task} — pooling sweep, averaged across seeds (bars = stdev)')
    margin = 0.05 if task == 'meltome' else 3
    ax.set_ylim(bottom=max(0, min(means) - margin), top=max(means) + margin / 2)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out = out_dir / f'bars_averaged{_task_suffix(task, multi_task)}.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'wrote {out}')


# ────────────────────────────────────────────────────────────────────────────
# Plot 3: dimension curve averaged across seeds
# ────────────────────────────────────────────────────────────────────────────
def plot_dc_curve_averaged(rows, task: str, out_dir: Path, multi_task: bool):
    metric_key, _ = _metric_key(task)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    drew_anything = False

    for method in ['cov', 'hybrid', 'la_cov']:
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
                    capsize=4, linewidth=2, markersize=7)
        drew_anything = True

    mean_pts = [r for r in rows if r['method'] == 'mean' and r.get(metric_key) is not None]
    if mean_pts:
        m, s = _stat([r[metric_key] for r in mean_pts])
        ax.axhline(m, linestyle='--', color=METHOD_COLOR['mean'],
                   label=f'mean ({m:.3f}±{s:.3f}, n={len(mean_pts)})' if task == 'meltome'
                   else f'mean ({m:.2f}±{s:.2f}%, n={len(mean_pts)})',
                   linewidth=2)
        ax.fill_between([min(DC_VALUES) - 4, max(DC_VALUES) + 4],
                        m - s, m + s, color=METHOD_COLOR['mean'], alpha=0.12)
        drew_anything = True

    la_pts = [r for r in rows if r['method'] == 'la' and r.get(metric_key) is not None]
    if la_pts:
        m, s = _stat([r[metric_key] for r in la_pts])
        ax.axhline(m, linestyle=':', color=METHOD_COLOR['la'],
                   label=f'LA ({m:.3f}±{s:.3f}, n={len(la_pts)})' if task == 'meltome'
                   else f'LA ({m:.2f}±{s:.2f}%, n={len(la_pts)})',
                   linewidth=2)
        ax.fill_between([min(DC_VALUES) - 4, max(DC_VALUES) + 4],
                        m - s, m + s, color=METHOD_COLOR['la'], alpha=0.12)
        drew_anything = True

    if not drew_anything:
        plt.close(fig)
        return

    ax.set_xlabel('d_c (projection dimension)')
    ax.set_ylabel(_axis_label(task))
    title_task = 'Meltome (FLIP)' if task == 'meltome' else 'DeepLoc setDeepLoc'
    ax.set_title(f'{title_task} — metric vs d_c, averaged across seeds')
    ax.set_xticks(DC_VALUES)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out_dir / f'dc_curve_averaged{_task_suffix(task, multi_task)}.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'wrote {out}')


# ────────────────────────────────────────────────────────────────────────────
# Plot 4: dimension curve, one line per seed
# ────────────────────────────────────────────────────────────────────────────
def plot_dc_curve_per_seed(rows, task: str, out_dir: Path, multi_task: bool):
    metric_key, stderr_key = _metric_key(task)
    seeds = sorted({r['seed'] for r in rows if r['seed'] is not None})
    if not seeds:
        return
    n_seeds = len(seeds)
    fig, axes = plt.subplots(1, n_seeds, figsize=(5 * n_seeds, 5), dpi=150, sharey=True)
    if n_seeds == 1:
        axes = [axes]

    for ax, seed in zip(axes, seeds):
        seed_rows = [r for r in rows if r['seed'] == seed]
        for method in ['cov', 'hybrid', 'la_cov']:
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

        for baseline_method, style in (('mean', '--'), ('la', ':')):
            base = [r for r in seed_rows if r['method'] == baseline_method
                     and r.get(metric_key) is not None]
            if base:
                m = base[0][metric_key]
                fmt = '{:.3f}' if task == 'meltome' else '{:.1f}'
                ax.axhline(m, linestyle=style, color=METHOD_COLOR[baseline_method],
                           label=f'{baseline_method} ({fmt.format(m)})')

        ax.set_xticks(DC_VALUES)
        ax.set_xlabel('d_c')
        if ax is axes[0]:
            ax.set_ylabel(_axis_label(task))
        ax.set_title(f'seed {seed}')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')

    title_task = 'Meltome (FLIP)' if task == 'meltome' else 'DeepLoc setDeepLoc'
    fig.suptitle(f'{title_task} — metric vs d_c, per seed', y=1.02)
    fig.tight_layout()
    out = out_dir / f'dc_curve_per_seed{_task_suffix(task, multi_task)}.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


# ────────────────────────────────────────────────────────────────────────────
# Plot 5: head-to-head at d_c=32
# ────────────────────────────────────────────────────────────────────────────
def plot_head_to_head(rows, task: str, out_dir: Path, multi_task: bool, dc: int = 32):
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

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    xs = range(len(labels))
    ax.bar(xs, means, yerr=stds, color=colors, capsize=5,
           edgecolor='black', linewidth=0.6, width=0.65)
    label_margin = 0.005 if task == 'meltome' else 0.2
    err_margin = 0.015 if task == 'meltome' else 0.5
    for i, (m, s, n) in enumerate(zip(means, stds, ns)):
        ax.text(i, m + label_margin, format(m, fmt), ha='center', va='bottom',
                fontsize=11, fontweight='bold')
        ax.text(i, max(0, m - err_margin), f'±{s:.3f}\nn={n}' if task == 'meltome'
                else f'±{s:.2f}\nn={n}',
                ha='center', va='top',
                fontsize=9, color='white', fontweight='bold')

    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(_axis_label(task))
    title_task = 'Meltome (FLIP)' if task == 'meltome' else 'DeepLoc setDeepLoc'
    ax.set_title(f'Head-to-head on {title_task} (cov/hybrid at d_c={dc})')
    margin = 0.05 if task == 'meltome' else 3
    ax.set_ylim(bottom=max(0, min(means) - margin), top=max(means) + margin / 2)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out = out_dir / f'head_to_head_dc{dc}{_task_suffix(task, multi_task)}.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'wrote {out}')


# ────────────────────────────────────────────────────────────────────────────
# Plot 6: per-seed scatter
# ────────────────────────────────────────────────────────────────────────────
def plot_seed_scatter(rows, task: str, out_dir: Path, multi_task: bool):
    metric_key, _ = _metric_key(task)
    rows = [r for r in rows if r.get(metric_key) is not None]
    if not rows:
        return
    grouped = group_rows(rows, lambda r: (r['method'], r['proj_dim']))
    entries = sorted(grouped.keys(),
                     key=lambda k: (METHOD_ORDER.index(k[0]), k[1] or -1))

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(entries)), 5.5), dpi=150)
    seeds_sorted = sorted({r['seed'] for r in rows if r['seed'] is not None})
    seed_markers = {s: m for s, m in zip(seeds_sorted, 'osDv^P*X')}

    for i, key in enumerate(entries):
        pts = grouped[key]
        for r in pts:
            ax.scatter(i, r[metric_key],
                        color=METHOD_COLOR.get(key[0], 'gray'),
                        marker=seed_markers.get(r['seed'], 'o'),
                        s=80, edgecolor='black', linewidth=0.6,
                        alpha=0.85, zorder=3)
        m, _ = _stat([r[metric_key] for r in pts])
        if m is not None:
            ax.hlines(m, i - 0.3, i + 0.3, colors=METHOD_COLOR.get(key[0], 'gray'),
                       linewidth=2.5, zorder=2)

    labels = [cell_label(*k) for k in entries]
    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(_axis_label(task))
    title_task = 'Meltome (FLIP)' if task == 'meltome' else 'DeepLoc setDeepLoc'
    ax.set_title(f'{title_task} — per-seed scatter (markers = seeds, bars = mean)')
    ax.grid(axis='y', alpha=0.3)

    handles = [plt.Line2D([], [], marker=seed_markers[s], linestyle='None',
                            color='gray', markeredgecolor='black',
                            label=f'seed {s}', markersize=10)
                for s in seeds_sorted]
    ax.legend(handles=handles, loc='lower right', fontsize=9)

    fig.tight_layout()
    out = out_dir / f'per_seed_scatter{_task_suffix(task, multi_task)}.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'wrote {out}')


# ────────────────────────────────────────────────────────────────────────────
# Text summary
# ────────────────────────────────────────────────────────────────────────────
def write_summary(rows, out_dir: Path):
    by_task = group_rows(rows, lambda r: r.get('task', 'loc'))
    plain_lines = []
    md_lines = []
    for task, task_rows in sorted(by_task.items()):
        metric_key, _ = _metric_key(task)
        fmt = _fmt(task)
        metric_name = 'Spearman R' if task == 'meltome' else 'Q10'
        plain_lines.append(f'\n=== Task: {task} ({metric_name}) ===')
        plain_lines.append(f"{'method':10s} {'d_c':>5s} {'n':>3s} "
                            f"{'mean':>10s} {'std':>9s}  per-seed")
        plain_lines.append('-' * 80)
        md_lines.append(f'\n## Task: {task} ({metric_name})\n')
        md_lines.append(f'| method | d_c | n | {metric_name} mean | {metric_name} std | per-seed |')
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

    txt_out = out_dir / 'summary_table.txt'
    txt_out.write_text('\n'.join(plain_lines) + '\n')
    print(f'wrote {txt_out}')

    md_out = out_dir / 'summary_table.md'
    md_out.write_text('\n'.join(md_lines) + '\n')
    print(f'wrote {md_out}')


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=Path, default=None)
    parser.add_argument('--csv', type=Path, default=None)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--head-to-head-dc', type=int, default=32)
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

    rows = read_csv(csv_path)
    if not rows:
        print('No rows in CSV.')
        return

    tasks_present = sorted({r.get('task', 'loc') for r in rows})
    multi_task = len(tasks_present) > 1
    print(f'Loaded {len(rows)} rows from {csv_path}')
    print(f'Tasks: {tasks_present}')
    print(f'Output dir: {out_dir}\n')

    for task in tasks_present:
        task_rows = [r for r in rows if r.get('task', 'loc') == task]
        plot_bars_per_seed(task_rows, task, out_dir, multi_task)
        plot_bars_averaged(task_rows, task, out_dir, multi_task)
        plot_dc_curve_averaged(task_rows, task, out_dir, multi_task)
        plot_dc_curve_per_seed(task_rows, task, out_dir, multi_task)
        plot_head_to_head(task_rows, task, out_dir, multi_task, dc=args.head_to_head_dc)
        plot_seed_scatter(task_rows, task, out_dir, multi_task)

    write_summary(rows, out_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
