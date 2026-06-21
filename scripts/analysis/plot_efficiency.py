"""
Size vs. performance efficiency curve (minimum-viable deliverable #2).

X = the pooled feature dimension fed to the FFN head, on a LOG scale:
    mean              -> d            (1024)
    cov / cov_unsup   -> d_c^2
    hybrid            -> d + d_c^2
Y = downstream metric (Q10 % for loc, Spearman R for meltome), averaged over seeds.

Supervised covariance is drawn as a curve over its d_c values; mean pooling as a
horizontal line with a shaded ±stdev band; a vertical dashed line marks the
equal-size crossover (cov dim = mean dim, d_c^2 = d, i.e. d_c = 32). Points left
of that line use *fewer* dimensions than mean pooling.

Two variants are written per task:
    efficiency_curve[_<task>].png             supervised cov vs mean
    efficiency_curve[_<task>]_with_unsup.png  + unsupervised cov curve

Reads the master results.csv (d_c-sweep rows; layer-sweep rows are ignored).

Usage:
    python scripts/analysis/plot_efficiency.py --csv results/results.csv \
        --out results/figures/dc_sweep --seeds 657 921 969
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                    if (p / 'configs').is_dir() and (p / 'models').is_dir())

EMB_DIM = 1024  # ProtT5-XL per-residue dim = mean-pooled output dim
DC_VALUES = [8, 16, 24, 32, 48]
# Shared repo colour scheme (matches plot_dc.py / plot_layers.py)
COV_COLOR = '#1f77b4'      # blue   — supervised covariance
HYBRID_COLOR = '#d62728'   # red    — hybrid (mean + cov)
UNSUP_COLOR = '#9467bd'    # purple — unsupervised covariance
MEAN_COLOR = '#2ca02c'     # green  — mean pooling

SLIDE_RC = {
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
    'figure.dpi': 150,
}


def feature_dim(method: str, dc: int) -> int:
    if method == 'mean':
        return EMB_DIM
    if method == 'hybrid':
        return EMB_DIM + dc * dc        # mean ++ covariance
    return dc * dc                      # cov / cov_unsup


def read_csv(path: Path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['proj_dim'] = int(r['proj_dim']) if r.get('proj_dim') else None
        r['seed'] = int(r['seed']) if r.get('seed') else None
        for col in ('test_acc', 'test_spearman'):
            if col in r:
                r[col] = float(r[col]) if r[col] not in (None, '') else None
    return rows


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, 0.0
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def _metric_key(task: str) -> str:
    return 'test_spearman' if task == 'meltome' else 'test_acc'


def _axis_label(task: str) -> str:
    return 'Spearman R (test)' if task == 'meltome' else 'Test Q10 accuracy (%)'


def _task_title(task: str) -> str:
    return 'Meltome (FLIP human_cell)' if task == 'meltome' else 'DeepLoc setDeepLoc'


def plot_task(rows, task, out_dir, multi_task, include_unsup):
    metric_key = _metric_key(task)
    fmt = '.3f' if task == 'meltome' else '.1f'
    fig, ax = plt.subplots(figsize=(8, 6))

    series = [('cov', 'Supervised covariance', COV_COLOR),
              ('hybrid', 'Hybrid (mean + cov)', HYBRID_COLOR)]
    if include_unsup:
        series.append(('cov_unsup', 'Unsupervised covariance', UNSUP_COLOR))

    drew = False
    for method, label, color in series:
        by_dc = defaultdict(list)
        for r in rows:
            if r['method'] == method and r['proj_dim'] and r.get(metric_key) is not None:
                by_dc[r['proj_dim']].append(r[metric_key])
        if not by_dc:
            continue
        dcs = sorted(by_dc)
        xs = [feature_dim(method, dc) for dc in dcs]
        ys, es = [], []
        for dc in dcs:
            m, s = _mean_std(by_dc[dc])
            ys.append(m)
            es.append(s)
        ax.errorbar(xs, ys, yerr=es, marker='o', label=label, color=color,
                    linewidth=2.5, markersize=8, capsize=4)
        # hybrid's points (dim = d + d_c²) pile up on the log axis, so its d_c
        # labels would overlap cov's — leave them out (legend identifies it).
        if method != 'hybrid':
            for dc, x, y in zip(dcs, xs, ys):
                ax.annotate(f'd_c={dc}', (x, y), textcoords='offset points', xytext=(0, 11),
                            ha='center', fontsize=9, color='black')
        drew = True

    if not drew:
        plt.close(fig)
        return

    # mean pooling: horizontal line + shaded ±stdev band
    m_mean, s_mean = _mean_std([r[metric_key] for r in rows
                                if r['method'] == 'mean' and r.get(metric_key) is not None])
    if m_mean is not None:
        ax.axhline(m_mean, color=MEAN_COLOR, linewidth=2,
                   label=f'Mean pooling ({format(m_mean, fmt)})')
        ax.axhspan(m_mean - s_mean, m_mean + s_mean, color=MEAN_COLOR, alpha=0.15)

    # equal-size crossover: cov dim (d_c^2) == mean dim (d)
    ax.axvline(EMB_DIM, linestyle='--', color='gray', linewidth=1.8,
               label=f'Equal size ({EMB_DIM})')

    ax.set_xscale('log', base=2)
    ticks = [dc * dc for dc in DC_VALUES]               # 64, 256, 576, 1024, 2304
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{int(v)}'))
    ax.minorticks_off()
    ax.set_xlabel('Pooled embedding dimension (log scale)')
    ax.set_ylabel(_axis_label(task))
    ax.set_title(f'{_task_title(task)}: size vs. performance')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    suffix = (f'_{task}' if multi_task else '') + ('_with_unsup' if include_unsup else '')
    out = out_dir / f'efficiency_curve{suffix}.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'wrote {out}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=PROJECT_ROOT / 'results' / 'results.csv')
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f'{args.csv} not found. Run scripts/analysis/collect_results.py first.')
    out_dir = args.out or (args.csv.parent / 'figures' / 'dc_sweep')
    out_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(SLIDE_RC)

    rows = read_csv(args.csv)
    rows = [r for r in rows if not (r.get('layer') or '').strip()]   # d_c sweep only
    if args.seeds:
        keep = set(args.seeds)
        rows = [r for r in rows if r['seed'] in keep]

    tasks = sorted({r.get('task', 'loc') for r in rows})
    multi_task = len(tasks) > 1
    for task in tasks:
        task_rows = [r for r in rows if r.get('task', 'loc') == task]
        plot_task(task_rows, task, out_dir, multi_task, include_unsup=False)
        plot_task(task_rows, task, out_dir, multi_task, include_unsup=True)
    print('Done.')


if __name__ == '__main__':
    main()
