"""
Covariance-matrix visualization (Target Delivery deliverable).

For one DeepLoc class, pick one protein for each of the 4 cov-vs-mean outcome
scenarios and show, per protein, the learned d_c x d_c covariance embedding as a
heatmap next to the mean-pooled vector as a 1-D strip — contrasting the 2-D
structure covariance keeps against what mean pooling collapses away.

The 4 scenarios (cov head correct?, mean head correct?):
    both correct | cov correct, mean wrong | mean correct, cov wrong | both wrong

Per-protein correctness comes from each head's
`results_array_test_set_after_train.npy` ([pred, target] in test-set order); the
class + accession come from the test FASTA at the same index.

Run `--list` first to verify the scenarios exist in the data (prints a
per-class count table), then again without it (or with --class) to plot.

Usage:
    python scripts/analysis/plot_covariance.py --list          # verify scenario counts
    python scripts/analysis/plot_covariance.py                 # auto-pick 3 fullest classes
    python scripts/analysis/plot_covariance.py --classes Nucleus Cytoplasm Mitochondrion
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from Bio import SeqIO

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                    if (p / 'configs').is_dir() and (p / 'models').is_dir())
sys.path.insert(0, str(PROJECT_ROOT))

import torch                              # noqa: E402
from models import PoolingFFN             # noqa: E402
from utils.general import LOCALIZATION    # noqa: E402

RESULTS_FILE = 'results_array_test_set_after_train.npy'
CKPT_FILE = 'checkpoint.pt'

# (label, cov_correct, mean_correct)
SCENARIOS = [
    ('both correct',            True,  True),
    ('cov right, mean wrong',   True,  False),
    ('mean right, cov wrong',   False, True),
    ('both wrong',              False, False),
]


def find_run(heads_dir: Path, pattern: str) -> Path:
    matches = [Path(p) for p in sorted(glob.glob(str(heads_dir / pattern)))
               if (Path(p) / RESULTS_FILE).exists()]
    if not matches:
        raise FileNotFoundError(f'no run with {RESULTS_FILE} matching {pattern} in {heads_dir}')
    return matches[0]


def parse_fasta(fasta_path: Path):
    """Return list of (h5_key, class_index, accession) in FASTA order."""
    out = []
    for rec in SeqIO.parse(str(fasta_path), 'fasta'):
        parts = rec.description.split(' ')
        class_name = parts[2].split('-')[0] if len(parts) > 2 else ''
        acc = parts[1] if len(parts) > 1 else rec.id
        idx = LOCALIZATION.index(class_name) if class_name in LOCALIZATION else -1
        out.append((str(rec.id), idx, acc))
    return out


def load_model(run_dir: Path, proj_dim: int):
    ta = yaml.safe_load((run_dir / 'train_arguments.yaml').read_text())
    mp = ta.get('model_parameters', {})
    model = PoolingFFN(embeddings_dim=mp.get('embeddings_dim', 1024),
                       pooling='cov', proj_dim=proj_dim,
                       output_dim=mp.get('output_dim', len(LOCALIZATION)))
    ckpt = torch.load(run_dir / CKPT_FILE, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.eval()


def cov_and_mean(model, h5, key, proj_dim):
    arr = h5[key][:]                                   # [L, d]
    x = torch.from_numpy(arr).float().t().unsqueeze(0)  # [1, d, L]
    mask = torch.ones((1, x.shape[-1]), dtype=torch.bool)
    with torch.no_grad():
        flat_C = model._bilinear_cov(x, mask).squeeze(0).numpy()
    return flat_C.reshape(proj_dim, proj_dim), arr.mean(axis=0)


def scenario_of(cov_ok, mean_ok):
    for i, (_, c, m) in enumerate(SCENARIOS):
        if cov_ok == c and mean_ok == m:
            return i
    return None


def render_columns(columns, title, out):
    """Render a row of columns; each = (col_title, C, mean_vec). C may be None.

    Uses a 3-row gridspec (matrix / mean-heat strip / mean-trend) so all three
    sit in the SAME column and therefore share an identical width. The colorbar
    is an inset anchored to the matrix, so it never steals column space.
    """
    cmax = max((np.abs(C).max() for _, C, _ in columns if C is not None), default=1.0)
    n = len(columns)
    fig, axes = plt.subplots(3, n, figsize=(4.0 * n + 0.8, 6.0),
                             gridspec_kw={'height_ratios': [6, 0.5, 2.2], 'hspace': 0.15},
                             squeeze=False)
    im, last_ax = None, None
    for c, (col_title, C, mean_vec) in enumerate(columns):
        ax, ax_strip, ax_line = axes[0][c], axes[1][c], axes[2][c]
        if C is None:
            for a in (ax, ax_strip, ax_line):
                a.axis('off')
            ax.text(0.5, 0.5, f'{col_title}\n(none)', ha='center', va='center', fontsize=10)
            continue
        im = ax.imshow(C, cmap='RdBu_r', vmin=-cmax, vmax=cmax, aspect='auto')
        # for a single-column figure the suptitle already names the scenario
        ax.set_title(col_title if n > 1 else col_title.split('\n')[-1], fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        last_ax = ax
        mmax = float(np.abs(mean_vec).max()) or 1.0
        ax_strip.imshow(mean_vec[None, :], cmap='RdBu_r', aspect='auto', vmin=-mmax, vmax=mmax)
        ax_strip.set_xticks([]); ax_strip.set_yticks([])
        ax_line.plot(np.arange(len(mean_vec)), mean_vec, color='#2a6f77', linewidth=0.7)
        ax_line.axhline(0, color='gray', linewidth=0.6, alpha=0.6)
        ax_line.set_xlim(0, len(mean_vec) - 1)
        ax_line.margins(x=0)
        ax_line.tick_params(axis='y', labelsize=7)
        ax_line.set_xlabel('mean-pooled dim (1024)', fontsize=8)
        if c == 0:                                             # row captions, left column only
            ax.set_ylabel('learned covariance', fontsize=9)
            ax_strip.set_ylabel('mean\n(heat)', fontsize=7, rotation=0, ha='right', va='center')
            ax_line.set_ylabel('mean value', fontsize=8)

    if im is not None:
        cax = fig.add_axes((0.92, 0.30, 0.018, 0.42))   # vertically centred in the figure
        fig.colorbar(im, cax=cax, label='covariance entry')
    fig.suptitle(title, fontsize=13)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


def plot_class(cls_idx, buckets, proteins, C_by, mean_by, args):
    """Summary figure (4 scenario columns) + optional per-scenario detail figures
    (the first --detail individual proteins of each scenario, one column each)."""
    cname = LOCALIZATION[cls_idx]
    fname = cname.replace('/', '_')
    cls_buckets = buckets.get(cls_idx, [[], [], [], []])
    base = f'(d_c={args.dc}, seed {args.seed}{", baseline-subtracted" if args.deviation else ""})'

    # summary columns: one per scenario (bucket average, or single protein)
    columns, labels = [], []
    for s, (lab, *_ ) in enumerate(SCENARIOS):
        labels.append(lab)
        rows = cls_buckets[s]
        if not rows:
            columns.append((lab, None, None))
            continue
        if args.reduce == 'average':
            C = np.mean([C_by[i] for i in rows], axis=0)
            mvec = np.mean([mean_by[i] for i in rows], axis=0)
            tag = f'n={len(rows)}'
        else:
            idx = rows[0]
            C, mvec, tag = C_by[idx], mean_by[idx], proteins[idx][2]
        columns.append((f'{lab}\n{tag}', C, mvec))

    # combined summary (all 4 scenarios)
    render_columns(columns, f'Learned covariance vs. mean pooling — {cname} {base}',
                   args.out_dir / f'covariance_{args.reduce}_{fname}.png')

    # one standalone image per scenario (same summary column on its own)
    if args.split:
        for lab, col in zip(labels, columns):
            if col[1] is None:
                continue
            slug = lab.split(',')[0].strip().replace(' ', '_')
            render_columns([col], f'{cname} — {lab} {base}',
                           args.out_dir / f'covariance_{args.reduce}_{fname}_{slug}.png')

    # detail: per scenario, the first N individual proteins as columns
    if args.detail:
        for s, (lab, *_ ) in enumerate(SCENARIOS):
            rows = cls_buckets[s][:args.detail]
            if not rows:
                continue
            slug = lab.split(',')[0].strip().replace(' ', '_')
            cols = [(proteins[i][2], C_by[i], mean_by[i]) for i in rows]
            render_columns(cols, f'{cname} — {lab} (individual proteins) {base}',
                           args.out_dir / f'covariance_detail_{fname}_{slug}.png')
            # ... and each of those proteins as its own standalone image
            if args.split:
                for i in rows:
                    acc = proteins[i][2]
                    render_columns([(acc, C_by[i], mean_by[i])],
                                   f'{cname} — {lab} — {acc} {base}',
                                   args.out_dir / f'covariance_protein_{fname}_{slug}_{acc}.png')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--heads-dir', type=Path,
                   default=PROJECT_ROOT / 'checkpoints/heads/supervised/deeploc')
    p.add_argument('--seed', type=int, default=657)
    p.add_argument('--dc', type=int, default=48)
    p.add_argument('--test-h5', type=Path, default=PROJECT_ROOT / 'data_files/deeploc_test_set_layer24.h5')
    p.add_argument('--test-fasta', type=Path, default=PROJECT_ROOT / 'data_files/deeploc_test_set_remapped.fasta')
    p.add_argument('--classes', dest='classes', nargs='+', default=None,
                   help='Class names; auto-picks the 3 fullest all-4 classes if omitted.')
    p.add_argument('--out-dir', dest='out_dir', type=Path,
                   default=PROJECT_ROOT / 'results/figures/covariance')
    p.add_argument('--reduce', choices=['average', 'single'], default='average',
                   help="summary columns: 'average' = mean covariance over each "
                        "(class,scenario) bucket; 'single' = one representative protein.")
    p.add_argument('--deviation', action='store_true',
                   help='Subtract the global-average covariance so only the distinctive '
                        'structure remains (removes the shared baseline).')
    p.add_argument('--split', action='store_true',
                   help='Also write each scenario as its own standalone single-column image.')
    p.add_argument('--detail', type=int, default=0, metavar='N',
                   help='Also write a per-scenario figure with the first N individual '
                        'proteins as columns (0 = off).')
    p.add_argument('--list', action='store_true', help='Only print the per-class scenario counts and exit.')
    args = p.parse_args()

    cov_run = find_run(args.heads_dir, f'PoolingFFN_loc_cov_dc{args.dc}_seed{args.seed}_*')
    mean_run = find_run(args.heads_dir, f'PoolingFFN_loc_mean_seed{args.seed}_*')
    cov_res = np.load(cov_run / RESULTS_FILE)
    mean_res = np.load(mean_run / RESULTS_FILE)
    proteins = parse_fasta(args.test_fasta)

    if not (len(proteins) == len(cov_res) == len(mean_res)):
        print(f'WARNING: length mismatch — fasta={len(proteins)}, cov={len(cov_res)}, '
              f'mean={len(mean_res)}. Alignment may be off.')
    n = min(len(proteins), len(cov_res), len(mean_res))

    # bucket[class_idx][scenario_idx] = list of protein row indices
    buckets = {}
    for i in range(n):
        key, cls_idx, acc = proteins[i]
        if cls_idx < 0:
            continue
        cov_ok = int(cov_res[i, 0]) == int(cov_res[i, 1])
        mean_ok = int(mean_res[i, 0]) == int(mean_res[i, 1])
        s = scenario_of(cov_ok, mean_ok)
        buckets.setdefault(cls_idx, [[], [], [], []])[s].append(i)

    # ── verification table ───────────────────────────────────────────────
    print(f'cov run : {cov_run.name}')
    print(f'mean run: {mean_run.name}\n')
    hdr = f'{"class":22s}' + ''.join(f'{lab:>24s}' for lab, *_ in SCENARIOS) + '  all4'
    print(hdr)
    print('-' * len(hdr))
    for cls_idx in sorted(buckets):
        counts = [len(b) for b in buckets[cls_idx]]
        all4 = 'yes' if all(counts) else ''
        print(f'{LOCALIZATION[cls_idx]:22s}' + ''.join(f'{c:>24d}' for c in counts) + f'  {all4}')

    if args.list:
        return

    # ── choose up to 3 classes ───────────────────────────────────────────
    if args.classes:
        sel = [LOCALIZATION.index(c) for c in args.classes]
    else:
        ranked = sorted(buckets, key=lambda c: -sum(len(b) for b in buckets[c]))
        full = [c for c in ranked if all(len(b) for b in buckets[c])]
        sel = (full or ranked)[:3]
    if not sel:
        print('\nNothing to plot.')
        return
    print(f'\nPlotting classes: {[LOCALIZATION[c] for c in sel]}')

    import h5py
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({'figure.dpi': 150, 'font.size': 11})
    model = load_model(cov_run, args.dc)

    # covariance + mean per protein needed by the summary and/or detail figures
    needed = set()
    for c in sel:
        for b in buckets[c]:
            if not b:
                continue
            needed.update(b if args.reduce == 'average' else [b[0]])
            if args.detail:
                needed.update(b[:args.detail])
    needed = sorted(needed)
    print(f'Computing covariance for {len(needed)} proteins...')
    C_by, mean_by = {}, {}
    with h5py.File(args.test_h5, 'r') as h5:
        for i in needed:
            key, _, _ = proteins[i]
            C_by[i], mean_by[i] = cov_and_mean(model, h5, key, args.dc)

    if args.deviation and C_by:
        global_C = np.mean(np.stack(list(C_by.values())), axis=0)
        C_by = {i: C - global_C for i, C in C_by.items()}
        print('Subtracted global-average covariance (deviation view).')

    for cls_idx in sel:
        plot_class(cls_idx, buckets, proteins, C_by, mean_by, args)


if __name__ == '__main__':
    main()
