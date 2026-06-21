"""
Plot 4-panel UMAP/t-SNE comparison of test-protein representations.

Reads the .npy files produced by extract_features.py and projects each to 2D
using UMAP (default) or t-SNE. Computes silhouette score and 2D k-NN accuracy
per panel so the visual story is backed by numbers.

Panel order (left → right, weakest to strongest expected structure):
    1. Random Gaussian noise          — noise-floor baseline
    2. Untrained cov (random init)    — architecture-only baseline
    3. Mean pooling (1024-d)          — first-order representation
    4. Trained cov (d_c² = 2304-d)    — the headline representation

Usage:
    python scripts/analysis/plot_tsne.py --features-dir features/loc
    python scripts/analysis/plot_tsne.py --features-dir features/loc --method tsne
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def reduce_2d(X: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == 'umap':
        try:
            import umap
        except ImportError:
            sys.exit('ERROR: umap-learn not installed. `pip install umap-learn`')
        # n_jobs=1 forces single-threaded UMAP, making the output
        # byte-reproducible across runs with the same random_state.
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.10,
                            metric='cosine', random_state=seed,
                            n_jobs=1, verbose=False)
        return reducer.fit_transform(X)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        # scikit-learn >=1.5 renamed n_iter -> max_iter
        try:
            tsne = TSNE(perplexity=30, max_iter=1000, random_state=seed, init='pca')
        except TypeError:
            tsne = TSNE(perplexity=30, n_iter=1000, random_state=seed, init='pca')
        return tsne.fit_transform(X)
    else:
        raise ValueError(method)


def quality_metrics_categorical(xy: np.ndarray, labels: np.ndarray):
    """Silhouette score + leave-one-out 5-NN accuracy in the 2D projection.
    LOO excludes each query from its own neighborhood — eliminates the
    self-leakage bias that fit-then-predict-same-data would have."""
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    sil = silhouette_score(xy, labels)
    # k=6 so that after dropping the self-match we still have 5 real neighbors
    nbrs = NearestNeighbors(n_neighbors=6).fit(xy)
    _, ind = nbrs.kneighbors(xy)
    # ind[:, 0] is each point itself; use 1..5 as the actual neighbors
    neighbor_labels = labels[ind[:, 1:6]]
    # majority vote, breaking ties by first-occurrence:
    preds = np.array([np.bincount(row).argmax() for row in neighbor_labels])
    knn = float((preds == labels).mean())
    return float(sil), knn


def quality_metrics_continuous(xy: np.ndarray, targets: np.ndarray):
    """Leave-one-out 5-NN regression metrics in the 2D projection.
    Returns (5-NN MAE, 5-NN Spearman R)."""
    from sklearn.neighbors import NearestNeighbors
    # k=6 so we can drop the self-match at index 0
    nbrs = NearestNeighbors(n_neighbors=6).fit(xy)
    _, ind = nbrs.kneighbors(xy)
    
    # Average the targets of the 5 nearest neighbors (columns 1 to 5)
    neighbor_targets = targets[ind[:, 1:6]]
    preds = neighbor_targets.mean(axis=1)
    
    mae = float(np.abs(preds - targets).mean())
    if len(targets) < 2:
        rho = 0.0
    else:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(preds, targets)
        if np.isnan(rho):
            rho = 0.0
            
    return mae, float(rho)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--features-dir', type=Path, required=True,
                   help='Dir containing features_*.npy and labels.npy from extract_features.py')
    p.add_argument('--out',          type=Path, default=None,
                   help='Output PNG path (default: <features-dir>/tsne_4panel_<method>.png)')
    p.add_argument('--method',       default='umap', choices=['umap', 'tsne'])
    p.add_argument('--seed',         type=int, default=42)
    p.add_argument('--task-name',    default='DeepLoc setDeepLoc',
                   help='Used in the figure title.')
    args = p.parse_args()

    out_path = args.out or args.features_dir / f'tsne_4panel_{args.method}.png'

    # ── Load features and labels ─────────────────────────────────────
    labels      = np.load(args.features_dir / 'labels.npy')
    task_path   = args.features_dir / 'task.txt'
    task = task_path.read_text().strip() if task_path.exists() else (
        'meltome' if labels.dtype.kind == 'f' else 'loc'
    )
    is_continuous = (task == 'meltome')
    class_names = []
    if not is_continuous and (args.features_dir / 'class_names.txt').exists():
        class_names = (args.features_dir / 'class_names.txt').read_text().strip().split('\n')

    panels = [
        ('Random (Gaussian noise)',     np.load(args.features_dir / 'features_random.npy')),
        ('Untrained cov (random init)', np.load(args.features_dir / 'features_cov_untrained.npy')),
        ('Mean pooling',                np.load(args.features_dir / 'features_mean.npy')),
        ('Trained cov',                 np.load(args.features_dir / 'features_cov.npy')),
    ]

    # ── Reduce each to 2D + score ────────────────────────────────────
    print(f'Reducing 4 representations with {args.method.upper()} (task={task})…')
    embeddings_2d = []
    for name, X in panels:
        print(f'  {name:30s}  shape={X.shape}')
        xy = reduce_2d(X, args.method, args.seed)
        if is_continuous:
            mae, rho = quality_metrics_continuous(xy, labels)
            print(f'    5-NN MAE={mae:.3f}   5-NN Spearman={rho:.3f}')
            embeddings_2d.append((name, xy, mae, rho))
        else:
            sil, knn = quality_metrics_categorical(xy, labels)
            print(f'    silhouette={sil:+.3f}   2D 5-NN acc={knn:.3f}')
            embeddings_2d.append((name, xy, sil, knn))

    # ── 4-panel plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), dpi=150)
    last_sc = None

    if is_continuous:
        vmin, vmax = float(np.min(labels)), float(np.max(labels))
        cmap_cont = plt.cm.viridis
    else:
        cmap_cat = plt.cm.tab10

    for ax, (name, xy, val1, val2) in zip(axes, embeddings_2d):
        if is_continuous:
            last_sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap=cmap_cont,
                                  s=10, alpha=0.80, vmin=vmin, vmax=vmax,
                                  linewidths=0)
            ax.set_title(f'{name}\n5-NN MAE: {val1:.2f}°C  |  5-NN ρ: {val2:.3f}',
                         fontsize=11)
        else:
            for c in range(len(class_names)):
                sel = labels == c
                if sel.sum() == 0:
                    continue
                ax.scatter(xy[sel, 0], xy[sel, 1],
                           c=[cmap_cat(c % 10)], s=10, alpha=0.75,
                           label=class_names[c], linewidths=0)
            knn_fmt = f'{val2:.2%}'
            ax.set_title(f'{name}\nSilhouette: {val1:+.3f}  |  5-NN Acc: {knn_fmt}',
                         fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Legend / colorbar
    if is_continuous:
        cbar_ax = fig.add_axes((0.93, 0.20, 0.012, 0.6))
        cbar = fig.colorbar(last_sc, cax=cbar_ax)
        cbar.set_label('Tm (°C)', fontsize=11)
    else:
        handles, leg_labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, leg_labels, loc='center right',
                   bbox_to_anchor=(1.10, 0.5), fontsize=10,
                   markerscale=2.5, frameon=True, title='Class')

    fig.suptitle(f'Test-set protein representations — '
                 f'{args.method.upper()} 2D projection ({args.task_name})',
                 fontsize=15, y=1.02)
    if is_continuous:
        footer = (f'n={len(labels)} test proteins. 5-NN MAE measures local temperature consistency in °C; '
                  f'5-NN Spearman correlation (ρ) measures alignment of local predictions.')
    else:
        footer = (f'n={len(labels)} test proteins. Silhouette > 0 indicates class separation; '
                  f'5-NN accuracy in 2D quantifies cluster purity.')
    fig.text(0.005, 0.005, footer, fontsize=9, style='italic', color='#555555')
    fig.tight_layout(rect=(0, 0.03, 0.92, 0.97))
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote {out_path}')

    # ── Also dump a small text summary ───────────────────────────────
    if is_continuous:
        summary = ['representation                  5-NN MAE     5-NN Spearman']
        summary.append('-' * 60)
        for name, _, mae, rho in embeddings_2d:
            summary.append(f'{name:30s}   {mae:.4f}       {rho:.4f}')
    else:
        summary = ['representation                  silhouette   2D 5-NN acc']
        summary.append('-' * 60)
        for name, _, sil, knn in embeddings_2d:
            summary.append(f'{name:30s}   {sil:+.4f}      {knn:.4f}')
    summary_path = args.features_dir / f'tsne_metrics_{args.method}.txt'
    summary_path.write_text('\n'.join(summary) + '\n')
    print(f'wrote {summary_path}')


if __name__ == '__main__':
    main()
