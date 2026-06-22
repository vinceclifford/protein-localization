"""
Determinism of the unsupervised covariance projections under PCA init.

Loads the per-seed pretrained projections from
    checkpoints/projections/compare/<dataset>_seed<seed>/cov_unsup_dc<dc>.pt
and measures how similar the learned d_c-dimensional projection *subspaces* are
across seeds: the mean cosine of the principal angles between the row-spaces of
W_L (and W_R). 1.0 = identical subspace. PCA init makes these near-deterministic,
so the projections can be trained once and reused.

Subspace cosine (not element-wise) is the right measure here — it's invariant to
the sign / in-subspace rotation ambiguity that PCA eigenvectors carry.

Usage:
    python scripts/analysis/plot_projection_determinism.py
    python scripts/analysis/plot_projection_determinism.py --dc 48
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                    if (p / 'configs').is_dir() and (p / 'models').is_dir())


def subspace_cos(Wa: np.ndarray, Wb: np.ndarray) -> float:
    """Mean cosine of principal angles between row-spaces of Wa, Wb ([d_c, d])."""
    Qa = np.linalg.qr(Wa.T)[0]                       # [d, d_c] orthonormal basis
    Qb = np.linalg.qr(Wb.T)[0]
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)   # cosines of principal angles
    return float(s.mean())


def load_projs(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return (ck['proj_L_state_dict']['weight'].numpy(),
            ck['proj_R_state_dict']['weight'].numpy())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--compare-dir', type=Path,
                   default=PROJECT_ROOT / 'checkpoints/projections/compare')
    p.add_argument('--dc', type=int, default=48)
    p.add_argument('--recon-csv', type=Path,
                   default=PROJECT_ROOT / 'results/reconstruction_summary.csv')
    p.add_argument('--out-dir', dest='out_dir', type=Path,
                   default=PROJECT_ROOT / 'results/figures/projection_consistency')
    args = p.parse_args()

    # rel_err mean +/- std per dataset (across seeds) from the reconstruction summary
    recon: dict[str, tuple] = {}
    if args.recon_csv.exists():
        by_ds: dict[str, list] = {}
        with open(args.recon_csv) as f:
            for r in csv.DictReader(f):
                by_ds.setdefault(r['dataset'], []).append(float(r['rel_err']))
        for ds, vals in by_ds.items():
            recon[ds] = (float(np.mean(vals)),
                         float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                         vals)

    # discover {dataset: {seed: (L, R)}}
    data: dict[str, dict[int, tuple]] = {}
    for d in sorted(glob.glob(str(args.compare_dir / '*_seed*'))):
        m = re.match(r'(.+)_seed(\d+)$', Path(d).name)
        if not m:
            continue
        ck = Path(d) / f'cov_unsup_dc{args.dc}.pt'
        if ck.exists():
            data.setdefault(m.group(1), {})[int(m.group(2))] = load_projs(ck)

    datasets = sorted(data)
    if not datasets:
        raise SystemExit(f'no cov_unsup_dc{args.dc}.pt checkpoints under {args.compare_dir}')

    n_ds = len(datasets)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: cross-seed subspace agreement (heatmaps) ──────────────────
    fig, axes = plt.subplots(1, n_ds, figsize=(3.8 * n_ds + 1, 4), squeeze=False)
    axes = axes[0]
    print(f'Cross-seed projection subspace agreement (L & R, d_c={args.dc}):')
    for ax, ds in zip(axes, datasets):
        seeds = sorted(data[ds])
        n = len(seeds)
        M = np.eye(n)
        for i in range(n):
            for j in range(n):
                La, Ra = data[ds][seeds[i]]
                Lb, Rb = data[ds][seeds[j]]
                M[i, j] = 0.5 * (subspace_cos(La, Lb) + subspace_cos(Ra, Rb))
        off = M[~np.eye(n, dtype=bool)]
        print(f'  {ds:8s}: off-diagonal mean {off.mean():.4f} (min {off.min():.4f})')

        im = ax.imshow(M, cmap='viridis', vmin=min(0.9, float(M.min())), vmax=1.0)
        thr = (float(M.min()) + 1.0) / 2
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{M[i, j]:.3f}', ha='center', va='center',
                        color='white' if M[i, j] < thr else 'black', fontsize=10)
        ax.set_xticks(range(n)); ax.set_xticklabels([str(s) for s in seeds], fontsize=9)
        ax.set_yticks(range(n)); ax.set_yticklabels([str(s) for s in seeds], fontsize=9)
        ax.set_xlabel('seed'); ax.set_ylabel('seed')
        ax.set_title(f'{ds}\nsubspace cos {off.mean():.3f}', fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f'Cross-seed subspace agreement of the unsupervised projections (d_c={args.dc})',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out1 = args.out_dir / 'subspace_agreement.png'
    fig.savefig(out1, bbox_inches='tight'); plt.close(fig)
    print(f'wrote {out1}')

    # ── Figure 2: reconstruction rel_err per seed (seed-invariance) ─────────
    if recon:
        fig, ax = plt.subplots(figsize=(1.7 * n_ds + 2.5, 4))
        for x, ds in enumerate(datasets):
            if ds not in recon:
                continue
            mu, sd, vals = recon[ds]
            ax.scatter([x] * len(vals), vals, color='#ff7f0e', s=34, zorder=3,
                       label='per seed' if x == 0 else None)
            ax.errorbar([x], [mu], yerr=[sd], fmt='_', color='#1f77b4', capsize=6,
                        markersize=18, zorder=2, label='mean ± std' if x == 0 else None)
            ax.annotate(f'±{sd:.4f}', (x, mu), textcoords='offset points',
                        xytext=(10, -3), fontsize=8, color='#555555')
        ax.set_xticks(range(n_ds))
        ax.set_xticklabels(datasets, rotation=20, ha='right', fontsize=10)
        ax.set_ylabel('reconstruction rel_err')
        ax.set_title(f'Reconstruction rel_err per seed (d_c={args.dc})\n'
                     '3 seeds overlap → seed-invariant', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.margins(x=0.3)
        ax.legend(fontsize=9, loc='best')
        fig.tight_layout()
        out2 = args.out_dir / 'reconstruction_relerr.png'
        fig.savefig(out2, bbox_inches='tight'); plt.close(fig)
        print(f'wrote {out2}')


if __name__ == '__main__':
    main()
