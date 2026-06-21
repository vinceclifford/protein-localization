"""
Extract per-protein representations for the t-SNE / UMAP visualisation
(visualisation #6 from the project spec).

Produces four feature matrices on the test set:
    features_random.npy           — Gaussian noise, same shape as cov
                                    (sanity-check baseline: what does noise look like?)
    features_cov_untrained.npy    — output of an UNTRAINED cov model (random init)
                                    (does training matter, or just the architecture?)
    features_mean.npy             — model-free mean pool over residues
                                    (first-order representation)
    features_cov.npy              — output of the trained cov model's bilinear pool
                                    (the headline representation)

Plus the per-protein integer class labels (for DeepLoc / SCL).

Usage (DeepLoc, last layer):
    python scripts/analysis/extract_features.py \\
        --checkpoint runs/PoolingFFN_loc_cov_dc48_seed657_<ts>/checkpoint.pt \\
        --test-h5    data_files/deeploc_test_set_layer24.h5 \\
        --test-fasta data_files/deeploc_test_set_remapped.fasta \\
        --out-dir    features/loc \\
        --proj-dim 48 --embeddings-dim 1024
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                    if (p / 'configs').is_dir() and (p / 'models').is_dir())
sys.path.insert(0, str(PROJECT_ROOT))

from models import PoolingFFN
from utils.general import LOCALIZATION


def parse_fasta_for_loc_labels(fasta_path: Path, key_format: str) -> list[tuple[str, int, str]]:
    """Return list of (h5_key, class_index, accession) per record."""
    out = []
    for rec in SeqIO.parse(str(fasta_path), 'fasta'):
        if key_format == 'hash':
            class_name = rec.description.split(' ')[2].split('-')[0]
            h5_key = str(rec.id)
            accession = rec.description.split(' ')[1] if len(rec.description.split(' ')) > 1 else rec.id
        elif key_format == 'fasta_descriptor':
            class_name = rec.description.split(' ')[1].split('-')[0]
            h5_key = str(rec.description).replace('.', '_').replace('/', '_')
            accession = rec.id
        else:  # fasta_descriptor_old
            class_name = rec.description.split(' ')[1].split('-')[0]
            h5_key = str(rec.description)
            accession = rec.id

        if class_name not in LOCALIZATION:
            continue
        out.append((h5_key, LOCALIZATION.index(class_name), accession))
    return out


def parse_fasta_for_meltome_targets(fasta_path: Path) -> list[tuple[str, float, str]]:
    """Meltome FASTA descriptions are '<idx> <accession> <Tm>' or contain 'TARGET=<Tm>'.
    Returns (h5_key, Tm_value, accession)."""
    out = []
    skipped = 0
    for rec in SeqIO.parse(str(fasta_path), 'fasta'):
        h5_key = str(rec.id)
        toks = rec.description.split()
        # Try several layouts and grab the first parseable float that looks like a Tm:
        tm = None
        accession = toks[1] if len(toks) > 1 else rec.id
        for t in toks[2:]:
            val_str = t.split('=', 1)[1] if '=' in t else t
            try:
                v = float(val_str)
                if 0 < v < 200:           # plausible Tm range
                    tm = v
                    break
            except ValueError:
                continue
        if tm is None:
            skipped += 1
            continue
        out.append((h5_key, tm, accession))
    if skipped:
        print(f'  (parse_meltome) skipped {skipped} records with no parseable Tm value')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint',     type=Path, required=True,
                   help='Path to a trained PoolingFFN cov checkpoint.pt')
    p.add_argument('--test-h5',        type=Path, required=True)
    p.add_argument('--test-fasta',     type=Path, required=True)
    p.add_argument('--out-dir',        type=Path, required=True)
    p.add_argument('--task',           default='loc', choices=['loc', 'meltome'],
                   help='loc = categorical class labels; meltome = continuous Tm values')
    p.add_argument('--key-format',     default='hash',
                   choices=['hash', 'fasta_descriptor', 'fasta_descriptor_old'],
                   help='How the FASTA descriptors map to h5 keys (loc task only)')
    p.add_argument('--proj-dim',       type=int, default=48,
                   help='d_c used by the trained cov model (must match the checkpoint)')
    p.add_argument('--embeddings-dim', type=int, default=1024,
                   help='per-residue feature dim of the input embeddings')
    p.add_argument('--output-dim',     type=int, default=None,
                   help='Model output dim (defaults to 10 for loc, 1 for meltome)')
    p.add_argument('--device',         default='cpu',
                   help='cpu / mps / cuda')
    p.add_argument('--seed',           type=int, default=42,
                   help='seed for the random baseline and untrained model init')
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    output_dim = args.output_dim if args.output_dim is not None else (1 if args.task == 'meltome' else 10)

    # ── 1. Trained cov model ──────────────────────────────────────────
    model = PoolingFFN(
        embeddings_dim=args.embeddings_dim,
        pooling='cov', proj_dim=args.proj_dim, output_dim=output_dim,
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    print(f'loaded trained model from {args.checkpoint}')

    # ── 2. Untrained cov model (architecturally identical, random init) ──
    torch.manual_seed(args.seed)
    model_untrained = PoolingFFN(
        embeddings_dim=args.embeddings_dim,
        pooling='cov', proj_dim=args.proj_dim, output_dim=output_dim,
    )
    model_untrained.eval().to(device)
    print(f'instantiated untrained model with seed={args.seed}')

    # ── 3. Parse labels from FASTA ────────────────────────────────────
    if args.task == 'meltome':
        proteins_raw = parse_fasta_for_meltome_targets(args.test_fasta)
    else:
        proteins_raw = parse_fasta_for_loc_labels(args.test_fasta, args.key_format)
    # Normalise to (key, label_value, accession). For loc, label is int class index;
    # for meltome, label is float Tm.
    proteins = proteins_raw
    n = len(proteins)
    print(f'{n} test proteins with parseable labels')

    # ── 4. Allocate output arrays ─────────────────────────────────────
    d_c, d = args.proj_dim, args.embeddings_dim
    feat_cov            = np.zeros((n, d_c * d_c), dtype=np.float32)
    feat_cov_untrained  = np.zeros((n, d_c * d_c), dtype=np.float32)
    feat_mean           = np.zeros((n, d),          dtype=np.float32)

    # Gaussian noise of same shape as cov features — the noise-floor baseline
    rng = np.random.default_rng(args.seed)
    feat_random = rng.standard_normal((n, d_c * d_c)).astype(np.float32)

    label_dtype  = np.float32 if args.task == 'meltome' else np.int64
    labels       = np.zeros(n, dtype=label_dtype)
    accessions   = []
    keep_idx     = []

    # ── 5. Iterate test set ───────────────────────────────────────────
    skipped = 0
    with h5py.File(args.test_h5, 'r') as h5, torch.no_grad():
        for i, (key, label, acc) in enumerate(tqdm(proteins, desc='extracting')):
            if key not in h5:
                skipped += 1
                continue
            arr = h5[key][:]                                          # [L, d]
            x = torch.from_numpy(arr).float().t().unsqueeze(0).to(device)  # [1, d, L]
            L = x.shape[-1]
            mask = torch.ones((1, L), dtype=torch.bool, device=device)

            # mean: model-free, just average residues
            feat_mean[i] = x.mean(dim=-1).squeeze(0).cpu().numpy()

            # trained cov: forward through _bilinear_cov only
            flat_C = model._bilinear_cov(x, mask)
            feat_cov[i] = flat_C.squeeze(0).cpu().numpy()

            # untrained cov: same call, fresh model
            flat_C_u = model_untrained._bilinear_cov(x, mask)
            feat_cov_untrained[i] = flat_C_u.squeeze(0).cpu().numpy()

            labels[i] = label
            accessions.append(acc)
            keep_idx.append(i)

    print(f'skipped {skipped} proteins not present in h5')

    # ── 6. Trim skipped rows ──────────────────────────────────────────
    keep_arr = np.array(keep_idx, dtype=np.int64)
    feat_cov           = feat_cov[keep_arr]
    feat_cov_untrained = feat_cov_untrained[keep_arr]
    feat_mean          = feat_mean[keep_arr]
    feat_random        = feat_random[keep_arr]
    labels             = labels[keep_arr]

    # ── 7. Save ───────────────────────────────────────────────────────
    np.save(args.out_dir / 'features_cov.npy',           feat_cov)
    np.save(args.out_dir / 'features_cov_untrained.npy', feat_cov_untrained)
    np.save(args.out_dir / 'features_mean.npy',          feat_mean)
    np.save(args.out_dir / 'features_random.npy',        feat_random)
    np.save(args.out_dir / 'labels.npy',                 labels)
    (args.out_dir / 'accessions.txt').write_text('\n'.join(accessions))
    (args.out_dir / 'task.txt').write_text(args.task)
    if args.task == 'loc':
        (args.out_dir / 'class_names.txt').write_text('\n'.join(LOCALIZATION))

    print(f'\nwrote {len(labels)} rows × {d_c*d_c} cov features to {args.out_dir}')
    print('files:')
    for f in sorted(args.out_dir.iterdir()):
        print(f'  {f.name:40s}  {f.stat().st_size/1e6:8.2f} MB')


if __name__ == '__main__':
    main()
