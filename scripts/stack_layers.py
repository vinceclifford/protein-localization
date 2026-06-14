"""
Stack per-layer per-residue embeddings into a single concatenated h5 file.

Given per-layer h5 files (each protein has shape [L, d]), produces a new h5
where each protein has shape [L, n_layers * d] — features concatenated along
the channel dimension.

Used for the "stacked" / "last-N layers" downstream-evaluation variant
described in the project spec.

Per-task convenience: pass --task to auto-resolve all input/output paths from
the standard data_files/ layout.

Examples:
    # Stack the four pre-extracted layers for DeepLoc:
    python scripts/stack_layers.py --task loc --layers 6 12 18 24

    # Same for Meltome:
    python scripts/stack_layers.py --task meltome --layers 6 12 18 24

    # Manual mode — explicit input/output paths:
    python scripts/stack_layers.py \
        --inputs data_files/deeploc_our_train_set_layer06.h5 \
                 data_files/deeploc_our_train_set_layer12.h5 \
                 data_files/deeploc_our_train_set_layer18.h5 \
                 data_files/deeploc_our_train_set_layer24.h5 \
        --output data_files/deeploc_our_train_set_layerStacked.h5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TASK_FASTA_STEMS = {
    'loc': [
        'data_files/deeploc_our_train_set',
        'data_files/deeploc_our_val_set',
        'data_files/deeploc_test_set',
    ],
    'meltome': [
        'data_files/flip_meltome/prepared/human_cell/human_cell_train',
        'data_files/flip_meltome/prepared/human_cell/human_cell_val',
        'data_files/flip_meltome/prepared/human_cell/human_cell_test',
    ],
}


def stack_one(inputs: list[Path], output: Path) -> None:
    """Concatenate per-residue embeddings from `inputs` along the channel dim."""
    if not inputs:
        raise ValueError('no input h5 files given')
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(f'missing input h5: {p}')

    files = [h5py.File(p, 'r') for p in inputs]
    try:
        keys = list(files[0].keys())
        for i, f in enumerate(files[1:], 1):
            other = set(f.keys())
            if other != set(keys):
                missing_in_other = set(keys) - other
                extra_in_other   = other - set(keys)
                msg = f'Key mismatch between {inputs[0]} and {inputs[i]}'
                if missing_in_other:
                    msg += f'\n  missing in {inputs[i].name}: {list(missing_in_other)[:3]}…'
                if extra_in_other:
                    msg += f'\n  extra in {inputs[i].name}:   {list(extra_in_other)[:3]}…'
                raise ValueError(msg)

        # peek at one entry to confirm shapes are consistent
        sample = [f[keys[0]][:] for f in files]
        if not all(s.shape[0] == sample[0].shape[0] for s in sample):
            raise ValueError(f'sequence length mismatch on key {keys[0]}: '
                             f'{[s.shape for s in sample]}')
        in_dims = [s.shape[1] for s in sample]
        out_dim = sum(in_dims)
        in_dtype = sample[0].dtype
        print(f'  stacking {len(files)} layers, in_dims={in_dims}, '
              f'out_dim={out_dim}, dtype={in_dtype}')

        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output, 'w') as fout:
            for k in tqdm(keys, desc=f'{output.name}'):
                parts = [f[k][:] for f in files]
                stacked = np.concatenate(parts, axis=-1).astype(in_dtype)
                fout.create_dataset(k, data=stacked,
                                    compression='gzip', compression_opts=4)
        print(f'  wrote {output} ({len(keys)} entries)')
    finally:
        for f in files:
            f.close()


def auto_stack_task(task: str, layers: list[int]) -> None:
    suffix = '_'.join(f'{l:02d}' for l in sorted(layers))
    out_tag = 'Stacked' if (len(layers) == 4 and sorted(layers) == [6, 12, 18, 24]) else suffix
    for stem in TASK_FASTA_STEMS[task]:
        stem_path = PROJECT_ROOT / stem
        inputs = [stem_path.with_name(f'{stem_path.name}_layer{l:02d}.h5') for l in layers]
        output = stem_path.with_name(f'{stem_path.name}_layer{out_tag}.h5')
        print(f'\n{stem}:')
        stack_one(inputs, output)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', choices=['loc', 'meltome'], default=None,
                   help='Auto-resolve input/output paths for the given task.')
    p.add_argument('--layers', nargs='+', type=int, default=[6, 12, 18, 24],
                   help='Layers to stack (default: 6 12 18 24). Only used with --task.')
    p.add_argument('--inputs', nargs='+', type=Path, default=None,
                   help='Manual input h5 paths (overrides --task).')
    p.add_argument('--output', type=Path, default=None,
                   help='Manual output h5 path (required with --inputs).')
    args = p.parse_args()

    if args.inputs:
        if not args.output:
            sys.exit('ERROR: --output is required when --inputs is given.')
        stack_one(args.inputs, args.output)
    elif args.task:
        auto_stack_task(args.task, args.layers)
    else:
        sys.exit('ERROR: must pass either --task or --inputs/--output.')


if __name__ == '__main__':
    main()
